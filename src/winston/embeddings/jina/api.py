"""Optional asynchronous Jina Embeddings API engine for Jina CLIP v1."""

import asyncio
import base64
import io
from collections.abc import Awaitable, Callable, Sequence

import httpx
import numpy as np
from pydantic import BaseModel, ValidationError

from winston.embeddings.base import RGBImage
from winston.embeddings.jina.identity import (
    JINA_CLIP_V1_DIMENSION,
    JINA_CLIP_V1_PREPROCESSING_VERSION,
)
from winston.embeddings.jina.preprocessing import prepare_jina_api_image, rgb_image_to_pil
from winston.embeddings.jina.rate_limit import JinaRateLimiter, RateReservation
from winston.embeddings.models import (
    EmbeddingBatch,
    EmbeddingConfigurationError,
    EmbeddingIdentity,
    EmbeddingInferenceError,
    EmbeddingResponseError,
    coerce_embedding_batch,
)

JINA_API_MODEL = "jina-clip-v1"
JINA_SEMANTIC_MODEL_ID = "jinaai/jina-clip-v1"
JINA_CLIP_V1_IMAGE_TOKEN_COST = 1_000
DEFAULT_API_URL = "https://api.jina.ai/v1/embeddings"


type ApiInput = dict[str, str]


class _ApiUsage(BaseModel):
    """Token usage fields returned by the Jina Embeddings API."""

    total_tokens: int


class _ApiEmbedding(BaseModel):
    """One indexed embedding item returned by the Jina Embeddings API."""

    index: int
    embedding: list[float]


class _ApiResponse(BaseModel):
    """Validated subset of a successful Jina Embeddings API response."""

    usage: _ApiUsage
    data: list[_ApiEmbedding]


def _estimate_text_tokens(text: str) -> int:
    """Estimate text tokens conservatively before API admission."""
    # A tokenizer cannot produce more content tokens than the UTF-8 byte sequence can represent,
    # but it may add model-specific special tokens. Reserve explicit headroom and reconcile the
    # estimate with Jina's authoritative usage.total_tokens after a successful response.
    return max(8, len(text.encode("utf-8")) + 8)


def _encode_api_image(image: RGBImage) -> str:
    """Convert RGB24 input to a canonical 224x224 PNG and return raw base64 bytes text."""
    prepared = prepare_jina_api_image(rgb_image_to_pil(image))
    buffer = io.BytesIO()
    prepared.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    """Return Retry-After when valid, otherwise a bounded exponential retry delay."""
    retry_after = response.headers.get("Retry-After")
    if retry_after is not None:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    return min(30.0, float(2**attempt))


class JinaApiEmbedder:
    """Rate-limited reusable async client for Jina CLIP v1 remote embeddings."""

    engine_id = "jina-api"
    backend = "remote"

    def __init__(
        self,
        *,
        api_key: str,
        api_url: str = DEFAULT_API_URL,
        batch_size: int = 16,
        rpm: int = 100,
        tpm: int = 100_000,
        max_concurrency: int = 2,
        max_retries: int = 3,
        client: httpx.AsyncClient | None = None,
        rate_limiter: JinaRateLimiter | None = None,
        retry_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Configure an API embedding session and allocate one reusable HTTP client."""
        if not api_key.strip():
            raise EmbeddingConfigurationError(
                "JINA_API_KEY is required when EMBEDDING_ENGINE=jina-api"
            )
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int) or max_concurrency <= 0:
            raise ValueError("max_concurrency must be a positive integer")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")

        self._api_key = api_key
        self._api_url = api_url
        self._batch_size = batch_size
        self._limiter = rate_limiter or JinaRateLimiter(rpm=rpm, tpm=tpm)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_retries = max_retries
        self._retry_sleep = retry_sleep
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        self._owns_client = client is None
        self._closed = False

    @property
    def identity(self) -> EmbeddingIdentity:
        """Return the semantic identity shared with the local Jina CLIP v1 engine."""
        return EmbeddingIdentity(
            model_id=JINA_SEMANTIC_MODEL_ID,
            dimension=JINA_CLIP_V1_DIMENSION,
            preprocessing_version=JINA_CLIP_V1_PREPROCESSING_VERSION,
        )

    def _ensure_open(self) -> None:
        """Reject API work after the engine session has been closed."""
        if self._closed:
            raise EmbeddingInferenceError("embedding engine is closed")

    async def _post(self, inputs: Sequence[ApiInput]) -> httpx.Response:
        """Send one admitted HTTP request while respecting the configured concurrency bound."""
        payload: dict[str, object] = {
            "model": JINA_API_MODEL,
            "input": list(inputs),
            "normalized": True,
            "embedding_type": "float",
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        async with self._semaphore:
            try:
                return await self._client.post(self._api_url, headers=headers, json=payload)
            except httpx.HTTPError as exc:
                raise EmbeddingInferenceError(f"Jina API request failed: {exc}") from exc

    def _parse_response(self, response: httpx.Response, *, expected_count: int) -> tuple[EmbeddingBatch, int]:
        """Validate response structure, restore input ordering, and return actual token usage."""
        try:
            parsed = _ApiResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise EmbeddingResponseError(f"Invalid Jina API response: {exc}") from exc

        if len(parsed.data) != expected_count:
            raise EmbeddingResponseError(
                f"Jina API returned {len(parsed.data)} embeddings for {expected_count} inputs"
            )

        ordered: list[list[float] | None] = [None] * expected_count
        for item in parsed.data:
            if item.index < 0 or item.index >= expected_count or ordered[item.index] is not None:
                raise EmbeddingResponseError(f"Invalid or duplicate Jina API embedding index: {item.index}")
            ordered[item.index] = item.embedding

        if any(vector is None for vector in ordered):
            raise EmbeddingResponseError("Jina API response omitted one or more embedding indices")

        vectors = [vector for vector in ordered if vector is not None]
        try:
            batch = coerce_embedding_batch(
                vectors,
                expected_count=expected_count,
                expected_dimension=JINA_CLIP_V1_DIMENSION,
            )
        except ValueError as exc:
            raise EmbeddingResponseError(f"Invalid Jina API embedding shape: {exc}") from exc
        return batch, parsed.usage.total_tokens

    async def _request_with_retries(
        self,
        inputs: Sequence[ApiInput],
        token_costs: Sequence[int],
        initial_reservation: RateReservation,
    ) -> EmbeddingBatch:
        """Send one logical API batch with bounded retry and fresh rate admission per retry."""
        reservation = initial_reservation
        total_estimated_tokens = sum(token_costs)

        for attempt in range(self._max_retries + 1):
            response = await self._post(inputs)
            if response.is_success:
                batch, actual_tokens = self._parse_response(response, expected_count=len(inputs))
                await self._limiter.reconcile(reservation, actual_tokens=actual_tokens)
                return batch

            retryable = response.status_code == 429 or response.status_code >= 500
            if not retryable or attempt >= self._max_retries:
                raise EmbeddingInferenceError(
                    f"Jina API returned HTTP {response.status_code}: {response.text.strip()}"
                )

            # A rejected/failed attempt keeps its conservative reservation. Every retry waits and
            # acquires a fresh RPM/TPM reservation so retries can never bypass configured limits.
            await self._retry_sleep(_retry_delay(response, attempt + 1))
            reservation = await self._limiter.reserve_up_to([total_estimated_tokens])

        raise EmbeddingInferenceError("Jina API retry loop ended unexpectedly")

    async def _embed_inputs(
        self,
        inputs: Sequence[ApiInput],
        token_costs: Sequence[int],
    ) -> EmbeddingBatch:
        """Split inputs by configured batch size and live TPM capacity while preserving order."""
        self._ensure_open()
        if len(inputs) != len(token_costs):
            raise ValueError("inputs and token_costs must have equal length")
        if not inputs:
            return coerce_embedding_batch(
                np.empty((0, JINA_CLIP_V1_DIMENSION), dtype=np.float32),
                expected_count=0,
                expected_dimension=JINA_CLIP_V1_DIMENSION,
            )

        matrices: list[np.ndarray] = []
        offset = 0
        while offset < len(inputs):
            upper = min(offset + self._batch_size, len(inputs))
            candidate_inputs = inputs[offset:upper]
            candidate_costs = token_costs[offset:upper]
            reservation = await self._limiter.reserve_up_to(candidate_costs)
            admitted_count = reservation.item_count
            admitted_inputs = candidate_inputs[:admitted_count]
            admitted_costs = candidate_costs[:admitted_count]
            batch = await self._request_with_retries(
                admitted_inputs,
                admitted_costs,
                reservation,
            )
            matrices.append(batch.vectors)
            offset += admitted_count

        return coerce_embedding_batch(
            np.concatenate(matrices, axis=0),
            expected_count=len(inputs),
            expected_dimension=JINA_CLIP_V1_DIMENSION,
        )

    async def embed_images(self, images: Sequence[RGBImage]) -> EmbeddingBatch:
        """Embed images remotely while materializing only the rate-admitted image batch."""
        self._ensure_open()
        if not images:
            return coerce_embedding_batch(
                np.empty((0, JINA_CLIP_V1_DIMENSION), dtype=np.float32),
                expected_count=0,
                expected_dimension=JINA_CLIP_V1_DIMENSION,
            )

        matrices: list[np.ndarray] = []
        offset = 0
        while offset < len(images):
            upper = min(offset + self._batch_size, len(images))
            candidate_costs = [JINA_CLIP_V1_IMAGE_TOKEN_COST] * (upper - offset)
            reservation = await self._limiter.reserve_up_to(candidate_costs)
            admitted_count = reservation.item_count
            admitted_costs = candidate_costs[:admitted_count]

            # Encode only the images that the limiter has actually admitted. This caps transient
            # Pillow/PNG/base64 memory at the live API batch instead of the whole indexing job.
            try:
                admitted_inputs = [
                    {"image": _encode_api_image(image)}
                    for image in images[offset : offset + admitted_count]
                ]
            except Exception:
                # No provider request was sent, so release this local failed reservation immediately.
                await self._limiter.reconcile(reservation, actual_tokens=0)
                raise

            batch = await self._request_with_retries(
                admitted_inputs,
                admitted_costs,
                reservation,
            )
            matrices.append(batch.vectors)
            offset += admitted_count

        return coerce_embedding_batch(
            np.concatenate(matrices, axis=0),
            expected_count=len(images),
            expected_dimension=JINA_CLIP_V1_DIMENSION,
        )

    async def embed_texts(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Embed text inputs remotely using conservative pre-admission token estimates."""
        self._ensure_open()
        inputs = [{"text": text} for text in texts]
        costs = [_estimate_text_tokens(text) for text in texts]
        return await self._embed_inputs(inputs, costs)

    async def close(self) -> None:
        """Close the API embedding session and its internally owned HTTP client exactly once."""
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await self._client.aclose()
