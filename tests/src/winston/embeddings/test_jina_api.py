import asyncio
import base64
import io
from dataclasses import dataclass
from collections.abc import Awaitable, Callable

import httpx
import numpy as np
import pytest
from PIL import Image

from winston.embeddings.jina.api import JinaApiEmbedder, _estimate_text_tokens
from winston.embeddings.jina.rate_limit import JinaRateLimiter
from winston.embeddings.models import EmbeddingConfigurationError, EmbeddingResponseError


@dataclass(frozen=True, slots=True)
class TinyImage:
    """Small solid-color RGB24 image for API transport tests."""

    width: int = 2
    height: int = 2
    rgb24: bytes = bytes((255, 0, 0) * 4)


class FakeClock:
    """Fake monotonic clock shared with the API limiter in deterministic tests."""

    def __init__(self) -> None:
        """Initialize zero time and recorded sleeps."""
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        """Return fake monotonic time."""
        return self.now

    async def sleep(self, seconds: float) -> None:
        """Advance fake time immediately."""
        self.sleeps.append(seconds)
        self.now += seconds


def response_for_count(count: int, *, tokens: int | None = None) -> dict[str, object]:
    """Build a valid Jina-like response with deliberately reversed data ordering."""
    return {
        "model": "jina-clip-v1",
        "object": "list",
        "usage": {"total_tokens": tokens if tokens is not None else count * 1_000},
        "data": [
            {"object": "embedding", "index": index, "embedding": [float(index)] * 768}
            for index in reversed(range(count))
        ],
    }


def test_text_token_estimate_reserves_special_token_headroom() -> None:
    """Pre-admission text estimates must not ignore tokenizer special-token overhead."""
    assert _estimate_text_tokens("a") > len("a".encode("utf-8"))


def test_api_engine_requires_key_only_when_instantiated() -> None:
    """Selecting the optional API engine without a key must fail clearly."""
    with pytest.raises(EmbeddingConfigurationError, match="JINA_API_KEY"):
        JinaApiEmbedder(api_key="")


def test_api_image_request_is_224_square_and_response_order_is_restored() -> None:
    """API transport must send canonical one-tile images and preserve original input ordering."""
    observed_sizes: list[tuple[int, int]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        """Inspect one mocked image request and return reversed provider results."""
        payload = __import__("json").loads(request.content)
        assert request.headers["Authorization"] == "Bearer test-key"
        assert payload["model"] == "jina-clip-v1"
        assert payload["normalized"] is True
        for item in payload["input"]:
            image_bytes = base64.b64decode(item["image"])
            with Image.open(io.BytesIO(image_bytes)) as image:
                observed_sizes.append(image.size)
        return httpx.Response(200, json=response_for_count(len(payload["input"])))

    async def exercise() -> None:
        """Embed two mocked images and verify transport and returned ordering."""
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        embedder = JinaApiEmbedder(api_key="test-key", client=client, batch_size=4)
        result = await embedder.embed_images([TinyImage(), TinyImage()])
        assert result.vectors.shape == (2, 768)
        assert result.vectors.dtype == np.float32
        assert result.vectors[0, 0] == 0.0
        assert result.vectors[1, 0] == 1.0
        assert observed_sizes == [(224, 224), (224, 224)]
        await embedder.close()
        await client.aclose()

    asyncio.run(exercise())


def test_api_batch_is_reduced_to_tpm_capacity() -> None:
    """Current TPM capacity must reduce an API image batch instead of waiting for the full batch."""
    request_sizes: list[int] = []
    clock = FakeClock()

    async def handler(request: httpx.Request) -> httpx.Response:
        """Record each admitted API batch size."""
        payload = __import__("json").loads(request.content)
        count = len(payload["input"])
        request_sizes.append(count)
        return httpx.Response(200, json=response_for_count(count))

    async def exercise() -> None:
        """Embed three images through a live TPM capacity of only two images."""
        limiter = JinaRateLimiter(rpm=10, tpm=2_000, clock=clock, sleep=clock.sleep)
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        embedder = JinaApiEmbedder(
            api_key="test-key",
            client=client,
            batch_size=4,
            rate_limiter=limiter,
        )
        result = await embedder.embed_images([TinyImage(), TinyImage(), TinyImage()])
        assert result.vectors.shape == (3, 768)
        assert request_sizes == [2, 1]
        assert clock.sleeps == [60.0]
        await embedder.close()
        await client.aclose()

    asyncio.run(exercise())


def test_api_reconciles_estimated_text_tokens_with_actual_usage() -> None:
    """API-reported usage must replace conservative text estimates in the limiter window."""
    clock = FakeClock()
    request_sizes: list[int] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        """Return a much smaller actual token cost than Winston reserved."""
        payload = __import__("json").loads(request.content)
        request_sizes.append(len(payload["input"]))
        return httpx.Response(200, json=response_for_count(len(payload["input"]), tokens=1))

    async def exercise() -> None:
        """Reuse reconciled TPM capacity without waiting for the minute window."""
        limiter = JinaRateLimiter(rpm=10, tpm=19, clock=clock, sleep=clock.sleep)
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        embedder = JinaApiEmbedder(api_key="test-key", client=client, batch_size=2, rate_limiter=limiter)
        await embedder.embed_texts(["1234567890"])
        await embedder.embed_texts(["1234567890"])
        assert request_sizes == [1, 1]
        assert clock.sleeps == []
        await embedder.close()
        await client.aclose()

    asyncio.run(exercise())


def test_api_429_honors_retry_after_and_retries_with_new_admission() -> None:
    """HTTP 429 must honor Retry-After and retry without bypassing the limiter."""
    attempts = 0
    retry_sleeps: list[float] = []

    async def retry_sleep(seconds: float) -> None:
        """Record provider-requested retry delays without real waiting."""
        retry_sleeps.append(seconds)

    async def handler(request: httpx.Request) -> httpx.Response:
        """Rate-limit the first request and accept the retry."""
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "2"}, text="rate limited")
        payload = __import__("json").loads(request.content)
        return httpx.Response(200, json=response_for_count(len(payload["input"]), tokens=3))

    async def exercise() -> None:
        """Embed one text and verify one bounded Retry-After retry."""
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        embedder = JinaApiEmbedder(
            api_key="test-key",
            client=client,
            rpm=100,
            tpm=100_000,
            retry_sleep=retry_sleep,
            max_retries=2,
        )
        result = await embedder.embed_texts(["red"])
        assert result.vectors.shape == (1, 768)
        assert attempts == 2
        assert retry_sleeps == [2.0]
        await embedder.close()
        await client.aclose()

    asyncio.run(exercise())


def test_api_rejects_wrong_embedding_dimension() -> None:
    """A malformed provider vector must fail before it can reach the vector index."""
    async def handler(request: httpx.Request) -> httpx.Response:
        """Return one intentionally malformed ten-dimensional embedding."""
        del request
        return httpx.Response(
            200,
            json={
                "model": "jina-clip-v1",
                "object": "list",
                "usage": {"total_tokens": 1},
                "data": [{"object": "embedding", "index": 0, "embedding": [0.0] * 10}],
            },
        )

    async def exercise() -> None:
        """Verify malformed dimensions are rejected at the provider boundary."""
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        embedder = JinaApiEmbedder(api_key="test-key", client=client)
        with pytest.raises(EmbeddingResponseError, match="embedding shape"):
            await embedder.embed_texts(["query"])
        await embedder.close()
        await client.aclose()

    asyncio.run(exercise())


class BombImage:
    """RGB image fixture that fails as soon as its pixel buffer is materialized."""

    width = 2
    height = 2

    @property
    def rgb24(self) -> bytes:
        """Raise to reveal whether later API batches were encoded eagerly."""
        raise RuntimeError("second image materialized")


def test_api_engine_encodes_only_the_current_image_batch() -> None:
    """API preprocessing must not base64-encode every source image before rate admission."""
    request_sizes: list[int] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        """Record each request that reached the provider mock."""
        payload = __import__("json").loads(request.content)
        request_sizes.append(len(payload["input"]))
        return httpx.Response(200, json=response_for_count(len(payload["input"])))

    async def exercise() -> None:
        """Prove the first request is sent before the second batch pixels are accessed."""
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        embedder = JinaApiEmbedder(api_key="test-key", client=client, batch_size=1)
        with pytest.raises(RuntimeError, match="second image materialized"):
            await embedder.embed_images([TinyImage(), BombImage()])
        assert request_sizes == [1]
        await embedder.close()
        await client.aclose()

    asyncio.run(exercise())
