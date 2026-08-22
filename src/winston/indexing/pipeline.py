"""Sequential orchestration for Winston's restartable visual indexing command."""

from collections.abc import AsyncGenerator
from itertools import batched
from pathlib import Path

from winston.config import Settings
from winston.embeddings.base import MultimodalEmbedder
from winston.embeddings.factory import create_embedder
from winston.embeddings.models import EmbeddingError
from winston.index.base import VisualIndex
from winston.index.models import IndexedVisual, RegionGeometry, SampleKind, VisualIndexError
from winston.index.qdrant import QdrantVisualIndex
from winston.indexing.manifest import IndexManifest, ManifestError, load_or_create_dataset_identity
from winston.indexing.models import AssetFailure, IndexingRunError, IndexRunResult, PipelineStage
from winston.ingest.identity import AssetIdentity, identify_asset
from winston.ingest.models import ImageMetadata, MediaFile, MediaType, VideoMetadata
from winston.ingest.probe import MediaProbeError, probe_media
from winston.ingest.scanner import scan_media
from winston.sampling.base import FrameSampler
from winston.sampling.images import ImageSamplingError, load_image
from winston.sampling.keyframes import FrameSamplingError, KeyframeSampler
from winston.sampling.regions import generate_regions


class IndexingPipeline:
    """Compose Winston media discovery, sampling, embeddings, and persistence sequentially."""

    def __init__(
        self,
        *,
        root: Path,
        settings: Settings,
        embedder: MultimodalEmbedder,
        visual_index: VisualIndex,
        frame_sampler: FrameSampler,
    ) -> None:
        """Bind one sequential indexing run to reusable embedding/storage dependencies."""
        try:
            self._root = Path(root).resolve(strict=True)
        except OSError as exc:
            raise IndexingRunError(f"indexing root does not exist: {root}") from exc
        if not self._root.is_dir():
            raise IndexingRunError(f"indexing root is not a directory: {self._root}")

        self._settings = settings
        self._embedder = embedder
        self._visual_index = visual_index
        self._frame_sampler = frame_sampler
        self._stage = PipelineStage.IDENTITY

    async def run(self) -> IndexRunResult:
        """Index every discovered asset sequentially and retain actionable per-asset failures."""
        dataset = load_or_create_dataset_identity(self._root)
        session = await self._visual_index.ensure_compatible(
            self._embedder.identity,
            dataset.dataset_instance_id,
        )
        manifest = IndexManifest(self._root)
        completed = manifest.completed_asset_ids(session.index_instance_id)
        media_files = scan_media(self._root)

        indexed = 0
        skipped = 0
        failures: list[AssetFailure] = []

        for media in media_files:
            self._stage = PipelineStage.IDENTITY
            failure_path = self._display_path(media.path)
            identity: AssetIdentity | None = None
            try:
                identity = identify_asset(index_root=self._root, source_path=media.path)
                failure_path = identity.source_path
                if identity.asset_id in completed:
                    skipped += 1
                    continue

                await self._process_media(media=media, identity=identity)

                self._stage = PipelineStage.MANIFEST
                manifest.mark_completed(
                    index_instance_id=session.index_instance_id,
                    asset_id=identity.asset_id,
                    source_path=identity.source_path,
                )
                # Keep the in-memory view current so duplicate discovery in one process is harmless too.
                completed.add(identity.asset_id)
                indexed += 1
            except (
                OSError,
                ValueError,
                MediaProbeError,
                FrameSamplingError,
                ImageSamplingError,
                EmbeddingError,
                VisualIndexError,
                ManifestError,
            ) as exc:
                failures.append(
                    AssetFailure(
                        source_path=failure_path,
                        stage=self._stage,
                        message=str(exc) or exc.__class__.__name__,
                        cause=exc,
                    )
                )

        return IndexRunResult(
            indexed=indexed,
            skipped=skipped,
            failures=tuple(failures),
        )

    async def _process_media(self, *, media: MediaFile, identity: AssetIdentity) -> None:
        """Delete stale revisions then route one current asset through its media-specific sampler."""
        self._stage = PipelineStage.DELETE
        await self._visual_index.delete_old_revisions(
            source_path=identity.source_path,
            current_asset_id=identity.asset_id,
        )

        self._stage = PipelineStage.PROBE
        metadata = probe_media(media)
        if isinstance(metadata, ImageMetadata):
            self._stage = PipelineStage.DECODE
            sample = load_image(metadata)
            await self._index_sample(
                identity=identity,
                media_type=MediaType.IMAGE,
                sample_kind=SampleKind.IMAGE,
                timestamp_seconds=None,
                width=sample.width,
                height=sample.height,
                rgb24=sample.rgb24,
            )
            return

        if isinstance(metadata, VideoMetadata):
            await self._index_video(identity=identity, metadata=metadata)
            return

        raise TypeError(f"unsupported probed media metadata: {type(metadata).__name__}")

    async def _index_video(
        self,
        *,
        identity: AssetIdentity,
        metadata: VideoMetadata,
    ) -> None:
        """Stream one video's keyframes and close its sampler promptly on downstream failure."""
        frames = self._frame_sampler.sample(metadata)
        primary: BaseException | None = None
        try:
            while True:
                # Reset this before each anext(): after a prior frame upsert, a later decoder failure
                # must still be diagnosed as sampling rather than as the previous persistence stage.
                self._stage = PipelineStage.SAMPLING
                try:
                    frame = await anext(frames)
                except StopAsyncIteration:
                    break

                await self._index_sample(
                    identity=identity,
                    media_type=MediaType.VIDEO,
                    sample_kind=SampleKind.KEYFRAME,
                    timestamp_seconds=frame.timestamp_seconds,
                    width=frame.width,
                    height=frame.height,
                    rgb24=frame.rgb24,
                )
        except BaseException as exc:
            primary = exc
            raise
        finally:
            if isinstance(frames, AsyncGenerator):
                try:
                    await frames.aclose()
                except Exception as close_error:
                    if primary is not None:
                        primary.add_note(f"frame sampler cleanup failed: {close_error}")
                    else:
                        raise

    async def _index_sample(
        self,
        *,
        identity: AssetIdentity,
        media_type: MediaType,
        sample_kind: SampleKind,
        timestamp_seconds: float | None,
        width: int,
        height: int,
        rgb24: bytes,
    ) -> None:
        """Generate, embed, and persist one photo or keyframe with bounded buffering."""
        region_batches = batched(
            generate_regions(width=width, height=height, rgb24=rgb24),
            int(self._settings.indexing.visual_batch_size),
        )

        while True:
            self._stage = PipelineStage.REGIONS
            try:
                regions = next(region_batches)
            except StopIteration:
                return

            self._stage = PipelineStage.EMBEDDING
            embedded = await self._embedder.embed_images(regions)
            if embedded.count != len(regions):
                raise ValueError(
                    f"embedder returned {embedded.count} vectors for {len(regions)} regions"
                )
            if embedded.dimension != self._embedder.identity.dimension:
                raise ValueError(
                    f"embedder returned dimension {embedded.dimension}; "
                    f"expected {self._embedder.identity.dimension}"
                )

            visuals = tuple(
                IndexedVisual(
                    asset_id=identity.asset_id,
                    source_path=identity.source_path,
                    media_type=media_type,
                    sample_kind=sample_kind,
                    timestamp_seconds=timestamp_seconds,
                    region_kind=region.region_kind,
                    region=RegionGeometry(
                        x=region.x,
                        y=region.y,
                        width=region.width,
                        height=region.height,
                        scale=region.scale,
                    ),
                    vector=embedded.vectors[position],
                    embedding_identity=self._embedder.identity,
                )
                for position, region in enumerate(regions)
            )

            self._stage = PipelineStage.UPSERT
            await self._visual_index.upsert(visuals)

    def _display_path(self, media_path: Path) -> str:
        """Return a stable relative path even when identity computation fails later."""
        try:
            return media_path.relative_to(self._root).as_posix()
        except ValueError:
            return str(media_path)


async def _close_dependencies(
    visual_index: VisualIndex,
    embedder: MultimodalEmbedder,
) -> tuple[str, ...]:
    """Close both reusable dependencies without allowing one cleanup failure to hide another."""
    errors: list[str] = []
    try:
        await visual_index.close()
    except Exception as exc:
        errors.append(f"visual index close failed: {exc}")
    try:
        await embedder.close()
    except Exception as exc:
        errors.append(f"embedder close failed: {exc}")
    return tuple(errors)


async def run_indexing(root: Path, settings: Settings) -> IndexRunResult:
    """Build one reusable indexing stack, execute it, and close resources without masking failures."""
    embedder = create_embedder(settings)
    try:
        visual_index: VisualIndex = QdrantVisualIndex(settings.qdrant)
    except BaseException as primary:
        try:
            await embedder.close()
        except Exception as cleanup_error:
            primary.add_note(f"embedder close failed: {cleanup_error}")
        raise

    try:
        pipeline = IndexingPipeline(
            root=root,
            settings=settings,
            embedder=embedder,
            visual_index=visual_index,
            frame_sampler=KeyframeSampler(),
        )
        result = await pipeline.run()
    except BaseException as primary:
        cleanup_errors = await _close_dependencies(visual_index, embedder)
        for cleanup_error in cleanup_errors:
            primary.add_note(cleanup_error)
        raise

    cleanup_errors = await _close_dependencies(visual_index, embedder)
    if cleanup_errors:
        raise IndexingRunError("; ".join(cleanup_errors))
    return result
