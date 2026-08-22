from collections.abc import AsyncIterator
from typing import Protocol

from winston.ingest.models import VideoMetadata
from winston.sampling.models import SampledFrame


class FrameSampler(Protocol):
    """Structural contract for asynchronously sampling frames from a video."""

    def sample(self, video: VideoMetadata) -> AsyncIterator[SampledFrame]:
        """Yield sampled frames without assuming a specific container format."""
        ...
