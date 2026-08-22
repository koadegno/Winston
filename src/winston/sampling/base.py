from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from winston.ingest.models import VideoMetadata
from winston.sampling.models import SampledFrame


class FrameSampler(ABC):
    """Strategy interface for asynchronously sampling frames from a video."""

    @abstractmethod
    def sample(self, video: VideoMetadata) -> AsyncIterator[SampledFrame]:
        """Yield sampled frames without assuming a specific container format."""
        raise NotImplementedError
