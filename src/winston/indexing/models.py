"""Structured results and failure stages for restartable indexing."""

from dataclasses import dataclass, field
from enum import StrEnum


class IndexingRunError(RuntimeError):
    """Raised for fatal initialization or cleanup failures that invalidate a whole run."""


class PipelineStage(StrEnum):
    """Asset stage retained in actionable failure diagnostics."""

    IDENTITY = "identity"
    DELETE = "delete"
    PROBE = "probe"
    DECODE = "decode"
    SAMPLING = "sampling"
    REGIONS = "regions"
    EMBEDDING = "embedding"
    UPSERT = "upsert"
    MANIFEST = "manifest"


@dataclass(frozen=True, slots=True)
class AssetFailure:
    """One failed media asset with its original exception preserved."""

    source_path: str
    stage: PipelineStage
    message: str
    cause: Exception = field(compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class IndexRunResult:
    """Structured outcome of one complete sequential indexing run."""

    indexed: int
    skipped: int
    failures: tuple[AssetFailure, ...]

    @property
    def failed(self) -> int:
        """Return the number of failed assets."""
        return len(self.failures)

    @property
    def exit_code(self) -> int:
        """Return zero only when every discovered asset succeeded or was skipped."""
        return 1 if self.failures else 0
