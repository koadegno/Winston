import asyncio

import pytest

from winston.embeddings.jina.rate_limit import JinaRateLimiter


class FakeClock:
    """Deterministic monotonic clock whose sleep advances time immediately."""

    def __init__(self) -> None:
        """Start the fake clock at zero seconds."""
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        """Return the current fake monotonic time."""
        return self.now

    async def sleep(self, seconds: float) -> None:
        """Record a requested delay and advance fake time without real waiting."""
        self.sleeps.append(seconds)
        self.now += seconds


def test_rate_limiter_reduces_batch_to_current_tpm_capacity() -> None:
    """The limiter must admit the largest prefix that fits the remaining token budget."""
    clock = FakeClock()

    async def exercise() -> None:
        """Reserve two batches and verify only the current TPM prefix is admitted."""
        limiter = JinaRateLimiter(rpm=10, tpm=3_000, clock=clock, sleep=clock.sleep)
        first = await limiter.reserve_up_to([1_000, 1_000])
        second = await limiter.reserve_up_to([1_000, 1_000])
        assert first.item_count == 2
        assert first.reserved_tokens == 2_000
        assert second.item_count == 1
        assert second.reserved_tokens == 1_000
        assert clock.sleeps == []

    asyncio.run(exercise())


def test_rate_limiter_waits_for_rpm_window_to_expire() -> None:
    """Requests beyond RPM capacity must wait until a rolling-window reservation expires."""
    clock = FakeClock()

    async def exercise() -> None:
        """Exhaust RPM capacity and observe one rolling-window wait."""
        limiter = JinaRateLimiter(rpm=2, tpm=10_000, clock=clock, sleep=clock.sleep)
        await limiter.reserve_up_to([100])
        await limiter.reserve_up_to([100])
        reservation = await limiter.reserve_up_to([100])
        assert reservation.item_count == 1
        assert clock.sleeps == [60.0]

    asyncio.run(exercise())


def test_rate_limiter_waits_when_even_one_item_cannot_fit_tpm() -> None:
    """TPM exhaustion must wait rather than spin or bypass the configured budget."""
    clock = FakeClock()

    async def exercise() -> None:
        """Exhaust TPM capacity and observe one rolling-window wait."""
        limiter = JinaRateLimiter(rpm=10, tpm=1_000, clock=clock, sleep=clock.sleep)
        await limiter.reserve_up_to([1_000])
        reservation = await limiter.reserve_up_to([1_000])
        assert reservation.item_count == 1
        assert clock.sleeps == [60.0]

    asyncio.run(exercise())


def test_rate_limiter_reconciliation_frees_over_reserved_tokens() -> None:
    """Actual API usage lower than the estimate must release token capacity in the current window."""
    clock = FakeClock()

    async def exercise() -> None:
        """Reconcile an overestimate and immediately reuse the released TPM capacity."""
        limiter = JinaRateLimiter(rpm=10, tpm=2_000, clock=clock, sleep=clock.sleep)
        reservation = await limiter.reserve_up_to([1_500])
        await limiter.reconcile(reservation, actual_tokens=500)
        next_reservation = await limiter.reserve_up_to([1_500])
        assert next_reservation.item_count == 1
        assert clock.sleeps == []

    asyncio.run(exercise())


def test_rate_limiter_rejects_item_larger_than_total_tpm_budget() -> None:
    """An impossible single-item reservation must fail instead of waiting forever."""
    clock = FakeClock()

    async def exercise() -> None:
        """Attempt a reservation that can never fit the configured TPM limit."""
        limiter = JinaRateLimiter(rpm=10, tpm=1_000, clock=clock, sleep=clock.sleep)
        with pytest.raises(ValueError, match="exceeds configured TPM"):
            await limiter.reserve_up_to([1_001])

    asyncio.run(exercise())
