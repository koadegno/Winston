"""Rolling RPM/TPM admission control for the Jina Embeddings API."""

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

WINDOW_SECONDS = 60.0


@dataclass(slots=True)
class RateReservation:
    """One admitted API request and the token budget reserved for it."""

    timestamp: float
    item_count: int
    reserved_tokens: int


class JinaRateLimiter:
    """Coordinate configurable rolling request and token budgets across async callers."""

    def __init__(
        self,
        *,
        rpm: int,
        tpm: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Create a rolling 60-second limiter for request and token capacity."""
        if isinstance(rpm, bool) or not isinstance(rpm, int) or rpm <= 0:
            raise ValueError("rpm must be a positive integer")
        if isinstance(tpm, bool) or not isinstance(tpm, int) or tpm <= 0:
            raise ValueError("tpm must be a positive integer")
        self._rpm = rpm
        self._tpm = tpm
        self._clock = clock
        self._sleep = sleep
        self._reservations: deque[RateReservation] = deque()
        self._lock = asyncio.Lock()

    def _prune(self, now: float) -> None:
        """Discard reservations that have left the rolling one-minute window."""
        cutoff = now - WINDOW_SECONDS
        while self._reservations and self._reservations[0].timestamp <= cutoff:
            self._reservations.popleft()

    def _used_tokens(self) -> int:
        """Return currently reserved token usage inside the rolling window."""
        return sum(reservation.reserved_tokens for reservation in self._reservations)

    def _largest_prefix_that_fits(self, costs: Sequence[int], available_tokens: int) -> tuple[int, int]:
        """Return the largest leading item count and token sum that fit current capacity."""
        item_count = 0
        token_sum = 0
        for cost in costs:
            if token_sum + cost > available_tokens:
                break
            item_count += 1
            token_sum += cost
        return item_count, token_sum

    def _wait_until_next_expiry(self, now: float) -> float:
        """Return the delay until the oldest active reservation leaves the window."""
        if not self._reservations:
            # This path is only defensive; impossible single-item costs are rejected earlier.
            return WINDOW_SECONDS
        return max(0.0, self._reservations[0].timestamp + WINDOW_SECONDS - now)

    async def reserve_up_to(self, token_costs: Sequence[int]) -> RateReservation:
        """Admit the largest input prefix allowed by current RPM and TPM capacity."""
        if not token_costs:
            raise ValueError("token_costs must contain at least one item")
        if any(isinstance(cost, bool) or not isinstance(cost, int) or cost <= 0 for cost in token_costs):
            raise ValueError("token costs must be positive integers")
        if token_costs[0] > self._tpm:
            raise ValueError(
                f"single item token cost {token_costs[0]} exceeds configured TPM {self._tpm}"
            )

        while True:
            async with self._lock:
                now = self._clock()
                self._prune(now)
                request_capacity = self._rpm - len(self._reservations)
                available_tokens = self._tpm - self._used_tokens()

                if request_capacity > 0:
                    item_count, token_sum = self._largest_prefix_that_fits(
                        token_costs,
                        available_tokens,
                    )
                    if item_count > 0:
                        reservation = RateReservation(
                            timestamp=now,
                            item_count=item_count,
                            reserved_tokens=token_sum,
                        )
                        self._reservations.append(reservation)
                        return reservation

                # Release the lock while sleeping so already-running requests can reconcile usage.
                wait_seconds = self._wait_until_next_expiry(now)

            await self._sleep(wait_seconds)

    async def reconcile(self, reservation: RateReservation, *, actual_tokens: int) -> None:
        """Replace an estimate with API-reported token usage for future admissions."""
        if isinstance(actual_tokens, bool) or not isinstance(actual_tokens, int) or actual_tokens < 0:
            raise ValueError("actual_tokens must be a non-negative integer")
        async with self._lock:
            # The object remains the same instance while active in the deque, so updating it
            # immediately frees over-reserved capacity or accounts for under-estimation.
            reservation.reserved_tokens = actual_tokens
