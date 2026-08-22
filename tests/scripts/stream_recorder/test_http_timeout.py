"""Regression tests for strict wall-clock HTTP discovery deadlines."""

from __future__ import annotations

import asyncio
import time

import pytest

from scripts.stream_recorder.discovery import fetch_text


@pytest.mark.asyncio
async def test_http_total_deadline_does_not_wait_for_cancellation_resistant_client() -> None:
    """A stuck async HTTP transport must not keep the workbook HTTP stage alive indefinitely."""

    class Client:
        """Model an async HTTP client whose request suppresses cancellation briefly."""

        async def get(self, _url: str):
            """Ignore cancellation long enough to expose cooperative-timeout bugs."""
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                await asyncio.sleep(0.20)
            raise AssertionError("request should have been abandoned at the hard deadline")

    started = time.monotonic()

    with pytest.raises(TimeoutError):
        await fetch_text(
            "https://example.test/hanging",
            client=Client(),
            total_timeout_seconds=0.03,
        )

    assert time.monotonic() - started < 0.12
