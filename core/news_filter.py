"""
News Filter — Forex Factory Economic Calendar Integration.

Fetches the weekly economic calendar from Forex Factory (via public JSON
endpoint) and detects whether a high-impact news event is imminent.
When inside the "blackout window" the live engine forces the regime to
HIGH_VOLATILITY so the Vol Assassin agent takes over before the spike.

Uses only ``urllib`` (stdlib) — no ``requests`` dependency.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Sequence

_FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
_TIMEOUT_SEC = 10

log = logging.getLogger("apex_live")


class NewsFilter:
    """Checks Forex Factory calendar for imminent high-impact events."""

    def __init__(
        self,
        currencies: Sequence[str] = ("USD",),
        blackout_minutes: int = 15,
        cache_hours: int = 4,
    ) -> None:
        self._currencies = {c.upper() for c in currencies}
        self._blackout_minutes = blackout_minutes
        self._cache_hours = cache_hours

        self._events: list[dict] = []
        self._last_fetch: datetime | None = None

    # ── Public API ────────────────────────────

    def is_blackout(self) -> tuple[bool, str]:
        """Return ``(True, event_title)`` if inside the pre-news window.

        Returns ``(False, "")`` when no imminent high-impact event.
        """
        self._refresh_if_stale()

        if not self._events:
            return False, ""

        now = datetime.now(timezone.utc)
        window = timedelta(minutes=self._blackout_minutes)

        for ev in self._events:
            ev_time = ev.get("_dt")
            if ev_time is None:
                continue

            delta = ev_time - now
            # Event is within [0, blackout_minutes] in the future
            if timedelta(0) <= delta <= window:
                title = ev.get("title", "Unknown")
                log.warning(
                    "NEWS BLACKOUT — %s in %d min → forcing HIGH_VOLATILITY",
                    title,
                    int(delta.total_seconds() / 60),
                )
                return True, title

        return False, ""

    # ── Internal ──────────────────────────────

    def _refresh_if_stale(self) -> None:
        """Re-fetch the calendar if cache has expired."""
        now = datetime.now(timezone.utc)
        if (
            self._last_fetch is not None
            and (now - self._last_fetch).total_seconds()
            < self._cache_hours * 3600
        ):
            return

        self._fetch_calendar()

    def _fetch_calendar(self) -> None:
        """Download the weekly calendar JSON and filter to high-impact events."""
        try:
            req = urllib.request.Request(
                _FF_URL,
                headers={"User-Agent": "ApexPredatorV2/1.0"},
            )
            with urllib.request.urlopen(req, timeout=_TIMEOUT_SEC) as resp:
                raw = json.loads(resp.read().decode("utf-8"))

            parsed: list[dict] = []
            for item in raw:
                impact = (item.get("impact") or "").strip().lower()
                currency = (item.get("country") or "").strip().upper()

                if impact != "high":
                    continue
                if currency not in self._currencies:
                    continue

                dt = self._parse_event_time(item)
                if dt is None:
                    continue

                item["_dt"] = dt
                parsed.append(item)

            self._events = parsed
            self._last_fetch = datetime.now(timezone.utc)
            log.info(
                "News calendar refreshed — %d high-impact events this week",
                len(parsed),
            )

        except Exception:
            log.exception("Failed to fetch Forex Factory calendar")
            # Keep stale cache if fetch fails
            if self._last_fetch is None:
                self._events = []

    @staticmethod
    def _parse_event_time(item: dict) -> datetime | None:
        """Parse the event date/time string into a UTC datetime."""
        date_str = item.get("date", "")  # e.g. "2026-02-20T08:30:00-05:00"
        if not date_str:
            return None
        try:
            dt = datetime.fromisoformat(date_str)
            return dt.astimezone(timezone.utc)
        except (ValueError, TypeError):
            return None
