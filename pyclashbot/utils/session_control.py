"""Session pacing: startup delay, battle/runtime limits, and an active-hours window.

Pure logic with no emulator or GUI imports so it can be unit tested offline. The
worker (``bot/worker.py``) owns the waiting and decides what to do with the results.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pyclashbot.interface.enums import StatField, UIField

if TYPE_CHECKING:
    from collections.abc import Mapping

_BATTLE_STAT_FIELDS = (
    StatField.CLASSIC_1V1_FIGHTS,
    StatField.CLASSIC_2V2_FIGHTS,
    StatField.TROPHY_ROAD_1V1_FIGHTS,
    StatField.WAR_FIGHTS,
)


def _as_int(value: Any, default: int = 0, lo: int = 0, hi: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    parsed = max(lo, parsed)
    return min(parsed, hi) if hi is not None else parsed


@dataclass(frozen=True)
class SessionSettings:
    """User-configured session pacing. A zero limit means "no limit"."""

    startup_delay_s: int = 0
    max_battles: int = 0
    max_minutes: int = 0
    schedule_enabled: bool = False
    active_start_hour: int = 0
    active_end_hour: int = 24

    @classmethod
    def from_jobs(cls, jobs: Mapping[str, Any]) -> SessionSettings:
        return cls(
            startup_delay_s=_as_int(jobs.get(UIField.STARTUP_DELAY_SECONDS.value), hi=3600),
            max_battles=_as_int(jobs.get(UIField.MAX_BATTLES_PER_SESSION.value)),
            max_minutes=_as_int(jobs.get(UIField.MAX_SESSION_MINUTES.value)),
            schedule_enabled=bool(jobs.get(UIField.SCHEDULE_TOGGLE.value, False)),
            active_start_hour=_as_int(jobs.get(UIField.ACTIVE_START_HOUR.value), hi=23),
            active_end_hour=_as_int(jobs.get(UIField.ACTIVE_END_HOUR.value), default=24, hi=24),
        )


def total_battles(stats: Mapping[str, Any] | None) -> int:
    """Battles fought this session, across every mode the bot plays."""
    if not stats:
        return 0
    return sum(_as_int(stats.get(field.value)) for field in _BATTLE_STAT_FIELDS)


def limit_reached(settings: SessionSettings, stats: Mapping[str, Any] | None, elapsed_s: float) -> str | None:
    """Return a human-readable reason when a session limit is hit, else None."""
    if settings.max_battles and total_battles(stats) >= settings.max_battles:
        return f"played {settings.max_battles} battles"
    if settings.max_minutes and elapsed_s >= settings.max_minutes * 60:
        return f"ran for {settings.max_minutes} minutes"
    return None


def within_active_hours(now: dt.datetime, start_hour: int, end_hour: int) -> bool:
    """True when ``now`` falls inside [start_hour, end_hour); windows may wrap midnight.

    ``start == end`` (or 0..24) means always active.
    """
    if start_hour == end_hour % 24:
        return True
    hour = now.hour
    if start_hour < end_hour:
        return start_hour <= hour < end_hour
    return hour >= start_hour or hour < end_hour


def is_active_now(settings: SessionSettings, now: dt.datetime) -> bool:
    if not settings.schedule_enabled:
        return True
    return within_active_hours(now, settings.active_start_hour, settings.active_end_hour)


def seconds_until_active(settings: SessionSettings, now: dt.datetime) -> float:
    """Seconds until the active window next opens (0 when already active)."""
    if is_active_now(settings, now):
        return 0.0
    start = now.replace(hour=settings.active_start_hour, minute=0, second=0, microsecond=0)
    if start <= now:
        start += dt.timedelta(days=1)
    return (start - now).total_seconds()
