"""Offline coverage for session pacing: limits, active hours, and the worker's _pace()."""

import datetime as dt
import threading
from typing import cast

import pytest

from pyclashbot.bot import worker as worker_mod
from pyclashbot.bot.worker import PACE_CONTINUE, PACE_RELAUNCH, PACE_STOP, WorkerProcess
from pyclashbot.interface.enums import StatField, UIField
from pyclashbot.utils.logger import ProcessLogger
from pyclashbot.utils.session_control import (
    SessionSettings,
    is_active_now,
    limit_reached,
    seconds_until_active,
    total_battles,
    within_active_hours,
)


def _at(hour: int, minute: int = 0) -> dt.datetime:
    return dt.datetime(2026, 10, 15, hour, minute)


# ---- pure logic --------------------------------------------------------------


def test_from_jobs_parses_and_clamps():
    s = SessionSettings.from_jobs(
        {
            UIField.STARTUP_DELAY_SECONDS.value: "99999",
            UIField.MAX_BATTLES_PER_SESSION.value: "12",
            UIField.MAX_SESSION_MINUTES.value: -5,
            UIField.SCHEDULE_TOGGLE.value: True,
            UIField.ACTIVE_START_HOUR.value: "22",
            UIField.ACTIVE_END_HOUR.value: "nonsense",
        }
    )
    assert s.startup_delay_s == 3600
    assert s.max_battles == 12
    assert s.max_minutes == 0
    assert s.schedule_enabled
    assert (s.active_start_hour, s.active_end_hour) == (22, 24)


def test_from_jobs_defaults_are_unlimited():
    s = SessionSettings.from_jobs({})
    assert limit_reached(s, {StatField.CLASSIC_1V1_FIGHTS.value: 500}, 10**6) is None
    assert is_active_now(s, _at(3))


def test_total_battles_sums_every_mode():
    stats = {
        StatField.CLASSIC_1V1_FIGHTS.value: 2,
        StatField.CLASSIC_2V2_FIGHTS.value: "3",
        StatField.TROPHY_ROAD_1V1_FIGHTS.value: 4,
        StatField.WAR_FIGHTS.value: None,
    }
    assert total_battles(stats) == 9
    assert total_battles(None) == 0


def test_limit_reached_battles_and_minutes():
    s = SessionSettings(max_battles=5, max_minutes=30)
    assert limit_reached(s, {StatField.CLASSIC_1V1_FIGHTS.value: 4}, 60) is None
    assert "5 battles" in (limit_reached(s, {StatField.CLASSIC_1V1_FIGHTS.value: 5}, 60) or "")
    assert "30 minutes" in (limit_reached(s, {}, 30 * 60) or "")


@pytest.mark.parametrize(
    ("start", "end", "hour", "expected"),
    [
        (9, 17, 9, True),
        (9, 17, 16, True),
        (9, 17, 17, False),
        (9, 17, 3, False),
        (22, 6, 23, True),  # wraps midnight
        (22, 6, 2, True),
        (22, 6, 6, False),
        (22, 6, 12, False),
        (0, 24, 5, True),  # full day
        (7, 7, 5, True),  # start == end means always
    ],
)
def test_within_active_hours(start, end, hour, expected):
    assert within_active_hours(_at(hour), start, end) is expected


def test_seconds_until_active():
    s = SessionSettings(schedule_enabled=True, active_start_hour=9, active_end_hour=17)
    assert seconds_until_active(s, _at(10)) == 0
    assert seconds_until_active(s, _at(8, 30)) == 30 * 60
    assert seconds_until_active(s, _at(20)) == 13 * 3600  # next morning


# ---- worker pacing -----------------------------------------------------------


class _FakeLogger:
    def __init__(self, stats=None):
        self.stats = stats or {}
        self.start_time = None
        self.statuses: list[str] = []

    def change_status(self, status):
        self.statuses.append(status)

    def log(self, _msg):
        pass

    def get_stats(self):
        return self.stats


class _FakeEmulator:
    def __init__(self):
        self.stopped = 0

    def stop(self):
        self.stopped += 1


def _log(stats=None) -> ProcessLogger:
    return cast("ProcessLogger", _FakeLogger(stats))


def _worker(jobs: dict, pause: threading.Event | None = None) -> WorkerProcess:
    return WorkerProcess(jobs, None, threading.Event(), "log.txt", pause)  # ty: ignore[invalid-argument-type]


def _freeze_now(monkeypatch, *moments: dt.datetime):
    seq = list(moments)

    class _DT(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return seq.pop(0) if len(seq) > 1 else seq[0]

    monkeypatch.setattr(worker_mod.dt, "datetime", _DT)


def test_pace_continues_with_no_limits():
    assert _worker({})._pace(_FakeEmulator(), _log()) == PACE_CONTINUE


def test_pace_stops_on_battle_limit():
    w = _worker({UIField.MAX_BATTLES_PER_SESSION.value: 3})
    logger = _log({StatField.CLASSIC_1V1_FIGHTS.value: 3})
    assert w._pace(_FakeEmulator(), logger) == PACE_STOP
    assert any("limit" in s for s in cast("_FakeLogger", logger).statuses)


def test_pace_stops_when_shutdown_requested_during_pause():
    pause = threading.Event()
    pause.set()
    w = _worker({}, pause)
    threading.Timer(0.05, w.shutdown_event.set).start()
    assert w._pace(_FakeEmulator(), _log()) == PACE_STOP


def test_pace_resumes_after_unpause():
    pause = threading.Event()
    pause.set()
    w = _worker({}, pause)
    threading.Timer(0.05, pause.clear).start()
    assert w._pace(_FakeEmulator(), _log()) == PACE_CONTINUE


def test_pace_closes_emulator_off_hours_then_relaunches(monkeypatch):
    jobs = {
        UIField.SCHEDULE_TOGGLE.value: True,
        UIField.ACTIVE_START_HOUR.value: 9,
        UIField.ACTIVE_END_HOUR.value: 17,
    }
    w = _worker(jobs)
    # 20:00 (off) on the first checks, then 09:00 once the wait loop polls again.
    _freeze_now(monkeypatch, _at(20), _at(20), _at(20), _at(9))
    monkeypatch.setattr(w.shutdown_event, "wait", lambda _s=None: False)
    emu = _FakeEmulator()
    assert w._pace(emu, _log()) == PACE_RELAUNCH
    assert emu.stopped == 1


def test_startup_countdown_aborts_on_stop():
    w = _worker({UIField.STARTUP_DELAY_SECONDS.value: 30})
    w.shutdown_event.set()
    assert w._startup_countdown(_log()) is False
