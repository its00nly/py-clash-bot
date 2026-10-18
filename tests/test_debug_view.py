"""Debug view: frame rendering, the non-blocking sink, setting plumbing and fight-loop wiring."""

from __future__ import annotations

import importlib
import json
import queue
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from pyclashbot.bot import debug_view
from pyclashbot.bot.battle_brain import Play
from pyclashbot.bot.find import EnemyUnit, TowerHealth
from pyclashbot.interface.enums import UIField

FIXTURE = Path(__file__).parent / "fixtures" / "card_hand_bgr_regression.png"


@pytest.fixture(scope="module")
def frame() -> np.ndarray:
    iar = cv2.imread(str(FIXTURE))
    assert iar is not None
    return iar


class _Brain:
    situation = "5 elixir · enemy 1 unit left"
    _hold_reason = ""

    def enemy_tower_hp(self, lane):
        return 1.0

    def own_tower_hp(self, lane):
        return 0.5 if lane == "left" else None


def _info(play: Play | None = None) -> dict[str, Any]:
    return debug_view.build_debug_info(
        elixir=5,
        available=[0, 2],
        scores=[(0, "knight", 120, 2400), (1, "UNKNOWN", 1900, 2000), (2, "fireball", 80, 1800), (3, "log", 90, 900)],
        enemies=[EnemyUnit(103, 299, 216), EnemyUnit(300, 320, 40, kind="hog_rider")],
        friends=[EnemyUnit(305, 219, 152)],
        threat=EnemyUnit(103, 299, 216),
        towers_raw=TowerHealth(enemy={"left": 1.0, "right": 0.8}, own={"left": 0.5, "right": 1.0}),
        brain=_Brain(),
        ability_ready=True,
        play=play,
        tick_ms=42.0,
        frame_age_ms=55.5,
        phase="single",
        elapsed=40.0,
    )


# ---- rendering ---------------------------------------------------------------


def test_render_returns_2x_annotated_copy(frame):
    before = frame.copy()
    play = Play(2, "fireball", "spell", (150, 384), "left", True, "defending the left lane")
    out = debug_view.render_debug_frame(frame, _info(play))
    assert out.shape == (frame.shape[0] * 2, frame.shape[1] * 2, 3)
    assert out.dtype == np.uint8
    plain = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    assert np.count_nonzero(np.any(out != plain, axis=-1)) > 5000  # visibly annotated
    assert np.array_equal(frame, before)  # input untouched


def test_render_hold_and_empty_info(frame):
    hold = debug_view.render_debug_frame(frame, _info(None))
    empty = debug_view.render_debug_frame(frame, {})
    assert hold.shape == empty.shape == (frame.shape[0] * 2, frame.shape[1] * 2, 3)
    assert not np.array_equal(hold, empty)


def test_build_info_is_json_and_marks_affordability():
    info = _info()
    json.dumps(info)  # plain JSON
    hand = {row["slot"]: row for row in info["hand"]}
    assert hand[0]["affordable"] and not hand[1]["affordable"]
    assert hand[1]["card"] == "UNKNOWN" and hand[1]["best"] == 1900
    assert info["threat"] == {"x": 103, "y": 299, "size": 216, "kind": None}
    assert info["enemies"][1]["kind"] == "hog_rider"
    assert info["towers"]["own_smoothed"] == {"left": 0.5, "right": None}
    assert info["play"] is None and info["situation"].startswith("5 elixir")


# ---- sink --------------------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _sink(tmp_path, **kw) -> tuple[debug_view.DebugFrameSink, _Clock]:
    clock = _Clock()
    sink = debug_view.DebugFrameSink(str(tmp_path), clock=clock, start_thread=False, **kw)
    return sink, clock


def test_sink_throttles_to_max_fps(tmp_path, frame):
    sink, clock = _sink(tmp_path, max_fps=5.0)
    assert sink.submit(frame, {"n": 1}) is True
    clock.now += 0.1
    assert sink.submit(frame, {"n": 2}) is False  # within 200 ms
    clock.now += 0.11
    assert sink.submit(frame, {"n": 3}) is True
    assert sink.submitted == 2


def test_sink_queue_keeps_only_the_newest_frame(tmp_path, frame):
    sink, clock = _sink(tmp_path, max_fps=0)
    for n in range(5):
        assert sink.submit(frame, {"n": n})
    assert sink.queue.qsize() == 1 and sink.dropped == 4
    assert sink.process_pending() == 1
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest["n"] == 4


def test_sink_writes_atomically(tmp_path, frame, monkeypatch):
    replaced: list[tuple[str, str]] = []
    real_replace = debug_view.os.replace

    def spy(src, dst):
        replaced.append((src, dst))
        real_replace(src, dst)

    monkeypatch.setattr(debug_view.os, "replace", spy)
    sink, _ = _sink(tmp_path, max_fps=0)
    sink.submit(frame, _info())
    assert sink.process_pending() == 1
    targets = {Path(dst).name for _src, dst in replaced}
    assert {"latest.jpg", "latest.json", "000.jpg", "000.json"} <= targets
    assert all(src.endswith(".tmp") for src, _dst in replaced)
    assert not list(tmp_path.rglob("*.tmp"))
    img = cv2.imread(str(tmp_path / "latest.jpg"))
    assert img is not None and img.shape == (frame.shape[0] * 2, frame.shape[1] * 2, 3)
    assert json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))["elixir"] == 5


def test_sink_history_ring_wraps(tmp_path, frame):
    sink, _ = _sink(tmp_path, max_fps=0, history=3)
    for n in range(5):
        sink.submit(frame, {"n": n})
        sink.process_pending()
    history = tmp_path / "history"
    assert sorted(p.name for p in history.glob("*.jpg")) == ["000.jpg", "001.jpg", "002.jpg"]
    assert json.loads((history / "000.json").read_text(encoding="utf-8"))["n"] == 3
    assert json.loads((history / "001.json").read_text(encoding="utf-8"))["n"] == 4
    assert json.loads((history / "002.json").read_text(encoding="utf-8"))["n"] == 2


def test_sink_saves_raw_frames_in_a_bounded_ring(tmp_path, frame):
    sink, _ = _sink(tmp_path, max_fps=0, history=5, raw_history=2)
    for n in range(3):
        varied = frame.copy()
        varied[0, 0] = (n, n, n)
        sink.submit(varied, {"n": n})
        sink.process_pending()
    raw = tmp_path / "raw"
    assert sorted(p.name for p in raw.glob("*.png")) == ["000.png", "001.png"]
    img = cv2.imread(str(raw / "000.png"))
    assert img is not None and img.shape == frame.shape  # native size, unannotated
    assert tuple(img[0, 0]) == (2, 2, 2) and np.array_equal(img[1:], frame[1:])  # lossless, newest wins
    meta = json.loads((tmp_path / "history" / "002.json").read_text(encoding="utf-8"))
    assert meta["raw_index"] == 0 and meta["n"] == 2


def test_sink_raw_frames_can_be_turned_off(tmp_path, frame):
    sink, _ = _sink(tmp_path, max_fps=0, raw_history=0)
    sink.submit(frame, {"n": 0})
    assert sink.process_pending() == 1
    assert not (tmp_path / "raw").exists()
    assert "raw_index" not in json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))


def test_default_raw_ring_is_60():
    assert debug_view.DEFAULT_RAW_HISTORY == 60


def test_build_info_marks_remembered_slots():
    info = debug_view.build_debug_info(
        available=[0],
        scores=[(0, "knight", 120, 2400), (1, "UNKNOWN", 1900, 2000)],
        remembered={1: "giant_skeleton"},
    )
    hand = {row["slot"]: row for row in info["hand"]}
    assert hand[1]["card"] == "giant_skeleton" and hand[1]["remembered"] and hand[1]["raw"] == "UNKNOWN"
    assert not hand[1]["affordable"] and "remembered" not in hand[0]
    debug_view.render_debug_frame(np.zeros((633, 419, 3), np.uint8), info)  # renders the (mem) label


class _AlwaysFull(queue.Queue):
    def put_nowait(self, item):
        raise queue.Full

    def put(self, item, block=True, timeout=None):
        raise AssertionError("submit must never block")


def test_sink_never_blocks_on_a_full_queue(tmp_path, frame):
    sink = debug_view.DebugFrameSink(str(tmp_path), max_fps=0, frame_queue=_AlwaysFull(maxsize=1), start_thread=False)
    assert sink.submit(frame, {}) is False
    assert sink.dropped == 1


def test_sink_swallows_errors(tmp_path, frame, monkeypatch):
    sink, _ = _sink(tmp_path, max_fps=0)
    assert sink.submit(None, {}) is False
    monkeypatch.setattr(debug_view, "render_debug_frame", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    assert sink.submit(frame, {}) is True
    assert sink.process_pending() == 0 and sink.errors == 1


def test_sink_swallows_disk_errors(tmp_path, frame):
    blocked = tmp_path / "file_not_dir"
    blocked.write_text("x")  # the debug "folder" is a file: makedirs fails
    sink = debug_view.DebugFrameSink(str(blocked), max_fps=0, start_thread=False)
    assert sink.submit(frame, {}) is True
    assert sink.process_pending() == 0 and sink.errors == 1


def test_sink_thread_writes_in_background(tmp_path, frame):
    sink = debug_view.DebugFrameSink(str(tmp_path), max_fps=0)
    try:
        assert sink.submit(frame, {"n": 1})
    finally:
        sink.close()  # drains the pending frame first
    assert (tmp_path / "latest.jpg").exists()


# ---- debug window helpers (no Tk) ----------------------------------------------


def test_window_reads_missing_and_partial_files(tmp_path):
    from pyclashbot.interface import debug_window

    assert debug_window.read_latest(str(tmp_path)) == (None, None)
    (tmp_path / "latest.json").write_text('{"elixir": 5', encoding="utf-8")  # half-written
    assert debug_window.read_info(str(tmp_path)) is None
    assert "Waiting for frames" in debug_window.format_info(None)
    text = debug_window.format_info(_info(), now=_info()["t"] + 10)
    assert "stale" in text and "Elixir: 5" in text and "knight" in text and "Threat" in text


# ---- setting plumbing --------------------------------------------------------


def test_setting_key_default_and_persisted():
    from pyclashbot.interface import config

    assert UIField.DEBUG_VIEW_USER_TOGGLE.value == "debug_view_user_toggle"
    assert UIField.DEBUG_VIEW_USER_TOGGLE.value in config.USER_CONFIG_KEYS
    job = next(job for job in config.JOBS if job.key == UIField.DEBUG_VIEW_USER_TOGGLE)
    assert job.default is False


def test_job_dictionary_maps_debug_view(monkeypatch):
    from pyclashbot.utils import logger as logger_mod

    monkeypatch.setattr(logger_mod, "initialize_pylogging", lambda: None)
    main = importlib.import_module("pyclashbot.__main__")
    key = UIField.DEBUG_VIEW_USER_TOGGLE.value
    assert main.make_job_dictionary({})[key] is False
    assert main.make_job_dictionary({key: True})[key] is True


class _NullLogger:
    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None


@pytest.mark.parametrize("state", ["1v1_fight", "2v2_fight"])
def test_states_pass_debug_view_to_the_fight(monkeypatch, state):
    from pyclashbot.bot import states

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(states, "do_fight_state", lambda *a, **kw: calls.append(kw) or True)
    monkeypatch.setattr(states.time, "sleep", lambda _s: None)
    monkeypatch.setattr(states, "mode_used_in_1v1", "Classic 2v2" if state == "2v2_fight" else "Classic 1v1")
    logger: Any = _NullLogger()
    jobs = {UIField.DEBUG_VIEW_USER_TOGGLE.value: True}
    states.state_tree(None, logger, state, jobs, states.StateHistory(logger), states.StateOrder())
    states.state_tree(None, logger, state, {}, states.StateHistory(logger), states.StateOrder())
    assert [c["debug_view"] for c in calls] == [True, False]


def test_do_fight_state_forwards_debug_view(monkeypatch):
    from pyclashbot.bot import fight

    seen: list[dict] = []
    monkeypatch.setattr(fight, "wait_for_battle_start", lambda *a: True)
    monkeypatch.setattr(fight, "_fight_loop", lambda *a, **kw: seen.append(kw) or False)
    logger: Any = _NullLogger()
    fight.do_fight_state(None, logger, False, "Classic 1v1", smart_battle=True, debug_view=True)
    assert seen == [{"debug_view": True}]
