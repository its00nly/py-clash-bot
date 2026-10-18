"""Fight-loop timing with a fake emulator and a fake clock (no ADB, no real sleeps)."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from pyclashbot.bot import fight, state_detect
from pyclashbot.bot.battle_brain import Play
from pyclashbot.bot.coords import CHAMPION_ABILITY_DISMISS_COORD, HAND_CARDS_COORDS
from pyclashbot.bot.find import EnemyUnit

SCREENCAP_S = 0.38
HAND = {0: "knight", 1: "musketeer", 2: "fireball", 3: "skeletons"}
DROP = (100, 400)
ABILITY = tuple(CHAMPION_ABILITY_DISMISS_COORD)


# ---- fakes -------------------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += max(0.0, seconds)


@dataclass
class _Spec:
    in_battle: bool = True
    elixir: int = 5
    ability: bool = False
    slots: tuple[int, ...] = (0, 1, 2, 3)
    shade: int = 0  # paints the hand slots so a "new card" changes the fingerprint


def _frame(spec: _Spec) -> np.ndarray:
    iar = np.zeros((633, 419, 3), dtype=np.uint8)
    if spec.shade:
        for x, y in fight.card_detection.toplefts:
            iar[y : y + 66, x : x + 54] = spec.shade
    if spec.in_battle:
        for y, x in [(528, 49), (532, 77), (546, 52), (546, 77)]:
            iar[y, x] = (255, 255, 255)
        iar[618, 115] = (220, 50, 220)
    for y, x in state_detect.ELIXIR_COORDS[: spec.elixir]:
        iar[y, x] = state_detect.ELIXIR_COLOR
    for slot in spec.slots:
        x, y = fight.card_detection.card_toplefts[slot]
        iar[y : y + 20, x : x + 20] = fight.card_detection.purple_color
    if spec.ability:
        iar[462, 324] = (215, 28, 223)
    return iar


class _Emulator:
    """screenshot() costs SCREENCAP_S of fake time; ``script(n)`` gives the n-th frame's spec."""

    def __init__(self, clock: _Clock, script) -> None:
        self.clock = clock
        self.script = script
        self.shots = 0
        self.shot_starts: list[float] = []
        self.clicks: list[tuple[int, int]] = []
        self.click_times: list[float] = []

    def screenshot(self):
        spec = self.script(self.shots)
        self.shot_starts.append(self.clock.now)
        self.shots += 1
        self.clock.now += SCREENCAP_S
        return _frame(spec)

    def click(self, x, y, *_a, **_k):
        self.clicks.append((x, y))
        self.click_times.append(self.clock.now)
        self.clock.now += 0.05


class _Logger:
    def __init__(self) -> None:
        self.statuses: list[str] = []
        self.logs: list[str] = []
        self.cards = 0

    def change_status(self, status):
        self.statuses.append(status)

    def log(self, msg):
        self.logs.append(msg)

    def add_card_played(self):
        self.cards += 1

    def get_cards_played(self):
        return self.cards


class _BrainBase:
    def __init__(self, emulator: _Emulator, play: bool = False, threat: EnemyUnit | None = None, budget: int = 0):
        self.emulator = emulator
        self.play = play
        self.threat = threat
        self.budget = budget
        self.attack_lane = "left"
        self.situation = "test"
        self.choose_calls: list[dict] = []
        self.records: list[tuple[Play, bool]] = []
        self.noted: list[EnemyUnit] = []

    def should_defend_now(self, threats, friends=()):
        return self.threat

    def note_threat(self, threat, now):
        self.noted.append(threat)

    def defense_budget(self, threat):
        return self.budget

    def observe_towers(self, health):
        pass

    def choose(self, hand, threats, elapsed_s, elixir_full=False, elixir=None, friends=(), clock=None):
        self.choose_calls.append(
            {"hand": dict(hand), "elixir": elixir, "clock": clock, "shot": self.emulator.shots - 1}
        )
        if not self.play or not hand:
            return None
        index = min(hand)
        return Play(index, hand[index], "troop", DROP, "left", False, "test play")


class _Brain(_BrainBase):
    def record(self, play, deployed=True):
        self.records.append((play, deployed))


class _LegacyBrain(_BrainBase):
    def record(self, play):  # no deployed kwarg
        self.records.append((play, True))


@pytest.fixture
def env(monkeypatch, tmp_path):
    clock = _Clock()
    monkeypatch.setattr(fight.time, "time", clock.time)
    monkeypatch.setattr(fight.time, "monotonic", clock.time)
    monkeypatch.setattr(fight.time, "sleep", clock.sleep)
    monkeypatch.setattr(fight, "DECISION_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(fight, "create_default_bridge_iar", lambda _e: None)
    monkeypatch.setattr(fight, "_configure_hand_crops", lambda *_a, **_k: None)
    monkeypatch.setattr(fight.card_detection, "bridge_iar", None)
    monkeypatch.setattr(fight, "find_enemy_units", lambda iar: [])
    monkeypatch.setattr(fight, "find_friendly_units", lambda iar: [])
    monkeypatch.setattr(fight, "read_tower_health", lambda iar: None)
    monkeypatch.setattr(fight, "classify_units", lambda iar, units: list(units))
    monkeypatch.setattr(fight, "identify_hand_cards", lambda _e, slot: HAND[slot])
    monkeypatch.setattr(
        fight.card_detection,
        "identify_hand_cards_scored",
        lambda iar=None, slots=(0, 1, 2, 3): [(s, HAND[s], 0, 9999) for s in slots],
        raising=False,
    )
    ended_checks: list[float] = []

    def ended(_e):
        ended_checks.append(clock.now)
        return True

    monkeypatch.setattr(fight, "check_if_battle_has_ended", ended)
    return clock, ended_checks, tmp_path


def _run_smart(monkeypatch, emulator, brain) -> bool:
    monkeypatch.setattr(fight, "BattleBrain", lambda: brain)
    logger: Any = _Logger()
    return fight._fight_loop(emulator, logger, False, "Classic 1v1", None, smart_battle=True)


# ---- one frame per tick -------------------------------------------------------


def test_one_screenshot_per_tick_while_holding(env, monkeypatch):
    clock, ended_checks, log_dir = env
    in_battle = 20  # 7.6s of holds: crosses the 5s preamble backstop too
    emulator = _Emulator(clock, lambda n: _Spec(in_battle=n < in_battle))
    brain = _Brain(emulator)

    assert _run_smart(monkeypatch, emulator, brain) is True
    assert len(brain.choose_calls) == in_battle
    assert [c["shot"] for c in brain.choose_calls] == list(range(in_battle))  # one decision per frame
    assert len(ended_checks) == 1
    assert 0.5 not in clock.sleeps  # no hold-path sleep
    lines = [json.loads(line) for f in log_dir.glob("*.jsonl") for line in f.read_text().splitlines()]
    holds = [entry for entry in lines if "play" in entry]
    # An unchanged hold is logged at most once a second, not every tick.
    assert 1 <= len(holds) < in_battle
    assert all(b["t"] - a["t"] >= fight.DECISION_LOG_MIN_GAP_S for a, b in itertools.pairwise(holds))
    assert lines[-1]["event"] == "end" and lines[-1]["frames"] == emulator.shots


def test_match_clock_is_passed_to_choose(env, monkeypatch):
    clock, _, _ = env
    emulator = _Emulator(clock, lambda n: _Spec(in_battle=n < 3))
    brain = _Brain(emulator)
    _run_smart(monkeypatch, emulator, brain)
    clocks = [c["clock"] for c in brain.choose_calls]
    assert clocks[0] == 0.0 and clocks[1] == pytest.approx(SCREENCAP_S)


# ---- battle-end debounce ------------------------------------------------------


def test_single_lost_frame_does_not_end_the_fight(env, monkeypatch):
    clock, ended_checks, _ = env
    emulator = _Emulator(clock, lambda n: _Spec(in_battle=n < 5 or 6 <= n < 11))
    brain = _Brain(emulator)

    _run_smart(monkeypatch, emulator, brain)
    assert len(brain.choose_calls) == 10
    assert len(ended_checks) == 1
    lost_start = emulator.shot_starts[11]
    assert ended_checks[0] - lost_start >= fight.BATTLE_END_DEBOUNCE_S - SCREENCAP_S


def test_lost_frames_are_never_tapped(env, monkeypatch):
    clock, _, _ = env
    emulator = _Emulator(clock, lambda n: _Spec(in_battle=n < 2, ability=True))
    brain = _Brain(emulator, play=False)
    _run_smart(monkeypatch, emulator, brain)
    assert emulator.clicks == []


# ---- Hero/Champion ability ----------------------------------------------------


def _assert_pairs_intact(clicks):
    hand = {tuple(c) for c in HAND_CARDS_COORDS}
    for i, click in enumerate(clicks):
        if click in hand:
            assert clicks[i + 1] == DROP  # nothing between the two taps of a play


def test_ability_fires_once_after_fallback_and_never_inside_a_play(env, monkeypatch):
    clock, _, _ = env
    start = clock.now
    # Ability lit for ~5s of fake time (fallback is 4s, cooldown 5s), then the battle ends.
    emulator = _Emulator(clock, lambda n: _Spec(in_battle=clock.now - start < 5.0, ability=True))
    brain = _Brain(emulator, play=True)

    _run_smart(monkeypatch, emulator, brain)
    assert emulator.clicks.count(ABILITY) == 1
    _assert_pairs_intact(emulator.clicks)
    fired_at = emulator.click_times[emulator.clicks.index(ABILITY)]
    assert fired_at - emulator.shot_starts[0] >= fight.ABILITY_FALLBACK_S


def test_ability_fires_early_when_an_enemy_is_on_our_half(env, monkeypatch):
    clock, _, _ = env
    monkeypatch.setattr(fight, "find_enemy_units", lambda iar: [EnemyUnit(100, 300, 30)])
    emulator = _Emulator(clock, lambda n: _Spec(in_battle=n < 8, ability=True))
    brain = _Brain(emulator)

    _run_smart(monkeypatch, emulator, brain)
    assert emulator.clicks.count(ABILITY) == 1
    fired_at = emulator.click_times[emulator.clicks.index(ABILITY)]
    assert fight.ABILITY_READY_S <= fired_at - emulator.shot_starts[0] < fight.ABILITY_FALLBACK_S


def test_ability_flickering_every_frame_never_fires(env, monkeypatch):
    clock, _, _ = env
    emulator = _Emulator(clock, lambda n: _Spec(in_battle=n < 30, ability=n % 2 == 0))
    _run_smart(monkeypatch, emulator, _Brain(emulator))
    assert ABILITY not in emulator.clicks


def test_random_loop_fires_ability_once_while_waiting(env, monkeypatch):
    clock, _, _ = env
    emulator = _Emulator(clock, lambda n: _Spec(in_battle=n < 12, elixir=0, slots=(), ability=True))
    assert fight._random_fight_loop(emulator, _Logger()) is True
    assert emulator.clicks == [ABILITY]


# ---- settle guard + play verification -----------------------------------------


def test_no_decision_on_a_frame_before_the_play_settled(env, monkeypatch):
    clock, _, _ = env
    emulator = _Emulator(clock, lambda n: _Spec(in_battle=n < 15))
    brain = _Brain(emulator, play=True)

    _run_smart(monkeypatch, emulator, brain)
    drops = [t for c, t in zip(emulator.clicks, emulator.click_times, strict=True) if c == DROP]
    assert len(drops) >= 2
    for call in brain.choose_calls[1:]:
        shot_start = emulator.shot_starts[call["shot"]]
        last_tap = max(t for t in drops if t < shot_start) + 0.05  # tap returns after the click
        assert shot_start >= last_tap + fight.PLAY_SETTLE_S - 1e-9


def test_play_that_did_not_deploy_is_recorded_as_such(env, monkeypatch):
    clock, _, _ = env
    emulator = _Emulator(clock, lambda n: _Spec(in_battle=n < 4))  # hand and elixir never change
    brain = _Brain(emulator, play=True)

    _run_smart(monkeypatch, emulator, brain)
    assert brain.records and all(deployed is False for _, deployed in brain.records)


def test_play_that_did_not_deploy_is_not_recorded_by_a_legacy_brain(env, monkeypatch):
    clock, _, _ = env
    emulator = _Emulator(clock, lambda n: _Spec(in_battle=n < 4))
    brain = _LegacyBrain(emulator, play=True)

    _run_smart(monkeypatch, emulator, brain)
    assert brain.choose_calls and brain.records == []


def test_play_verified_by_elixir_drop(env, monkeypatch):
    clock, _, _ = env
    emulator = _Emulator(clock, lambda n: _Spec(in_battle=n < 3, elixir=5 if n == 0 else 2))
    brain = _Brain(emulator, play=True)

    _run_smart(monkeypatch, emulator, brain)
    assert brain.records[0][1] is True


def test_play_verified_by_slot_change(env, monkeypatch):
    clock, _, _ = env
    emulator = _Emulator(clock, lambda n: _Spec(in_battle=n < 3, shade=0 if n == 0 else 200))
    brain = _Brain(emulator, play=True)
    # The hand is read by identify_hand_cards_scored; the verify-time re-read sees a new card.
    monkeypatch.setattr(fight, "identify_hand_cards", lambda _e, slot: "hog_rider")

    _run_smart(monkeypatch, emulator, brain)
    assert brain.records[0][1] is True


def test_raised_card_that_did_not_deploy_is_not_a_slot_change(env, monkeypatch):
    clock, _, _ = env
    x, y = fight.card_detection.toplefts[0]
    before = np.zeros((633, 419, 3), dtype=np.uint8)
    before[y + 10 : y + 40, x + 5 : x + 45] = (40, 200, 90)  # some card art
    raised = np.zeros_like(before)
    raised[y + 2 : y + 32, x + 5 : x + 45] = (40, 200, 90)  # same card, 8px higher
    fingerprint = fight._slot_fingerprint(before, 0)
    assert float(np.mean(np.abs(fight._slot_fingerprint(raised, 0) - fingerprint))) >= fight.SLOT_CHANGE_THRESHOLD
    assert not fight._slot_changed(raised, 0, fingerprint)
    other = np.zeros_like(before)
    other[y : y + 66, x : x + 54] = (200, 30, 200)  # a different card arrived
    assert fight._slot_changed(other, 0, fingerprint)


def test_resolved_unknown_that_stays_in_hand_is_not_deployed(env, monkeypatch):
    clock, _, _ = env
    brain = _Brain(_Emulator(clock, lambda n: _Spec()))
    ctx = fight.FightContext(brain)
    before = _frame(_Spec(shade=0))
    play = Play(0, "knight", "troop", DROP, "left", False, "test play")  # brain resolved UNKNOWN -> knight
    ctx.pending = fight.PendingPlay(play, 0, "knight", 5, fight._slot_fingerprint(before, 0), clock.now, "UNKNOWN")
    monkeypatch.setattr(fight, "identify_hand_cards", lambda _e, slot: "UNKNOWN")  # still the unreadable card
    after = fight.TickFrame(_frame(_Spec(shade=200)), clock.now + 1, True, 5, [0, 1, 2, 3])
    logger = _Logger()

    ctx._verify(logger, after)
    assert brain.records == [(play, False)]
    assert ctx.slot_blocked(0, after.captured_at + 0.5)
    assert not ctx.slot_blocked(0, after.captured_at + fight.NOT_DEPLOYED_BACKOFF_S)


def test_resolved_unknown_with_elixir_drop_is_deployed(env, monkeypatch):
    clock, _, _ = env
    brain = _Brain(_Emulator(clock, lambda n: _Spec()))
    ctx = fight.FightContext(brain)
    frame = _frame(_Spec())
    play = Play(0, "knight", "troop", DROP, "left", False, "test play")
    ctx.pending = fight.PendingPlay(play, 0, "knight", 5, fight._slot_fingerprint(frame, 0), clock.now, "UNKNOWN")
    monkeypatch.setattr(fight, "identify_hand_cards", lambda _e, slot: "UNKNOWN")

    ctx._verify(_Logger(), fight.TickFrame(frame, clock.now + 1, True, 2, [0, 1, 2, 3]))
    assert brain.records == [(play, True)]


def test_failed_play_backs_off_that_slot(env, monkeypatch):
    clock, _, _ = env
    emulator = _Emulator(clock, lambda n: _Spec(in_battle=n < 30))  # hand and elixir never change
    brain = _Brain(emulator, play=True)
    logger: Any = _Logger()
    monkeypatch.setattr(fight, "BattleBrain", lambda: brain)

    fight._fight_loop(emulator, logger, False, "Classic 1v1", None, smart_battle=True)
    slot0 = [t for c, t in zip(emulator.clicks, emulator.click_times, strict=True) if c == tuple(HAND_CARDS_COORDS[0])]
    assert len(slot0) >= 2
    assert all(b - a >= fight.NOT_DEPLOYED_BACKOFF_S for a, b in itertools.pairwise(slot0))
    # While slot 0 backs off the brain is offered the other slots instead.
    assert any(0 not in call["hand"] and call["hand"] for call in brain.choose_calls)
    not_deployed = [m for m in logger.logs if m.startswith("Play not deployed")]
    assert len(not_deployed) < len(brain.records)


def test_holding_status_is_throttled(env, monkeypatch):
    clock, _, _ = env
    emulator = _Emulator(clock, lambda n: _Spec(in_battle=n < 20))
    brain = _Brain(emulator)
    logger: Any = _Logger()
    monkeypatch.setattr(fight, "BattleBrain", lambda: brain)

    fight._fight_loop(emulator, logger, False, "Classic 1v1", None, smart_battle=True)
    holding = [s for s in logger.statuses if s.startswith("Holding")]
    assert 1 <= len(holding) <= 9  # ~7.6s of holds, at most one line a second


def test_decision_log_prunes_old_files(tmp_path):
    for i in range(40):
        (tmp_path / f"2026-01-01_00-00-{i:02d}.jsonl").write_text("{}\n")
    log = fight.DecisionLog(str(tmp_path))
    log.write({"x": 1})
    files = sorted(tmp_path.glob("*.jsonl"))
    assert len(files) == fight.DECISION_LOG_KEEP
    assert not (tmp_path / "2026-01-01_00-00-00.jsonl").exists()


def test_hand_crops_follow_the_record_fights_setting(monkeypatch, tmp_path):
    from pyclashbot.bot import hand_crop_sink

    calls = []
    monkeypatch.setattr(fight, "_hand_crop_setting", None)
    monkeypatch.setattr(hand_crop_sink, "configure", lambda root=None, enabled=True: calls.append((root, enabled)))
    monkeypatch.setattr(hand_crop_sink, "default_root", lambda custom_path=None: tmp_path / "hand_crops")

    fight._configure_hand_crops(False)
    fight._configure_hand_crops(False)  # unchanged: no new sink
    fight._configure_hand_crops(True, "custom")
    assert calls == [(None, False), (tmp_path / "hand_crops", True)]


def test_uses_tap_sequence_when_available(env, monkeypatch):
    clock, _, _ = env
    emulator = _Emulator(clock, lambda n: _Spec(in_battle=n < 2))
    sequences = []
    setattr(emulator, "tap_sequence", lambda points: sequences.append(list(points)))
    _run_smart(monkeypatch, emulator, _Brain(emulator, play=True))
    assert sequences[0] == [tuple(HAND_CARDS_COORDS[0]), DROP]
    assert emulator.clicks == []


# ---- defense elixir gating ----------------------------------------------------


def test_wait_keeps_waiting_for_the_defense_budget(env, monkeypatch):
    clock, _, _ = env
    threat = EnemyUnit(100, 300, 30)
    monkeypatch.setattr(fight, "find_enemy_units", lambda iar: [threat])
    emulator = _Emulator(clock, lambda n: _Spec(elixir=2 if n < 3 else 4))
    brain: Any = _Brain(emulator, threat=threat, budget=4)

    assert fight.wait_for_elixir(emulator, _Logger(), 8, brain=brain) is True
    assert emulator.shots == 4
    assert brain.noted and brain.noted[0] is threat


def test_wait_answers_a_deep_threat_immediately(env, monkeypatch):
    clock, _, _ = env
    threat = EnemyUnit(100, 380, 30)
    monkeypatch.setattr(fight, "find_enemy_units", lambda iar: [threat])
    emulator = _Emulator(clock, lambda n: _Spec(elixir=2))
    brain: Any = _Brain(emulator, threat=threat, budget=4)

    assert fight.wait_for_elixir(emulator, _Logger(), 8, brain=brain) is True
    assert emulator.shots == 1


# ---- debug view ----------------------------------------------------------------


class _FakeSink:
    def __init__(self) -> None:
        self.frames: list[tuple[Any, dict]] = []

    def submit(self, iar, info) -> bool:
        self.frames.append((iar, info))
        return True


@pytest.mark.parametrize("enabled", [True, False])
def test_fight_loop_publishes_debug_frames_only_when_enabled(env, monkeypatch, enabled):
    clock, _, _ = env
    sink = _FakeSink()
    monkeypatch.setattr(fight.debug_frames, "get_sink", lambda: sink)
    emulator = _Emulator(clock, lambda n: _Spec(in_battle=n < 12))
    brain = _Brain(emulator, play=True)
    monkeypatch.setattr(fight, "BattleBrain", lambda: brain)
    logger: Any = _Logger()

    fight._fight_loop(emulator, logger, False, "Classic 1v1", None, smart_battle=True, debug_view=enabled)
    if not enabled:
        assert sink.frames == []
        return
    assert sink.frames
    shots = emulator.shots
    assert len(sink.frames) <= shots  # never more than one publish per screenshot (no extra captures)
    _iar, info = sink.frames[0]
    assert info["elixir"] == 5 and info["play"]["card"] == "knight"
    assert [row["card"] for row in info["hand"]] == list(HAND.values())


def test_debug_view_errors_never_reach_the_fight(env, monkeypatch):
    clock, _, _ = env

    class _Boom:
        def submit(self, iar, info):
            raise RuntimeError("disk on fire")

    monkeypatch.setattr(fight.debug_frames, "get_sink", lambda: _Boom())
    emulator = _Emulator(clock, lambda n: _Spec(in_battle=n < 12))
    brain = _Brain(emulator, play=True)
    monkeypatch.setattr(fight, "BattleBrain", lambda: brain)
    logger: Any = _Logger()
    assert fight._fight_loop(emulator, logger, False, "Classic 1v1", None, smart_battle=True, debug_view=True)
    assert brain.records  # plays still happened


class _StopLoopError(Exception):
    pass


@pytest.mark.parametrize(("smart", "expected"), [(False, None), (True, "sink")])
def test_debug_view_only_for_smart_battles(env, monkeypatch, smart, expected):
    seen: list[Any] = []

    def ctx_factory(brain, decision_log, debug_sink=None):
        seen.append(debug_sink)
        raise _StopLoopError

    monkeypatch.setattr(fight.debug_frames, "get_sink", lambda: "sink")
    monkeypatch.setattr(fight, "FightContext", ctx_factory)
    monkeypatch.setattr(fight, "BattleBrain", lambda: _Brain(_Emulator(env[0], lambda n: _Spec())))
    logger: Any = _Logger()
    with pytest.raises(_StopLoopError):
        fight._fight_loop(None, logger, False, "Classic 1v1", None, smart_battle=smart, debug_view=True)
    assert seen == [expected]


# ---- hand slot memory ---------------------------------------------------------


class _FullHandBrain(_Brain):
    """A brain that takes the full 4-slot hand and the affordable set."""

    def choose(
        self, hand, threats, elapsed_s, elixir_full=False, elixir=None, friends=(), clock=None,
        full_hand=None, affordable=None,
    ):  # fmt: skip
        self.choose_calls.append({"hand": dict(hand), "full_hand": dict(full_hand or {}), "affordable": affordable})


def test_remembered_card_fills_a_greyed_slot_but_stays_unaffordable(env, monkeypatch):
    clock, _, log_dir = env
    # Shot 0: slot 0 reads giant_skeleton (affordable). Later: it greys out (unaffordable) and reads UNKNOWN.
    emulator = _Emulator(clock, lambda n: _Spec(in_battle=n < 4, slots=(0, 1, 2, 3) if n == 0 else (1, 2, 3)))

    def scored(iar=None, slots=(0, 1, 2, 3)):
        first = emulator.shots <= 1
        return [(s, ("giant_skeleton" if first else "UNKNOWN") if s == 0 else HAND[s], 0, 9999) for s in slots]

    monkeypatch.setattr(fight.card_detection, "identify_hand_cards_scored", scored, raising=False)
    brain = _FullHandBrain(emulator)

    _run_smart(monkeypatch, emulator, brain)
    later = brain.choose_calls[-1]
    assert later["full_hand"][0] == "giant_skeleton"  # remembered
    assert 0 not in later["hand"] and 0 not in later["affordable"]  # affordability unchanged
    lines = [json.loads(line) for f in log_dir.glob("*.jsonl") for line in f.read_text().splitlines()]
    slot0 = [line["hand"]["0"] for line in lines if "hand" in line]
    assert {"card": "giant_skeleton", "affordable": False, "remembered": True} in slot0
    assert slot0[0] == {"card": "giant_skeleton", "affordable": True}  # a live read isn't flagged


def test_verified_deploy_clears_the_slot_memory(env, monkeypatch):
    clock, _, _ = env
    brain = _Brain(_Emulator(clock, lambda n: _Spec()))
    ctx = fight.FightContext(brain)
    ctx.remember_hand({0: "giant_skeleton", 1: "arrows"}, None)
    assert ctx.slot_memory == {0: "giant_skeleton", 1: "arrows"}
    frame = _frame(_Spec())
    play = Play(0, "giant_skeleton", "troop", DROP, "left", False, "test play")
    ctx.pending = fight.PendingPlay(play, 0, "giant_skeleton", 8, fight._slot_fingerprint(frame, 0), clock.now)

    ctx._verify(_Logger(), fight.TickFrame(frame, clock.now + 1, True, 2, [0, 1, 2, 3]))
    assert brain.records == [(play, True)]
    assert ctx.slot_memory == {1: "arrows"}
    hand, _full, remembered = ctx.remember_hand({0: "UNKNOWN", 1: "UNKNOWN"}, None)
    assert hand == {0: "UNKNOWN", 1: "arrows"} and remembered == {1: "arrows"}


def test_stale_read_of_the_card_just_played_is_not_remembered(env):
    clock, _, _ = env
    ctx = fight.FightContext(_Brain(_Emulator(clock, lambda n: _Spec())))
    ctx.slot_last_deployed[0] = "giant_skeleton"
    ctx.remember_hand({0: "giant_skeleton"}, None)  # the frame still shows the played card
    assert 0 not in ctx.slot_memory
    ctx.remember_hand({0: "evo_lumberjack"}, None)  # the new card from 'Next'
    assert ctx.slot_memory == {0: "evo_lumberjack"}


# ---- learned card art -----------------------------------------------------------

LEARN_DECK = ["balloon", "arrows", "furnace", "giant_skeleton", "ice_spirit", "lumberjack", "skeletons", "snowball"]


class _DeckBrain(_Brain):
    def __init__(self, emulator):
        super().__init__(emulator)
        from pyclashbot.bot.battle_brain import DeckTracker

        self.deck = DeckTracker("hero_balloon")
        for card in LEARN_DECK[1:]:
            self.deck.observe_hand([card])


def _art(seed: int, nudge: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hists = np.stack([np.bincount(rng.integers(0, 16, 891), minlength=16) for _ in range(4)]).astype(np.int32)
    hists[:, 0] += nudge
    hists[:, 1] -= min(nudge, int(hists[:, 1].min()))
    return hists


def _learning_play(ctx, clock, card, elixir_before, art, **extra):
    frame = _frame(_Spec())
    play = Play(0, card, "troop", DROP, "left", False, "test play")
    ctx.pending = fight.PendingPlay(
        play, 0, card, elixir_before, fight._slot_fingerprint(frame, 0), clock.now, "UNKNOWN",
        hist_before=art, chroma_before=80.0, **extra,
    )  # fmt: skip
    return frame


def test_exact_elixir_drop_learns_the_hero_card_once(env):
    clock, _, _ = env
    brain = _DeckBrain(_Emulator(clock, lambda n: _Spec()))
    assert sorted(brain.deck.deck) == sorted(LEARN_DECK)
    ctx = fight.FightContext(brain)
    logger = _Logger()
    pool = frozenset(LEARN_DECK) - {"arrows", "snowball", "ice_spirit"}  # the other three slots
    for nudge in (0, 20, 40):
        frame = _learning_play(ctx, clock, "hero_balloon", 7, _art(1, nudge), learn_pool=pool)
        ctx._verify(logger, fight.TickFrame(frame, clock.now + 1, True, 2, [0, 1, 2, 3]))  # dropped exactly 5
    assert fight.card_detection.learned_store().is_active("hero_balloon")
    assert logger.logs.count("Learned card art for hero_balloon") == 1


def test_ambiguous_elixir_drop_learns_nothing(env):
    clock, _, _ = env
    ctx = fight.FightContext(_DeckBrain(_Emulator(clock, lambda n: _Spec())))
    pool = frozenset(LEARN_DECK) - {"arrows", "snowball", "ice_spirit"}  # furnace and lumberjack both cost 4
    for nudge in (0, 20):
        frame = _learning_play(ctx, clock, "evo_furnace", 6, _art(2, nudge), learn_pool=pool)
        ctx._verify(_Logger(), fight.TickFrame(frame, clock.now + 1, True, 2, [0, 1, 2, 3]))
    assert fight.card_detection.learned_store().samples == {}


def test_rotation_resolve_learns_without_an_elixir_match(env):
    clock, _, _ = env
    ctx = fight.FightContext(_DeckBrain(_Emulator(clock, lambda n: _Spec())))
    logger = _Logger()
    for nudge in (0, 20):
        frame = _learning_play(ctx, clock, "evo_lumberjack", 6, _art(3, nudge), rotation_label="lumberjack")
        ctx._verify(logger, fight.TickFrame(frame, clock.now + 1, True, 3, [0, 1, 2, 3]))  # drop 3, not 4
    assert fight.card_detection.learned_store().is_active("evo_lumberjack")
    assert "Learned card art for evo_lumberjack" in logger.logs


def test_rotation_guess_contradicted_by_elixir_learns_nothing(env):
    clock, _, _ = env
    ctx = fight.FightContext(_DeckBrain(_Emulator(clock, lambda n: _Spec())))
    for nudge in (0, 20):  # rotation says ice_spirit (1) but 5 elixir went: it was the Balloon
        frame = _learning_play(ctx, clock, "ice_spirit", 7, _art(3, nudge), rotation_label="ice_spirit")
        ctx._verify(_Logger(), fight.TickFrame(frame, clock.now + 1, True, 2, [0, 1, 2, 3]))
    assert fight.card_detection.learned_store().samples == {}


def test_learning_context_skips_greyed_and_known_slots(env):
    clock, _, _ = env
    ctx = fight.FightContext(_DeckBrain(_Emulator(clock, lambda n: _Spec())))
    grey = np.full((633, 419, 3), 128, dtype=np.uint8)
    assert ctx.learning_context(grey, 0, "UNKNOWN", "hero_balloon", {0: "UNKNOWN"}) == {}
    colour = np.zeros((633, 419, 3), dtype=np.uint8)
    colour[..., 2] = 220  # saturated red everywhere
    assert ctx.learning_context(colour, 0, "arrows", "arrows", {0: "arrows"}) == {}  # read fine: nothing to learn
    context = ctx.learning_context(colour, 0, "UNKNOWN", "hero_balloon", {0: "UNKNOWN", 1: "arrows"})
    assert context["hist_before"].shape == (4, 16) and context["chroma_before"] >= 20
    assert context["learn_pool"] == frozenset(LEARN_DECK) - {"arrows"}
