"""Offline coverage for the smart battle AI: enemy detection, card roles, play choice."""

import random
from pathlib import Path

import cv2
import pytest

from pyclashbot.bot import fight
from pyclashbot.bot.battle_brain import (
    COUNTER_PUSH_WINDOW_S,
    DEFENSE_COOLDOWN_S,
    ROLE_BUILDING,
    ROLE_OWN_SIDE_SPELL,
    ROLE_SPELL,
    ROLE_SUPPORT,
    ROLE_TROOP,
    ROLE_UTILITY,
    ROLE_WIN,
    BattleBrain,
    card_role,
    defensive_drop,
)
from pyclashbot.bot.coords import HAND_CARDS_COORDS, LANE_SPLIT_X, OWN_HALF_DROP_Y_RANGE, OWN_TOWER_FOOTPRINTS_LTRB
from pyclashbot.bot.find import EnemyUnit, find_enemy_units

FIXTURE = Path(__file__).parent / "fixtures" / "card_hand_bgr_regression.png"
# Hand in the fixture frame, all affordable at 5 elixir.
FIXTURE_HAND = {0: "lumberjack", 1: "ram_rider", 2: "ice_wizard", 3: "hero_bowler"}


@pytest.fixture
def frame():
    image = cv2.imread(str(FIXTURE))
    assert image is not None
    return image


class _Clock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def _brain(clock=None, seed=0) -> BattleBrain:
    return BattleBrain(clock=clock or _Clock(), rng=random.Random(seed))


# ---- enemy detection ---------------------------------------------------------


def test_finds_the_enemy_balloon_on_our_left(frame):
    units = find_enemy_units(frame)
    assert len(units) == 1
    balloon = units[0]
    assert balloon.lane == "left"
    assert 85 <= balloon.x <= 125
    assert 280 <= balloon.y <= 320


def test_no_enemies_on_an_empty_half(frame):
    frame[270:340, 60:150] = (72, 190, 160)  # paint grass over the balloon
    assert find_enemy_units(frame) == []


def test_enemy_towers_and_their_half_are_ignored(frame):
    # The enemy unit fighting near their right bridge (y~215) and the enemy tower
    # badges are above the scan band.
    assert all(u.y >= 245 for u in find_enemy_units(frame))
    assert find_enemy_units(None) == []


# ---- roles -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("card", "role"),
    [
        ("hog", ROLE_WIN),
        ("evo_royal_giant", ROLE_WIN),
        ("ram_rider", ROLE_WIN),
        ("knight", ROLE_TROOP),
        ("lumberjack", ROLE_TROOP),
        ("hero_bowler", ROLE_TROOP),
        ("ice_wizard", ROLE_SUPPORT),
        ("musketeer", ROLE_SUPPORT),
        ("cannon", ROLE_BUILDING),
        ("tombstone", ROLE_BUILDING),
        ("fireball", ROLE_SPELL),
        ("zap", ROLE_SPELL),
        ("barb_barrel", ROLE_OWN_SIDE_SPELL),
        ("mirror", ROLE_UTILITY),
        ("elixir_collector", ROLE_UTILITY),
        ("UNKNOWN", ROLE_TROOP),
        (None, ROLE_TROOP),
    ],
)
def test_card_roles(card, role):
    assert card_role(card) == role


# ---- placement ---------------------------------------------------------------


def _inside_tower(x, y):
    return any(lft <= x <= r and t <= y <= b for lft, t, r, b in OWN_TOWER_FOOTPRINTS_LTRB)


@pytest.mark.parametrize("threat", [EnemyUnit(103, 299, 50), EnemyUnit(300, 380, 50), EnemyUnit(209, 450, 50)])
@pytest.mark.parametrize("role", [ROLE_TROOP, ROLE_SUPPORT])
def test_troop_drops_stay_on_our_half_and_off_towers(threat, role):
    x, y = defensive_drop(role, threat)
    assert OWN_HALF_DROP_Y_RANGE[0] <= y <= OWN_HALF_DROP_Y_RANGE[1]
    assert not _inside_tower(x, y)
    assert (x < LANE_SPLIT_X) == (threat.lane == "left") or abs(x - LANE_SPLIT_X) < 5


def test_spell_lands_on_the_threat():
    x, y = defensive_drop(ROLE_SPELL, EnemyUnit(103, 299, 50))
    assert (x, y) == (103, 311)


# ---- decisions ---------------------------------------------------------------


def test_defends_the_balloon_with_a_troop_not_the_win_condition(frame):
    play = _brain().choose(FIXTURE_HAND, find_enemy_units(frame), elapsed_s=30)
    assert play is not None
    assert play.defensive and play.lane == "left"
    assert play.card_id in {"lumberjack", "hero_bowler"}
    assert play.coord[0] < LANE_SPLIT_X
    assert play.coord[1] >= OWN_HALF_DROP_Y_RANGE[0]


def test_empty_board_plays_the_win_condition():
    brain = _brain()
    play = brain.choose(FIXTURE_HAND, [], elapsed_s=30)
    assert play is not None
    assert play.card_id == "ram_rider"
    assert not play.defensive
    assert play.lane == brain.attack_lane


def test_holds_buildings_and_utility_unless_elixir_is_full():
    hand = {0: "cannon", 1: "mirror"}
    assert _brain().choose(hand, [], elapsed_s=30) is None
    play = _brain().choose(hand, [], elapsed_s=30, elixir_full=True)
    assert play is not None and play.card_id == "cannon"


def test_building_defends_with_a_center_pull():
    play = _brain().choose({0: "cannon", 1: "hog"}, [EnemyUnit(300, 330, 40)], elapsed_s=30)
    assert play is not None and play.card_id == "cannon" and play.defensive


def test_only_win_conditions_pressure_the_other_lane():
    play = _brain().choose({0: "hog"}, [EnemyUnit(103, 330, 40)], elapsed_s=30)
    assert play is not None and play.card_id == "hog"
    assert play.lane == "right" and not play.defensive


def test_defense_cooldown_and_deeper_push():
    clock = _Clock()
    brain = _brain(clock)
    threat = EnemyUnit(103, 300, 40)
    play = brain.choose(FIXTURE_HAND, [threat], elapsed_s=30)
    assert play is not None
    brain.record(play)

    assert brain.should_defend_now([threat]) is None  # just answered
    assert brain.should_defend_now([EnemyUnit(103, play.coord[1] + 60, 40)]) is not None  # pushed deeper
    clock.now += DEFENSE_COOLDOWN_S
    assert brain.should_defend_now([threat]) is threat


def test_counter_push_follows_the_defended_lane():
    clock = _Clock()
    brain = _brain(clock)
    brain.attack_lane = "right"
    play = brain.choose(FIXTURE_HAND, [EnemyUnit(103, 300, 40)], elapsed_s=30)
    assert play is not None
    brain.record(play)
    assert brain.offense_lane() == "left"
    clock.now += COUNTER_PUSH_WINDOW_S + 1
    assert brain.offense_lane() == "right"


def test_avoids_replaying_the_same_slot():
    brain = _brain()
    brain.recent = [0]
    play = brain.choose({0: "knight", 3: "valkyrie"}, [EnemyUnit(103, 300, 40)], elapsed_s=30)
    assert play is not None and play.index == 3


# ---- end to end through fight.py ------------------------------------------------


class _FakeEmulator:
    def __init__(self, image):
        self.image = image
        self.clicks: list[tuple[int, int]] = []

    def screenshot(self):
        return self.image

    def click(self, x, y, *_a, **_k):
        self.clicks.append((x, y))


class _FakeLogger:
    def __init__(self):
        self.cards = 0
        self.statuses: list[str] = []

    def change_status(self, status):
        self.statuses.append(status)

    def log(self, _msg):
        pass

    def add_card_played(self):
        self.cards += 1


def test_play_smart_card_defends_on_the_fixture(frame, monkeypatch):
    monkeypatch.setattr(fight.card_detection, "bridge_iar", frame)
    emulator = _FakeEmulator(frame)
    logger = _FakeLogger()
    strategy = fight.BattleStrategy()
    strategy.start_battle()

    assert fight.play_smart_card(emulator, logger, False, strategy, _brain()) is True
    hand_click, drop = emulator.clicks
    assert hand_click in (tuple(HAND_CARDS_COORDS[0]), tuple(HAND_CARDS_COORDS[3]))
    assert drop[0] < LANE_SPLIT_X and drop[1] >= OWN_HALF_DROP_Y_RANGE[0]
    assert logger.cards == 1
    assert any("defending the left lane" in s for s in logger.statuses)


def test_wait_for_elixir_stops_early_to_defend(frame, monkeypatch):
    monkeypatch.setattr(fight.card_detection, "bridge_iar", frame)
    logger = _FakeLogger()
    # Fixture has 5 elixir; waiting for 8 would normally block, but the balloon is on our side.
    assert fight.wait_for_elixir(_FakeEmulator(frame), logger, 8, brain=_brain()) is True
    assert any("Enemy push in the left lane" in s for s in logger.statuses)
