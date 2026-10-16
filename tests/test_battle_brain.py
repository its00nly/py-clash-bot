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
    assert play.card_id != "ram_rider"
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
    assert hand_click != tuple(HAND_CARDS_COORDS[1])  # not the win condition
    assert drop[0] < LANE_SPLIT_X and drop[1] >= OWN_HALF_DROP_Y_RANGE[0]
    assert logger.cards == 1
    assert any("defending the left lane" in s for s in logger.statuses)


def test_wait_for_elixir_stops_early_to_defend(frame, monkeypatch):
    monkeypatch.setattr(fight.card_detection, "bridge_iar", frame)
    logger = _FakeLogger()
    # Fixture has 5 elixir; waiting for 8 would normally block, but the balloon is on our side.
    assert fight.wait_for_elixir(_FakeEmulator(frame), logger, 8, brain=_brain()) is True
    assert any("Enemy push in the left lane" in s for s in logger.statuses)


# ---- round 2: towers, opening, follow-up, spells ------------------------------


def _health(enemy_left=1.0, enemy_right=1.0, own_left=1.0, own_right=1.0):
    from pyclashbot.bot.find import TowerHealth

    return TowerHealth(enemy={"left": enemy_left, "right": enemy_right}, own={"left": own_left, "right": own_right})


def test_reads_full_towers_on_the_fixture(frame):
    from pyclashbot.bot.find import read_tower_health

    health = read_tower_health(frame)
    assert health is not None
    assert all(v > 0.9 for v in (*health.enemy.values(), *health.own.values()))


def test_reads_a_damaged_tower(frame):
    from pyclashbot.bot.find import read_tower_health

    frame[90:100, 122:142] = (35, 0, 117)  # darken the right half of the enemy left bar
    health = read_tower_health(frame)
    assert health is not None
    assert 0.4 < health.enemy["left"] < 0.6
    assert health.enemy["right"] > 0.9


def test_attacks_the_weaker_enemy_tower():
    brain = _brain()
    brain.attack_lane = "left"
    for _ in range(3):
        brain.observe_towers(_health(enemy_left=0.9, enemy_right=0.5))
    assert brain.offense_lane() == "right"


def test_tower_median_ignores_one_bad_read():
    brain = _brain()
    brain.attack_lane = "left"
    for hp in (1.0, 1.0, 0.0, 1.0, 1.0):  # one frame with a troop covering the bar
        brain.observe_towers(_health(enemy_right=hp))
    assert brain.offense_lane() == "left"


def test_weak_own_tower_is_defended_first():
    brain = _brain()
    for _ in range(3):
        brain.observe_towers(_health(own_right=0.2))
    deep_left, shallow_right = EnemyUnit(100, 400, 40), EnemyUnit(300, 300, 40)
    assert brain.should_defend_now([deep_left, shallow_right]) is shallow_right


def test_opening_plays_support_at_the_back():
    brain = _brain()
    play = brain.choose(FIXTURE_HAND, [], elapsed_s=8)
    assert play is not None
    assert play.card_id in {"ice_wizard", "lumberjack", "hero_bowler"}
    assert play.coord[1] >= 440
    brain.record(play)
    # After the first play, offense is normal again.
    play = brain.choose(FIXTURE_HAND, [], elapsed_s=12)
    assert play is not None and play.card_id == "ram_rider"


def test_backs_up_a_win_condition_instead_of_stacking():
    clock = _Clock()
    brain = _brain(clock)
    brain.record(brain.choose({0: "hog", 1: "musketeer"}, [], elapsed_s=40) or pytest.fail("no play"))
    clock.now += 2
    play = brain.choose({0: "balloon", 1: "musketeer"}, [], elapsed_s=42)
    assert play is not None and play.card_id == "musketeer"
    assert "backing up" in play.reason


def test_spells_saved_unless_they_finish_a_tower_or_overtime():
    hand = {0: "fireball"}
    assert _brain().choose(hand, [], elapsed_s=60) is None
    assert _brain().choose(hand, [], elapsed_s=160) is not None

    brain = _brain()
    brain.attack_lane = "left"
    for _ in range(3):
        brain.observe_towers(_health(enemy_left=0.2, enemy_right=0.2))
    play = brain.choose(hand, [], elapsed_s=60)
    assert play is not None and "finishing" in play.reason


# ---- round 3: elixir discipline, swarms, stuns, rotation, opponent pattern ----------


def test_finds_our_fighting_giant_but_not_tower_decorations(frame):
    from pyclashbot.bot.find import find_friendly_units

    friends = find_friendly_units(frame)
    assert len(friends) == 1
    assert friends[0].x > LANE_SPLIT_X and friends[0].y < 245  # our Giant on the enemy side
    assert find_friendly_units(None) == []


def test_reads_elixir_on_the_fixture(frame):
    from pyclashbot.bot.state_detect import read_elixir

    assert read_elixir(frame) == 5


def _swarm(lane_x=110, y=320):
    return [EnemyUnit(lane_x, y, 20), EnemyUnit(lane_x + 15, y + 5, 20), EnemyUnit(lane_x - 12, y + 10, 20)]


def test_swarm_gets_the_cheap_spell_the_moment_it_crosses():
    hand = {0: "knight", 1: "log", 2: "fireball", 3: "musketeer"}
    play = _brain().choose(hand, _swarm(), elapsed_s=40, elixir=6)
    assert play is not None and play.card_id == "log" and play.defensive
    assert "swarm" in play.reason


def test_push_our_troops_already_fight_is_left_alone():
    from pyclashbot.bot.battle_brain import summarize_pushes

    minion = [EnemyUnit(110, 330, 20)]
    phoenix = [EnemyUnit(120, 350, 60)]
    assert summarize_pushes(minion, phoenix)["left"].handled
    brain = _brain()
    assert brain.should_defend_now(minion, phoenix) is None
    assert brain.choose({0: "arrows", 1: "knight"}, minion, elapsed_s=40, elixir=6, friends=phoenix) is None


def test_no_swarm_cards_into_a_multi_unit_push():
    hand = {0: "skeletons", 1: "valkyrie"}
    push = [EnemyUnit(110, 320, 40), EnemyUnit(130, 300, 40)]  # e.g. Bowler + support
    play = _brain().choose(hand, push, elapsed_s=40, elixir=6)
    assert play is not None and play.card_id == "valkyrie"


def test_cheapest_answer_for_a_lone_troop():
    hand = {0: "pekka", 1: "knight", 2: "mega_knight"}
    play = _brain().choose(hand, [EnemyUnit(110, 320, 40)], elapsed_s=40, elixir=8)
    assert play is not None and play.card_id == "knight"


def test_defenders_are_not_dropped_on_top_of_the_enemy():
    threat = EnemyUnit(110, 300, 40)
    x, y = defensive_drop(ROLE_TROOP, threat)
    assert y - threat.y >= 45
    sx, _ = defensive_drop(ROLE_SUPPORT, threat)
    assert sx > threat.x  # ranged support sits toward the middle, out of lane splash


def test_stuns_a_lone_unit_hitting_our_tower():
    hand = {0: "zap", 1: "knight", 2: "hog"}
    play = _brain().choose(hand, [EnemyUnit(110, 390, 60)], elapsed_s=40, elixir=6)
    assert play is not None and play.card_id == "zap" and "stun" in play.reason


def test_keeps_a_defense_reserve_instead_of_dumping_troops():
    brain = _brain()
    brain._played_any = True
    hand = {0: "musketeer", 1: "hog", 2: "knight"}
    assert brain.choose(hand, [], elapsed_s=60, elixir=6) is None  # hog would leave 2 < reserve 3
    play = brain.choose(hand, [], elapsed_s=60, elixir=7)
    assert play is not None and play.card_id == "hog"


def test_golem_starts_at_the_back_with_full_elixir_then_gets_support():
    from pyclashbot.bot.coords import BACK_TANK_COORDS

    clock = _Clock()
    brain = _brain(clock)
    brain._played_any = True
    hand = {0: "golem", 1: "night_witch", 2: "zap", 3: "lumberjack"}
    assert brain.choose(hand, [], elapsed_s=60, elixir=7) is None  # not enough for a real push
    play = brain.choose(hand, [], elapsed_s=60, elixir=9)
    assert play is not None and play.card_id == "golem"
    assert play.coord == BACK_TANK_COORDS[play.lane]
    brain.record(play)
    clock.now += 6
    follow = brain.choose({1: "night_witch", 2: "zap", 3: "lumberjack"}, [], elapsed_s=66, elixir=5)
    assert follow is not None and follow.card_id in {"night_witch", "lumberjack"} and "backing up" in follow.reason


def test_cycles_cheap_cards_toward_the_win_condition():
    brain = _brain()
    brain._played_any = True
    brain.deck.observe_hand(["hog", "ice_spirit", "skeletons", "musketeer"])
    brain.deck.played("hog")
    play = brain.choose({0: "ice_spirit", 1: "musketeer", 2: "skeletons", 3: "fireball"}, [], elapsed_s=60, elixir=8)
    assert play is not None and play.card_id in {"ice_spirit", "skeletons"}
    assert "cycling toward hog" in play.reason


def test_deck_rotation_order():
    from pyclashbot.bot.battle_brain import DeckTracker

    deck = DeckTracker()
    for card in ("hog", "log", "knight"):
        deck.played(card)
    hand = ["log", "a", "b", "c"]
    assert deck.queue(hand) == ["hog", "knight"]
    assert deck.plays_until("hog", hand) == 1
    assert deck.plays_until("knight", hand) == 2
    assert deck.plays_until("log", hand) == 0


def test_spots_bridge_spam_and_raises_the_reserve():
    from pyclashbot.bot.battle_brain import ARCHETYPE_PRESSURE, PRESSURE_DEFENSE_RESERVE

    clock = _Clock()
    brain = _brain(clock)
    for i in range(5):
        brain.choose({0: "knight"}, [EnemyUnit(110 if i % 2 else 300, 320, 30)], elapsed_s=20 + i * 8)
        clock.now += 8
    assert brain.opponent.archetype() == ARCHETYPE_PRESSURE
    assert brain.reserve() == PRESSURE_DEFENSE_RESERVE
    assert "bridge spam" in brain.situation
