"""Scenario tests for the battle brain with the user's real deck.

A port of the offline decision harness: each scenario runs over every 4-card hand from the
deck and several seeds, and a hand-written "strong player" oracle judges each decision.
The BAD-rate ceilings are what the brain must stay under. The rest are targeted rules
(coordinated Giant Skeleton push, spell discipline, threat memory, defense sizing,
counter-push gating, match phases, tower reads).
"""

from __future__ import annotations

import itertools
import random

import pytest

from pyclashbot.bot import battle_brain as bb
from pyclashbot.bot.battle_brain import BattleBrain, MatchClock, Play, card_cost, match_phase
from pyclashbot.bot.coords import BACK_TANK_COORDS, ENEMY_PRINCESS_TOWER_CENTERS
from pyclashbot.bot.find import EnemyUnit, TowerHealth

DECK = ["arrows", "evo_furnace", "giant_skeleton", "ice_spirit", "lumberjack", "skeletons", "snowball", "UNKNOWN"]
HANDS = list(itertools.combinations(DECK, 4))
SEEDS = range(3)
GOOD, OK, BAD = "GOOD", "OK", "BAD"
L_BRIDGE, R_BRIDGE = 117, 300
CROSS_Y = 295


class Clock:
    def __init__(self, t: float = 10_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def unit(x, y, size=40, kind=None):
    return EnemyUnit(x, y, size, kind)


def affordable(hand4, elixir):
    return {i: c for i, c in enumerate(hand4) if elixir is None or card_cost(c) <= elixir}


def _health(enemy=(1.0, 1.0), own=(1.0, 1.0)):
    return TowerHealth(enemy={"left": enemy[0], "right": enemy[1]}, own={"left": own[0], "right": own[1]})


def _brain(clock=None, seed=0, played=True) -> BattleBrain:
    brain = BattleBrain(clock=clock or Clock(), rng=random.Random(seed))
    brain._played_any = played
    return brain


def tiered(tiers):
    def judge(hand, play):
        if play is None:
            return BAD if any(tiers.get(c) == GOOD for c in hand) else OK
        return tiers.get(play.card_id, BAD)

    return judge


def bad_rate(threats, elixir, judge, elapsed=60.0, setup=None) -> float:
    bad = total = 0
    for hand4 in HANDS:
        for seed in SEEDS:
            brain = _brain(seed=seed)
            if setup:
                setup(brain)
            hand = affordable(hand4, elixir)
            play = brain.choose(hand, threats, elapsed, elixir=elixir)
            total += 1
            bad += judge(list(hand.values()), play) == BAD
    return bad / total


# ---- harness scenarios ------------------------------------------------------------------

HOG = {"lumberjack": GOOD, "evo_furnace": GOOD, "ice_spirit": GOOD, "skeletons": GOOD, "giant_skeleton": OK}
HORDE = [unit(L_BRIDGE + dx, CROSS_Y + dy, 20) for dx, dy in ((-15, 0), (0, -8), (15, 0), (-10, 14), (10, 14), (0, 22))]
GOLEM = [unit(L_BRIDGE, 300, 70), unit(L_BRIDGE - 20, 262, 40), unit(L_BRIDGE + 25, 268, 40)]
BARBS = [unit(R_BRIDGE - 8, 335, 40), unit(R_BRIDGE + 10, 338, 40), unit(R_BRIDGE - 20, 345, 35)]
GRAVEYARD = [unit(100, 340, 14), unit(135, 345, 14), unit(150, 372, 14), unit(95, 392, 14), unit(140, 400, 14)]


def _full_judge(hand, play):
    if play is None:
        return BAD
    if play.card_id in ("giant_skeleton", "evo_furnace"):
        return GOOD
    if play.card_id == "lumberjack" and "giant_skeleton" in hand:
        return OK
    if play.card_id in ("ice_spirit", "skeletons") and play.coord[1] >= 380:
        return OK
    return BAD


def _no_chip_judge(hand, play):
    return BAD if play is not None and play.card_id in ("arrows", "snowball") else OK


@pytest.mark.parametrize(
    ("name", "threats", "elixir", "judge", "elapsed", "ceiling"),
    [
        ("hog_alone_bridge", [unit(L_BRIDGE, CROSS_Y, 45)], 5, tiered(HOG), 60, 0.05),
        ("hog_alone_at_tower", [unit(L_BRIDGE, 340, 45)], 5, tiered(HOG), 60, 0.05),
        (
            "minion_horde",
            HORDE,
            6,
            tiered({"arrows": GOOD, "evo_furnace": OK, "ice_spirit": OK}),
            60,
            0.10,
        ),
        (
            "golem_nw_babyd",
            GOLEM,
            7,
            tiered(
                {"giant_skeleton": GOOD, "lumberjack": GOOD, "evo_furnace": GOOD, "skeletons": OK, "ice_spirit": OK}
            ),
            90,
            0.20,
        ),
        (
            "ram_broke_2barbs_bandit",
            BARBS,
            5,
            tiered({"lumberjack": GOOD, "giant_skeleton": GOOD, "evo_furnace": OK, "ice_spirit": OK, "skeletons": OK}),
            60,
            0.10,
        ),
        (
            "graveyard_keeps_spelling",
            GRAVEYARD,
            6,
            tiered(
                {
                    "arrows": GOOD,
                    "evo_furnace": GOOD,
                    "giant_skeleton": GOOD,
                    "snowball": OK,
                    "lumberjack": OK,
                    "ice_spirit": OK,
                }
            ),
            100,
            0.05,
        ),
        (
            "balloon_at_tower",
            [unit(L_BRIDGE, 360, 50)],
            5,
            tiered({"evo_furnace": GOOD, "ice_spirit": GOOD, "arrows": OK, "snowball": OK}),
            60,
            0.40,  # without unit recognition a lone marker could be anything
        ),
        ("full_elixir_empty_board", [], 10, _full_judge, 60, 0.10),
        ("nine_elixir_empty_board", [], 9, _full_judge, 60, 0.10),
        ("double_elixir_empty_board", [], 6, _no_chip_judge, 160, 0.0),
        ("double_elixir_empty_board_9", [], 9, _full_judge, 160, 0.10),
    ],
)
def test_scenario_bad_rate(name, threats, elixir, judge, elapsed, ceiling):
    assert bad_rate(threats, elixir, judge, elapsed) <= ceiling, name


def test_recognized_horde_gets_arrows_not_snowball():
    horde = [unit(u.x, u.y, u.size, "minion_horde") for u in HORDE]
    judge = tiered({"arrows": GOOD, "evo_furnace": OK, "ice_spirit": OK})
    assert bad_rate(horde, 6, judge) <= 0.10


def test_recognized_golem_push_is_not_spelled():
    kinds = [
        unit(GOLEM[0].x, 300, 70, "golem"),
        unit(GOLEM[1].x, 262, 40, "night_witch"),
        unit(GOLEM[2].x, 268, 40, "baby_dragon"),
    ]
    judge = tiered({"giant_skeleton": GOOD, "lumberjack": GOOD, "evo_furnace": GOOD, "skeletons": OK, "ice_spirit": OK})
    assert bad_rate(kinds, 7, judge, elapsed=90) <= 0.20


def test_lone_skeleton_never_gets_a_spell():
    # Ignoring it would be best (step 12, needs unit recognition); a spell is never right.
    for threats in (
        [unit(L_BRIDGE, CROSS_Y, 12)],
        [unit(L_BRIDGE + dx, CROSS_Y + dy, 12) for dx, dy in ((-10, 0), (10, 0), (0, 12))],
    ):
        for hand4 in HANDS:
            play = _brain().choose(affordable(hand4, 5), threats, 60, elixir=5)
            assert play is None or play.card_id not in ("arrows", "snowball"), hand4


# ---- coordinated Giant Skeleton push ------------------------------------------------------


def _learned_brain(clock, seed=0) -> BattleBrain:
    brain = _brain(clock, seed)
    brain.deck.observe_hand(DECK)
    return brain


def test_giant_skeleton_starts_at_the_back_then_gets_a_follow_up_behind_it():
    clock = Clock()
    brain = _learned_brain(clock)
    hand = {0: "giant_skeleton", 1: "lumberjack", 2: "skeletons", 3: "ice_spirit"}
    play = brain.choose(hand, [], 60, elixir=9)
    assert play is not None and play.card_id == "giant_skeleton"
    assert play.coord == BACK_TANK_COORDS[play.lane] and play.role == bb.ROLE_WIN
    brain.record(play)
    lane = play.lane
    rest = {1: "lumberjack", 2: "skeletons", 3: "ice_spirit"}
    follow = None
    for _ in range(12):
        clock.t += 1.0
        follow = brain.choose(rest, [], 60, elixir=7)
        if follow is not None:
            break
    assert follow is not None and clock.t - 10_000.0 >= 6.0
    assert follow.lane == lane and "backing up" in follow.reason
    assert follow.coord[1] < BACK_TANK_COORDS[lane][1] - 60  # up the lane, behind the walking tank


def test_follow_up_goes_early_once_the_tank_reaches_the_bridge():
    clock = Clock()
    brain = _learned_brain(clock)
    play = brain.choose({0: "giant_skeleton", 1: "lumberjack"}, [], 60, elixir=9)
    assert play is not None
    brain.record(play)
    clock.t += 3.0
    tank = unit(BACK_TANK_COORDS[play.lane][0], 320, 60)
    follow = brain.choose({1: "lumberjack"}, [], 60, elixir=7, friends=[tank])
    assert follow is not None and follow.card_id == "lumberjack"
    assert follow.coord[1] > tank.y  # just behind our tank


def test_giant_skeleton_still_defends():
    hand = {0: "giant_skeleton", 1: "arrows", 2: "snowball", 3: "skeletons"}
    push = [unit(110, 320, 45), unit(128, 305, 40)]
    for seed in SEEDS:
        play = _learned_brain(Clock(), seed).choose(hand, push, 60, elixir=8)
        assert play is not None and play.card_id == "giant_skeleton" and play.defensive


# ---- spell discipline ---------------------------------------------------------------------


def test_no_chip_in_double_elixir_or_overtime():
    hand = {0: "arrows", 1: "lumberjack", 2: "giant_skeleton", 3: "UNKNOWN"}
    assert _brain().choose(affordable(hand.values(), 3), [], 185, elixir=3) is None
    assert _brain().choose({0: "arrows"}, [], 160, elixir=10) is None  # the only swarm answer


def test_arrows_finish_only_a_nearly_dead_tower():
    for hp, finishes in ((0.03, True), (0.2, False)):
        brain = _brain()
        for _ in range(3):
            brain.observe_towers(_health(enemy=(1.0, hp)))
        play = brain.choose({0: "arrows"}, [], 60, elixir=5)
        if finishes:
            assert play is not None and play.coord == ENEMY_PRINCESS_TOWER_CENTERS["right"]
            assert "finishing" in play.reason
        else:
            assert play is None


def test_destroyed_enemy_tower_gets_no_spell():
    for elapsed in (60, 200, 290):
        brain = _brain()
        for _ in range(3):
            brain.observe_towers(_health(enemy=(0.0, 1.0)))
        play = brain.choose({0: "snowball", 1: "fireball"}, [], elapsed, elixir=10)
        assert play is None or play.coord not in {(118, 185), ENEMY_PRINCESS_TOWER_CENTERS["left"]}


def test_overtime_tiebreak_chip_in_the_last_seconds():
    brain = _brain()
    for _ in range(3):
        brain.observe_towers(_health(enemy=(0.6, 0.5)))
    assert brain.choose({0: "fireball"}, [], 250, elixir=6) is None
    play = brain.choose({0: "fireball"}, [], 290, elixir=6)
    assert play is not None and play.coord == ENEMY_PRINCESS_TOWER_CENTERS["right"]


def test_overtime_threat_gets_no_chip():
    brain = _brain()
    for _ in range(3):
        brain.observe_towers(_health(enemy=(0.6, 0.5)))
    play = brain.choose({0: "arrows", 1: "giant_skeleton"}, [unit(L_BRIDGE, CROSS_Y, 45)], 290, elixir=9)
    assert play is None or play.defensive


def test_six_markers_get_arrows_three_markers_no_spell():
    hand = {0: "arrows", 1: "snowball", 2: "lumberjack", 3: "giant_skeleton"}
    six = [unit(110 + dx, 300 + dy, 20) for dx, dy in ((0, 0), (15, 0), (-15, 0), (0, 15), (12, 15), (-12, 15))]
    play = _brain().choose(hand, six, 60, elixir=8)
    assert play is not None and play.card_id == "arrows"
    three = six[:3]
    play = _brain().choose(hand, three, 60, elixir=8)
    assert play is None or play.card_id not in ("arrows", "snowball")


# ---- defender choice ----------------------------------------------------------------------


def test_lone_unknown_marker_gets_the_air_safe_answer():
    for seed in range(10):
        play = _brain(seed=seed).choose(
            {0: "giant_skeleton", 1: "lumberjack", 2: "skeletons", 3: "evo_furnace"}, [unit(115, 300)], 60, elixir=8
        )
        assert play is not None and play.card_id == "evo_furnace"
        play = _brain(seed=seed).choose(
            {0: "giant_skeleton", 1: "skeletons", 2: "ice_spirit", 3: "arrows"}, [unit(115, 300)], 60, elixir=8
        )
        assert play is not None and play.card_id == "ice_spirit"


def test_unknown_card_loses_ties():
    for seed in range(10):
        play = _brain(seed=seed).choose({0: "UNKNOWN", 1: "lumberjack"}, [unit(115, 300, 45, "hog")], 60, elixir=8)
        assert play is not None and play.card_id == "lumberjack"


def test_cycle_cards_land_on_the_threat():
    play = _brain().choose({0: "skeletons"}, [unit(115, 380, 45)], 60, elixir=5)
    assert play is not None and play.defensive and abs(play.coord[1] - 390) <= 10


# ---- threat memory ------------------------------------------------------------------------


def test_threat_seen_a_moment_ago_is_not_forgotten():
    clock = Clock()
    brain = _brain(clock)
    brain.note_threat(unit(L_BRIDGE, 320, 45))
    clock.t += 1.8
    hand = {0: "arrows", 1: "giant_skeleton", 2: "skeletons", 3: "lumberjack"}
    for elixir in (5, 10):
        play = brain.choose(hand, [], 60, elixir=elixir)
        assert play is None or (play.defensive and play.card_id != "arrows")
    clock.t += 3.0
    play = brain.choose(hand, [], 60, elixir=10)
    assert play is not None and not play.defensive  # memory expired: normal offense again


# ---- defense sizing, locks ----------------------------------------------------------------


def test_defense_budget_sizes_the_answer():
    brain = _brain()
    assert brain.defense_budget(bb.EnemyPush("left", unit(110, 300), (110, 300), 1, False)) == 3
    assert brain.defense_budget(bb.EnemyPush("left", unit(110, 300), (110, 300), 2, False)) == 4
    assert brain.defense_budget(bb.EnemyPush("left", unit(110, 300), (110, 300), 3, False)) == 5
    tank = bb.EnemyPush("left", unit(110, 300, kind="golem"), (110, 300), 1, False, ("golem",))
    assert brain.defense_budget(tank) == 5
    brain.should_defend_now([unit(110, 300), unit(125, 310)])
    assert brain.defense_budget(unit(110, 300)) == 4  # a unit uses its lane's push


def test_short_on_elixir_waits_then_answers_cheaply_at_the_tower():
    hand = {0: "skeletons", 1: "ice_spirit"}
    far = [unit(110, 300), unit(125, 290)]
    assert _brain().choose(hand, far, 60, elixir=2) is None
    near = [unit(110, 380), unit(125, 370)]
    play = _brain().choose(hand, near, 60, elixir=2)
    assert play is not None and play.defensive and card_cost(play.card_id) <= 2


def test_lane_lock_holds_until_the_push_grows_or_hurts():
    clock = Clock()
    brain = _brain(clock)
    threat = unit(110, 300, 45)
    play = brain.choose({0: "lumberjack", 1: "skeletons"}, [threat], 60, elixir=8)
    assert play is not None and play.threat_y == 300
    brain.record(play)
    clock.t += 2.0
    assert brain.should_defend_now([unit(110, 320, 45)]) is None  # same push, a bit deeper
    two = [unit(110, 320, 45), unit(125, 310, 40)]
    assert brain.should_defend_now(two) is None  # grew, one read
    clock.t += 0.5
    assert brain.should_defend_now(two) is not None  # grew on a second read
    clock2 = Clock()
    hurt = _brain(clock2)
    hurt.observe_towers(_health())
    hurt.record(Play(0, "lumberjack", "troop", (110, 365), "left", True, "defending", 300))
    clock2.t += 1.0
    hurt.observe_towers(_health(own=(0.8, 1.0)))
    hurt.observe_towers(_health(own=(0.8, 1.0)))
    assert hurt.should_defend_now([unit(110, 310, 45)]) is not None


def test_handled_swarm_is_left_to_our_troops():
    swarm = [unit(110 + dx, 330 + dy, 20) for dx, dy in ((0, 0), (12, 5), (-12, 5), (0, 14))]
    friends = [unit(110 + dx, 345, 30) for dx in (-10, 0, 10, 20)]
    assert _brain().should_defend_now(swarm, friends) is None
    assert _brain().should_defend_now(swarm, friends[:1]) is not None


# ---- counter-push gating ------------------------------------------------------------------


def test_counter_push_needs_a_surviving_defender_and_has_a_cap():
    clock = Clock()
    brain = _brain(clock)
    brain.record(Play(0, "lumberjack", "troop", (110, 365), "left", True, "defending the left lane", 300))
    clock.t += 3.0
    hand = {0: "skeletons", 1: "ice_spirit", 2: "lumberjack", 3: "giant_skeleton"}
    assert brain.choose(hand, [], 43, elixir=4, friends=[]) is None
    friend = [unit(110, 320, 30)]
    play = brain.choose(hand, [], 43, elixir=4, friends=friend)
    assert play is not None and play.lane == "left" and "counter-push" in play.reason
    plays = 0
    for _ in range(8):
        play = brain.choose({0: "skeletons", 1: "ice_spirit"}, [], 43, elixir=2, friends=friend)
        if play is None:
            break
        brain.record(play)
        plays += 1
    assert plays == bb.COUNTER_PUSH_SPEND_CAP


# ---- match phases, towers ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("elapsed", "phase", "near_leak", "back_tank"),
    [(60, "single", 9, 9), (150, "double", 8, 7), (200, "ot2", 8, 7), (260, "ot3", 7, 7)],
)
def test_phase_thresholds(elapsed, phase, near_leak, back_tank):
    assert match_phase(elapsed) == phase
    assert bb.NEAR_LEAK_BY_PHASE[phase] == near_leak
    assert bb.BACK_TANK_MIN_BY_PHASE[phase] == back_tank
    brain = _brain()
    hand = {0: "giant_skeleton"}
    assert brain.choose(hand, [], elapsed, elixir=back_tank - 1) is None
    play = brain.choose(hand, [], elapsed, elixir=back_tank)
    assert play is not None and play.coord == BACK_TANK_COORDS[play.lane]
    knight = {0: "knight"}
    assert _brain().choose(knight, [], elapsed, elixir=near_leak - 1) is None
    assert _brain().choose(knight, [], elapsed, elixir=near_leak) is not None


def test_match_clock_overrides_elapsed():
    clock = Clock()
    match = MatchClock(start=clock.t - 150, clock=clock)
    assert match.phase() == "double"
    brain = _brain(clock)
    play = brain.choose({0: "giant_skeleton"}, [], 30, elixir=7, clock=match)
    assert play is not None  # 7 is enough in double elixir
    play = _brain().choose({0: "giant_skeleton"}, [], 30, elixir=7, clock=150.0)
    assert play is not None


def test_overtime_defends_the_lowest_tower_first():
    brain = _brain()
    for _ in range(3):
        brain.observe_towers(_health(own=(0.7, 0.5)))
    brain.choose({0: "knight"}, [], 200, elixir=1)  # sets the phase
    deep_left, shallow_right = unit(100, 400), unit(300, 300)
    assert brain.should_defend_now([deep_left, shallow_right]) is shallow_right


def test_none_tower_reads_are_skipped():
    brain = _brain()
    brain.observe_towers(None)
    brain.observe_towers(TowerHealth(enemy={"left": None, "right": 0.5}, own={}))
    assert brain.enemy_tower_hp("left") is None
    assert brain.enemy_tower_hp("right") == 0.5
    assert brain.own_tower_hp("left") is None
    brain.choose({0: "arrows"}, [], 60, elixir=5)


def test_tower_median_is_time_windowed():
    clock = Clock()
    brain = _brain(clock)
    for _ in range(5):
        brain.observe_towers(_health(enemy=(1.0, 1.0)))
    clock.t += 10
    brain.observe_towers(_health(enemy=(0.5, 1.0)))
    assert brain.enemy_tower_hp("left") == 0.5  # old reads aged out


def test_undeployed_play_changes_nothing():
    clock = Clock()
    brain = _brain(clock)
    play = Play(0, "lumberjack", "troop", (110, 365), "left", True, "defending the left lane", 300)
    brain.record(play, deployed=False)
    assert brain.recent == [] and brain.should_defend_now([unit(110, 300)]) is not None
    assert brain.deck.plays[-1] == (0, "lumberjack", False)
