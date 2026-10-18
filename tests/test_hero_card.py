"""Deck hero-card hint (``deck_hero_card`` setting -> ``BattleBrain(hero_card=...)``).

The user's deck runs Hero Balloon, which has no hand fingerprint and reads UNKNOWN. With the
hint, an UNKNOWN slot the rotation can't name becomes the Balloon when its cost bounds allow,
the Balloon counts as the deck's win condition (Giant Skeleton becomes its tank), and the
Balloon is only ever played tank-led: behind the Giant Skeleton or right behind a Lumberjack.
"""

from __future__ import annotations

import importlib
import itertools
import random
from typing import Any

import pytest

from pyclashbot.bot import battle_brain as bb
from pyclashbot.bot import card_knowledge as ck
from pyclashbot.bot.battle_brain import BattleBrain, DeckTracker, card_cost, normalize_hero_card
from pyclashbot.bot.coords import BACK_TANK_COORDS, LANE_BRIDGE_X, OWN_HALF_DROP_Y_RANGE
from pyclashbot.bot.find import EnemyUnit, TowerHealth
from pyclashbot.interface.enums import UIField

HERO = "hero_balloon"
DECK = ["arrows", "evo_furnace", "giant_skeleton", "ice_spirit", "lumberjack", "skeletons", "snowball", "UNKNOWN"]
TRUE_COST = {"UNKNOWN": 5}  # the UNKNOWN slot is the Balloon
L_BRIDGE = 117


class Clock:
    def __init__(self, t: float = 10_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _brain(clock=None, seed=0) -> BattleBrain:
    brain = BattleBrain(clock=clock or Clock(), rng=random.Random(seed), hero_card=HERO)
    brain._played_any = True
    brain.deck.observe_hand(DECK)
    return brain


def _choose(brain: BattleBrain, full: dict[int, str], elixir: int, threats=(), friends=(), elapsed=60.0):
    """choose() the way fight.py calls it: playable = affordable slots, plus the full hand."""
    playable = {i: c for i, c in full.items() if TRUE_COST.get(c, card_cost(c)) <= elixir}
    return brain.choose(
        playable, list(threats), elapsed, elixir=elixir, friends=list(friends), full_hand=full, affordable=set(playable)
    )


def _tank_led(play) -> bool:
    return play is not None and bb.is_tank_led_win(play.card_id)


# ---- the setting value ------------------------------------------------------------------


@pytest.mark.parametrize("raw", [None, "", "None", "none", "auto", "  ", "UNKNOWN"])
def test_blank_settings_mean_no_hero(raw):
    assert normalize_hero_card(raw) is None


def test_hero_ids_keep_their_base_card():
    assert normalize_hero_card(" hero_balloon ") == HERO
    assert ck.base_card_id(HERO) == "balloon"
    assert bb.card_role(HERO) == bb.ROLE_WIN and card_cost(HERO) == 5
    assert bb.is_tank_led_win(HERO) and not bb.is_tank_led_win("UNKNOWN")


def test_brain_without_args_has_no_hero():
    brain = BattleBrain()
    assert brain.hero_card is None and brain.deck.hero_card is None and brain.deck.deck == []


# ---- DeckTracker --------------------------------------------------------------------------


def test_hero_counts_as_a_deck_card_and_is_the_win_condition():
    deck = DeckTracker(HERO)
    assert deck.deck == ["balloon"]
    deck.observe_hand(DECK)
    assert len(deck.deck) == 8 and "balloon" in deck.deck
    assert deck.inferred_win() is None  # Giant Skeleton is the tank, not the win condition
    assert deck.win_conditions() == ["balloon"]


def test_hint_names_an_unknown_slot_the_rotation_cant():
    deck = DeckTracker(HERO)
    assert deck.resolve_hand({0: "UNKNOWN", 1: "arrows"}) == {0: HERO, 1: "arrows"}
    assert deck.maybe_hero_slots == set()
    # Without the hint the slot stays unknown.
    assert DeckTracker().resolve_hand({0: "UNKNOWN"}) == {0: "UNKNOWN"}


def test_cost_bounds_veto_the_hint():
    deck = DeckTracker(HERO)
    deck.observe_costs({0: "UNKNOWN", 1: "arrows"}, affordable={0, 1}, elixir=4)  # costs <= 4: not a Balloon
    assert deck.resolve_hand({0: "UNKNOWN", 1: "arrows"})[0] == "UNKNOWN"
    deck = DeckTracker(HERO)
    deck.observe_costs({0: "UNKNOWN"}, affordable=set(), elixir=4)  # costs >= 5: fits
    assert deck.resolve_hand({0: "UNKNOWN"})[0] == HERO


def test_rotation_beats_the_hint():
    deck = DeckTracker(HERO)
    for slot, card in enumerate(["lumberjack", "arrows", "skeletons", "snowball"]):
        deck.played(card, slot)
    deck.played("ice_spirit", 0)
    assert deck.resolve_hand({0: "UNKNOWN"})[0] == "lumberjack"  # e.g. the evo-ready Lumberjack


def test_a_balloon_still_in_the_queue_is_not_in_hand():
    deck = DeckTracker(HERO)
    deck.played(HERO, 1)
    deck.played("arrows", 2)
    assert deck.resolve_hand({3: "UNKNOWN"})[3] == "UNKNOWN"  # played 1 play ago: can't be back
    for slot, card in enumerate(["skeletons", "snowball", "ice_spirit"]):
        deck.played(card, slot)
    assert deck.resolve_hand({3: "UNKNOWN"})[3] == HERO


def test_two_candidates_stay_unknown_and_are_not_played():
    brain = _brain()
    full = {0: "UNKNOWN", 1: "UNKNOWN", 2: "arrows", 3: "snowball"}
    for elixir in (5, 7, 10):
        play = brain.choose(full, [], 60, elixir=elixir, full_hand=full, affordable=set(full))
        assert play is None or play.index not in (0, 1)
    assert brain.deck.maybe_hero_slots == {0, 1}


# ---- play rules ---------------------------------------------------------------------------


def test_giant_skeleton_goes_first_then_balloon_behind_it():
    for seed in range(5):
        clock = Clock()
        brain = _brain(clock, seed)
        full = {0: "giant_skeleton", 1: "UNKNOWN", 2: "lumberjack", 3: "arrows"}
        play = _choose(brain, full, 9)
        assert play is not None and play.card_id == "giant_skeleton", play
        assert play.coord == BACK_TANK_COORDS[play.lane] and play.role != bb.ROLE_WIN
        brain.record(play)
        lane, elixir = play.lane, 3.0
        full = {0: "skeletons", 1: "UNKNOWN", 2: "lumberjack", 3: "arrows"}
        balloon = None
        for _ in range(14):
            clock.t += 1.0
            elixir = min(10.0, elixir + 1 / 2.8)
            follow = _choose(brain, full, int(elixir))
            if follow is None:
                continue
            assert _tank_led(follow), f"{follow} before the balloon"
            balloon = follow
            break
        assert balloon is not None and balloon.card_id == HERO and balloon.lane == lane
        assert clock.t - 10_000.0 >= bb.FOLLOW_UP_HOLD_S - 0.5  # waited for the tank to walk up
        assert balloon.coord[1] < BACK_TANK_COORDS[lane][1] - 60  # up the lane behind the tank


def test_balloon_goes_right_behind_the_tank_once_it_nears_the_bridge():
    clock = Clock()
    brain = _brain(clock)
    play = _choose(brain, {0: "giant_skeleton", 1: "UNKNOWN", 2: "lumberjack", 3: "arrows"}, 9)
    assert play is not None
    brain.record(play)
    clock.t += 4.0
    tank = EnemyUnit(BACK_TANK_COORDS[play.lane][0], 320, 60)
    follow = _choose(brain, {0: "skeletons", 1: "UNKNOWN", 2: "lumberjack", 3: "arrows"}, 5, friends=[tank])
    assert follow is not None and follow.card_id == HERO
    assert 0 < follow.coord[1] - tank.y <= 25  # ~1 tile behind our tank, same lane
    assert follow.lane == play.lane


@pytest.mark.parametrize("elixir", [9, 8])
def test_lumberloon_lumberjack_at_the_bridge_then_balloon_behind(elixir):
    clock = Clock()
    brain = _brain(clock)
    full = {0: "lumberjack", 1: "UNKNOWN", 2: "skeletons", 3: "arrows"}
    play = _choose(brain, full, elixir)
    assert play is not None and play.card_id == "lumberjack" and "lumberloon" in play.reason
    assert play.coord == (LANE_BRIDGE_X[play.lane], OWN_HALF_DROP_Y_RANGE[0])
    brain.record(play)
    left = float(elixir - 4)
    full = {0: "ice_spirit", 1: "UNKNOWN", 2: "skeletons", 3: "arrows"}
    for _ in range(12):
        clock.t += 0.35
        left += 0.35 / 2.8
        follow = _choose(brain, full, int(left))
        if follow is not None:
            break
    assert follow is not None and follow.card_id == HERO and follow.lane == play.lane
    assert follow.coord[0] == play.coord[0] and follow.coord[1] > play.coord[1]
    assert clock.t - 10_000.0 <= bb.PAIR_FOLLOW_S


def test_lumberloon_waits_for_enough_elixir_in_single():
    play = _choose(_brain(), {0: "lumberjack", 1: "UNKNOWN", 2: "skeletons", 3: "arrows"}, 7)
    assert not _tank_led(play)
    assert play is None or "lumberloon" not in play.reason


def test_balloon_never_defends():
    pushes = {
        "hog": [EnemyUnit(L_BRIDGE, 295, 45)],
        "hog_at_tower": [EnemyUnit(L_BRIDGE, 340, 45)],
        "pair": [EnemyUnit(L_BRIDGE, 320, 45), EnemyUnit(L_BRIDGE + 15, 305, 40)],
        "swarm": [EnemyUnit(L_BRIDGE + dx, 300 + dy, 20) for dx, dy in ((-15, 0), (0, -8), (15, 0), (0, 14))],
        "air": [EnemyUnit(L_BRIDGE, 360, 50, "balloon")],
    }
    for name, threats in pushes.items():
        for hand4 in itertools.combinations(DECK, 4):
            for elixir in (3, 5, 7, 9, 10):
                play = _choose(_brain(seed=elixir), dict(enumerate(hand4)), elixir, threats=threats)
                assert not _tank_led(play), (name, hand4, elixir, play)


def test_no_lone_balloon_mid_elixir_or_with_a_tank_in_hand():
    for seed in range(5):
        assert not _tank_led(_choose(_brain(seed=seed), {0: "UNKNOWN", 1: "arrows", 2: "snowball", 3: "ice_spirit"}, 7))
        play = _choose(_brain(seed=seed), {0: "UNKNOWN", 1: "giant_skeleton", 2: "snowball", 3: "arrows"}, 9)
        assert play is not None and play.card_id == "giant_skeleton"


def test_lone_balloon_near_a_leak_or_at_a_low_tower():
    hand = {0: "UNKNOWN", 1: "arrows", 2: "snowball", 3: "ice_spirit"}
    assert _tank_led(_choose(_brain(), hand, 10))
    brain = _brain()
    for _ in range(3):
        brain.observe_towers(TowerHealth(enemy={"left": 0.3, "right": 0.3}, own={"left": 1.0, "right": 1.0}))
    assert _tank_led(_choose(brain, hand, 8))


def test_tank_led_push_prefers_the_weaker_tower():
    brain = _brain()
    brain.attack_lane = "left"
    for _ in range(3):
        brain.observe_towers(TowerHealth(enemy={"left": 0.9, "right": 0.8}, own={"left": 1.0, "right": 1.0}))
    play = _choose(brain, {0: "giant_skeleton", 1: "UNKNOWN", 2: "lumberjack", 3: "arrows"}, 9)
    assert play is not None and play.card_id == "giant_skeleton" and play.lane == "right"


def test_spells_are_kept_after_a_balloon_push():
    clock = Clock()
    brain = _brain(clock)
    play = _choose(brain, {0: "lumberjack", 1: "UNKNOWN", 2: "skeletons", 3: "arrows"}, 9)
    brain.record(play)
    clock.t += 0.3
    balloon = _choose(brain, {0: "ice_spirit", 1: "UNKNOWN", 2: "skeletons", 3: "arrows"}, 5)
    assert _tank_led(balloon)
    brain.record(balloon)
    clock.t += 5.0
    leak = _choose(brain, {0: "arrows", 1: "snowball"}, 10)
    assert leak is None or leak.role != bb.ROLE_SPELL


# ---- setting plumbing ---------------------------------------------------------------------


def test_setting_key_options_and_default():
    from pyclashbot.interface import config

    assert UIField.DECK_HERO_CARD.value == "deck_hero_card"
    assert UIField.DECK_HERO_CARD.value in config.USER_CONFIG_KEYS
    options = config.deck_hero_card_options()
    assert options[0] == config.DECK_HERO_NONE == "None"
    heroes = [o for o in options if o.startswith("hero_")]
    assert HERO in heroes and options[1 : 1 + len(heroes)] == heroes  # heroes first
    assert {"giant_skeleton", "balloon", "evo_furnace"} <= set(options)
    smart = next(job for job in config.JOBS if job.key == UIField.SMART_BATTLE_USER_TOGGLE)
    assert smart.extras is not None and smart.extras[UIField.DECK_HERO_CARD].default == "None"


def test_job_dictionary_maps_none_to_no_hero(monkeypatch):
    from pyclashbot.utils import logger as logger_mod

    monkeypatch.setattr(logger_mod, "initialize_pylogging", lambda: None)  # no log file from the import
    main = importlib.import_module("pyclashbot.__main__")
    assert main.make_job_dictionary({UIField.DECK_HERO_CARD.value: "None"})[UIField.DECK_HERO_CARD.value] is None
    assert main.make_job_dictionary({})[UIField.DECK_HERO_CARD.value] is None
    jobs = main.make_job_dictionary({UIField.DECK_HERO_CARD.value: HERO})
    assert jobs[UIField.DECK_HERO_CARD.value] == HERO


class _Logger:
    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None


def test_states_pass_the_hero_card_to_the_fight(monkeypatch):
    from pyclashbot.bot import states

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(states, "do_fight_state", lambda *a, **kw: calls.append(kw) or True)
    monkeypatch.setattr(states.time, "sleep", lambda _s: None)
    monkeypatch.setattr(states, "mode_used_in_1v1", "Classic 1v1")
    logger: Any = _Logger()
    jobs = {UIField.SMART_BATTLE_USER_TOGGLE.value: True, UIField.DECK_HERO_CARD.value: HERO}
    states.state_tree(None, logger, "1v1_fight", jobs, states.StateHistory(logger), states.StateOrder())
    assert calls and calls[0]["hero_card"] == HERO and calls[0]["smart_battle"] is True
    calls.clear()
    states.state_tree(None, logger, "1v1_fight", {}, states.StateHistory(logger), states.StateOrder())
    assert calls[0]["hero_card"] is None


class _StopError(Exception):
    pass


@pytest.mark.parametrize(("hero", "expected"), [(HERO, {"hero_card": HERO}), ("None", {}), (None, {})])
def test_fight_loop_builds_the_brain_with_the_hero(monkeypatch, hero, expected):
    from pyclashbot.bot import fight

    built: list[dict[str, Any]] = []

    def factory(**kwargs):
        built.append(kwargs)
        raise _StopError

    monkeypatch.setattr(fight, "create_default_bridge_iar", lambda _e: None)
    monkeypatch.setattr(fight, "_maybe_start_fight_recording", lambda *a: None)
    monkeypatch.setattr(fight, "_configure_hand_crops", lambda *a: None)
    monkeypatch.setattr(fight, "BattleBrain", factory)
    logger: Any = _Logger()
    with pytest.raises(_StopError):
        fight._fight_loop(None, logger, False, "Classic 1v1", None, smart_battle=True, hero_card=hero)
    assert built == [expected]


def test_do_fight_state_forwards_the_hero(monkeypatch):
    from pyclashbot.bot import fight

    seen: list[tuple] = []
    monkeypatch.setattr(fight, "wait_for_battle_start", lambda *a: True)
    monkeypatch.setattr(fight, "_fight_loop", lambda *a, **_k: seen.append(a) or False)
    logger: Any = _Logger()
    fight.do_fight_state(None, logger, False, "Classic 1v1", smart_battle=True, hero_card=HERO)
    assert seen and seen[0][-1] == HERO


# ---- tempo: Giant Skeleton + Balloon, low-tower reserve -------------------------------------


@pytest.mark.parametrize("balloon", [HERO, "UNKNOWN"])  # remembered/learned, or resolved by the hint
def test_giant_skeleton_starts_at_8_with_the_balloon_in_hand(balloon):
    for seed in range(5):
        hand = {0: balloon, 1: "giant_skeleton", 2: "arrows", 3: "ice_spirit"}
        play = _choose(_brain(seed=seed), hand, 8)
        assert play is not None and play.card_id == "giant_skeleton", (seed, play)
        assert play.coord == BACK_TANK_COORDS[play.lane]


def test_giant_skeleton_still_waits_for_9_without_the_balloon():
    for seed in range(5):
        play = _choose(_brain(seed=seed), {0: "skeletons", 1: "giant_skeleton", 2: "arrows", 3: "ice_spirit"}, 8)
        assert play is None or play.card_id != "giant_skeleton"


def test_balloon_out_of_hand_unaffordable_still_counts_for_the_tank():
    # A remembered Balloon in a greyed slot (not affordable) is still in hand: GS goes at 8.
    brain = _brain()
    full = {0: HERO, 1: "giant_skeleton", 2: "arrows", 3: "ice_spirit"}
    playable = {1: "giant_skeleton", 2: "arrows", 3: "ice_spirit"}
    play = brain.choose(playable, [], 60.0, elixir=8, full_hand=full, affordable=set(playable))
    assert play is not None and play.card_id == "giant_skeleton"


def _low_own_tower(brain: BattleBrain) -> None:
    for _ in range(3):
        brain.observe_towers(TowerHealth(enemy={"left": 1.0, "right": 1.0}, own={"left": 1.0, "right": 0.4}))


def test_low_own_tower_keeps_a_4_elixir_reserve_before_a_push():
    brain = _brain()
    _low_own_tower(brain)
    assert brain.reserve() >= 4
    hand = {0: HERO, 1: "giant_skeleton", 2: "arrows", 3: "ice_spirit"}
    for elixir in (8, 9):
        play = _choose(brain, hand, elixir)
        assert play is None or play.defensive or play.card_id != "giant_skeleton", (elixir, play)
    play = _choose(_brain(), hand, 9)  # healthy towers: the same push goes
    assert play is not None and play.card_id == "giant_skeleton"


def test_low_own_tower_still_spends_at_full_elixir():
    brain = _brain()
    _low_own_tower(brain)
    play = _choose(brain, {0: HERO, 1: "giant_skeleton", 2: "arrows", 3: "ice_spirit"}, 10)
    assert play is not None
