"""card_knowledge: per-card facts the battle brain plays by, and the DeckTracker built on them."""

import pytest

from pyclashbot.bot import card_knowledge as ck
from pyclashbot.bot.battle_brain import ELIXIR_COST, BattleBrain, DeckTracker, OpponentDeck, _base_card_id
from pyclashbot.bot.card_detection import CARD_GROUPS, card_color_data

USER_DECK = ["arrows", "evo_furnace", "giant_skeleton", "ice_spirit", "lumberjack", "skeletons", "snowball", "UNKNOWN"]


def _all_ids() -> set[str]:
    return {c for group in CARD_GROUPS.values() for c in group} | set(card_color_data) | set(ELIXIR_COST)


@pytest.mark.parametrize("card_id", sorted(_all_ids()))
def test_every_known_id_has_card_info(card_id):
    info = ck.card_info(card_id)
    assert info is not None, card_id
    assert 1 <= info.cost <= 10
    assert info.kind in {ck.TROOP, ck.BUILDING, ck.SPELL, ck.CHAMPION}


def test_costs_agree_with_elixir_cost_table():
    for card_id, cost in ELIXIR_COST.items():
        info = ck.card_info(card_id)
        assert info is not None and info.cost == cost, card_id


def test_user_deck_cards():
    gs = ck.card_info("giant_skeleton")
    assert gs is not None and gs.has(ck.TANK) and gs.has(ck.WIN2) and gs.def_place == ck.FRONT
    lj = ck.card_info("lumberjack")
    assert lj is not None and lj.roles >= {ck.MT, ck.TK}
    for cycle in ("skeletons", "ice_spirit"):
        info = ck.card_info(cycle)
        assert info is not None and info.has(ck.CY) and info.def_place == ck.ON
    furnace = ck.card_info("evo_furnace")
    assert furnace is not None and furnace.id == "furnace" and furnace.def_place == ck.PULL
    assert furnace.roles >= {ck.BLD, ck.SPN}
    for spell in ("arrows", "snowball"):
        info = ck.card_info(spell)
        assert info is not None and info.has(ck.SPS) and info.targets_air


def test_facts_that_change_decisions():
    night_witch = ck.card_info("night_witch")
    assert night_witch is not None and not night_witch.targets_air
    for ground_spell in ("log", "barb_barrel", "earthquake"):
        info = ck.card_info(ground_spell)
        assert info is not None and info.ground_only_spell
    assert ck.card_info("UNKNOWN") is None
    assert ck.card_info(None) is None


@pytest.mark.parametrize(
    ("detected", "base"),
    [
        ("evo_furnace", "furnace"),
        ("hero_giant", "giant"),
        ("fire_cracker", "firecracker"),
        ("spirit_empress_air", "spirit_empress"),
        ("spirit_empress_ground", "spirit_empress"),
        ("furnace_evo_charging", "furnace"),
        ("lumberjack_evo_ready", "lumberjack"),
    ],
)
def test_aliases_match_the_brain(detected, base):
    assert ck.base_card_id(detected) == base
    assert _base_card_id(detected) == base


# ---- DeckTracker ----------------------------------------------------------------


def test_evo_and_base_ids_are_one_card():
    deck = DeckTracker()
    deck.observe_hand(["evo_furnace", "furnace", "skeletons"])
    assert deck.deck == ["furnace", "skeletons"]
    deck.played("evo_furnace")
    assert deck.plays_until("furnace", ["skeletons"]) == 1
    assert deck.plays_until("evo_furnace", ["furnace"]) == 0


def test_rotation_names_the_unknown_slot():
    deck = DeckTracker()
    # Slots 0-3 then 0 again: slot 0 is refilled with the card played 4 deployed plays earlier.
    for slot, card in enumerate(["arrows", "lumberjack", "skeletons", "snowball"]):
        deck.played(card, slot)
    deck.played("ice_spirit", 0)
    assert deck.resolve_unknown(0) == "arrows"
    assert deck.resolve_unknown(3) is None  # its last play has no 4-plays-earlier card yet


def test_undeployed_plays_dont_shift_the_rotation():
    deck = DeckTracker()
    for slot, card in enumerate(["arrows", "lumberjack", "skeletons"]):
        deck.played(card, slot)
    deck.played("giant_skeleton", 1, deployed=False)  # tapped, never went down
    deck.played("snowball", 3)
    deck.played("ice_spirit", 0)
    assert deck.resolve_unknown(0) == "arrows"
    assert deck.queue([]) == ["arrows", "lumberjack", "skeletons", "snowball", "ice_spirit"]


def test_unknown_becomes_deck_x_once_seven_cards_are_known():
    deck = DeckTracker()
    deck.observe_hand(USER_DECK)
    assert "UNKNOWN" not in deck.deck and len(deck.deck) == 7
    deck.played("UNKNOWN")
    assert deck.queue([]) == ["deck_x"]


def test_unknown_cost_bounds_from_affordability():
    deck = DeckTracker()
    deck.observe_costs({0: "UNKNOWN", 1: "arrows"}, affordable={1}, elixir=3)
    assert deck.cost_bounds[0] == (4, 10)
    deck.observe_costs({0: "UNKNOWN", 1: "arrows"}, affordable={0, 1}, elixir=5)
    assert deck.cost_bounds[0] == (4, 5)


def test_giant_skeleton_is_the_inferred_win_condition():
    deck = DeckTracker()
    deck.observe_hand(USER_DECK)
    assert deck.inferred_win() == "giant_skeleton"
    assert deck.win_conditions() == ["giant_skeleton"]
    # A deck with a real win condition infers nothing.
    hog = DeckTracker()
    hog.observe_hand(["hog", "giant_skeleton", "log", "skeletons", "musketeer", "cannon"])
    assert hog.inferred_win() is None


def test_inference_waits_for_enough_of_the_deck():
    deck = DeckTracker()
    deck.observe_hand(["giant_skeleton", "lumberjack", "skeletons", "arrows"])
    assert deck.inferred_win() is None


def test_enemy_giant_skeleton_is_not_their_win_condition():
    clock = [100.0]
    from pyclashbot.bot.find import EnemyUnit

    opponent = OpponentDeck(lambda: clock[0])
    opponent.observe([EnemyUnit(110, 320, 60, kind="giant_skeleton")])
    assert opponent.cards == ["giant_skeleton"]
    assert opponent.win_condition_out_of_cycle() is False


def test_brain_records_slots_and_resolves_unknown_in_hand():
    brain = BattleBrain(clock=lambda: 1000.0)
    brain.deck.observe_hand(USER_DECK[:7])
    for slot, card in enumerate(["arrows", "lumberjack", "skeletons", "snowball", "ice_spirit"]):
        brain.deck.played(card, slot % 4)
    assert brain.deck.resolve_unknown(0) == "arrows"
