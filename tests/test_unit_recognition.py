"""Offline coverage for enemy unit-type recognition: crops, capture, classifier, dataset, OpponentDeck."""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from pyclashbot.bot import fight, recorder, unit_classifier, unit_dataset
from pyclashbot.bot.battle_brain import (
    ARCHETYPE_PRESSURE,
    CYCLE_FORGET_S,
    DEFENSE_RESERVE,
    PRESSURE_DEFENSE_RESERVE,
    PUNISH_DEFENSE_RESERVE,
    BattleBrain,
    OpponentDeck,
    unit_traits,
)
from pyclashbot.bot.coords import ENEMY_UNIT_CROP_OFFSETS_LTRB, HAND_CARDS_COORDS
from pyclashbot.bot.find import EnemyUnit, crop_unit, find_enemy_units
from pyclashbot.bot.unit_classifier import (
    NOT_A_UNIT,
    NearestNeighborUnitClassifier,
    NullUnitClassifier,
    classify_units,
    unit_features,
)

FIXTURE = Path(__file__).parent / "fixtures" / "card_hand_bgr_regression.png"
FIXTURE_HAND = {0: "lumberjack", 1: "ram_rider", 2: "ice_wizard", 3: "hero_bowler"}
_L, _T, _R, _B = ENEMY_UNIT_CROP_OFFSETS_LTRB
CROP_SHAPE = (_B - _T, _R - _L, 3)
# Empty spots on the fixture board (grass, river, tower) — negatives for the classifier.
BACKGROUND_SPOTS = [(300, 330), (200, 300), (60, 380), (350, 450), (209, 270)]


@pytest.fixture
def frame():
    image = cv2.imread(str(FIXTURE))
    assert image is not None
    return image


@pytest.fixture
def balloon(frame):
    (unit,) = find_enemy_units(frame)
    return unit


def _crop(frame, unit: EnemyUnit) -> np.ndarray:
    crop = crop_unit(frame, unit)
    assert crop is not None
    return crop


def _model(frame, balloon) -> NearestNeighborUnitClassifier:
    """Trained on the fixture's balloon (plus a couple of 2px-shifted copies) and background spots."""
    crops, labels = [], []
    for dx, dy in ((0, 0), (2, 0), (0, 2)):
        crops.append(_crop(frame, EnemyUnit(balloon.x + dx, balloon.y + dy, balloon.size)))
        labels.append("balloon")
    for x, y in BACKGROUND_SPOTS:
        crops.append(_crop(frame, EnemyUnit(x, y, 0)))
        labels.append(NOT_A_UNIT)
    return NearestNeighborUnitClassifier.fit(crops, labels)


class _Clock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


# ---- crops ---------------------------------------------------------------------


def test_crop_has_a_fixed_shape_even_at_the_screen_edge(frame, balloon):
    crop = _crop(frame, balloon)
    assert crop is not None and crop.shape == CROP_SHAPE
    corner = crop_unit(frame, EnemyUnit(2, 630, 10))
    assert corner is not None and corner.shape == CROP_SHAPE
    assert crop_unit(None, balloon) is None


def test_balloon_crop_holds_the_red_balloon(frame, balloon):
    crop = _crop(frame, balloon)
    b, g, r = (float(v) for v in crop.reshape(-1, 3).mean(axis=0))
    assert r > g and r > b  # the red balloon dominates the crop


# ---- classifier ----------------------------------------------------------------


def test_features_are_normalized_and_cheap(frame, balloon):
    vector = unit_features(_crop(frame, balloon))
    assert vector.ndim == 1 and vector.size < 300
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-5


def test_classifier_recognizes_the_fixture_balloon(frame, balloon):
    model = _model(frame, balloon)
    moved = EnemyUnit(balloon.x - 3, balloon.y + 1, balloon.size)  # a frame later, slightly elsewhere
    guess = model.classify(_crop(frame, moved))
    assert guess is not None and guess.label == "balloon" and guess.confidence >= 0.6


def test_classifier_abstains_on_background(frame, balloon):
    model = _model(frame, balloon)
    for x, y in BACKGROUND_SPOTS:
        assert model.classify(_crop(frame, EnemyUnit(x + 1, y + 1, 0))) is None


def test_model_round_trips_through_disk(frame, balloon, tmp_path):
    model = _model(frame, balloon)
    path = tmp_path / "model.npz"
    model.save(path)
    loaded = NearestNeighborUnitClassifier.load(path)
    assert loaded.labels == model.labels and loaded.k == model.k
    assert loaded.classify(_crop(frame, balloon)) == model.classify(_crop(frame, balloon))


def test_without_a_model_nothing_is_recognized(frame, balloon, monkeypatch, tmp_path):
    monkeypatch.setattr(unit_classifier, "DEFAULT_MODEL_PATH", tmp_path / "missing.npz")
    unit_classifier.default_classifier.cache_clear()
    try:
        assert isinstance(unit_classifier.default_classifier(), NullUnitClassifier)
        assert classify_units(frame, [balloon]) == [balloon]
    finally:
        unit_classifier.default_classifier.cache_clear()


def test_classify_units_fills_in_kind(frame, balloon):
    (unit,) = classify_units(frame, [balloon], _model(frame, balloon))
    assert unit.kind == "balloon" and (unit.x, unit.y) == (balloon.x, balloon.y)


def test_traits_for_cards_and_categories():
    balloon = unit_traits("balloon")
    assert balloon is not None and balloon.air and balloon.win_condition and not balloon.swarm
    horde = unit_traits("minion_horde")
    assert horde is not None and horde.air and horde.swarm
    bowler = unit_traits("evo_bowler")
    assert bowler is not None and bowler.splash and not bowler.air
    assert unit_traits("ground_tank") == unit_classifier.CATEGORY_TRAITS["ground_tank"]
    assert unit_traits(None) is None


# ---- capture -------------------------------------------------------------------


def test_recorder_saves_unit_crops_and_frame(frame, balloon, tmp_path):
    rec = recorder.FightPackRecorder()
    rec.dir = str(tmp_path)
    assert rec.log_units(frame, [balloon], elapsed_s=12.345) == 1
    assert rec.log_units(frame, [balloon]) == 0  # throttled
    (line,) = (tmp_path / "units.jsonl").read_text(encoding="utf-8").splitlines()
    entry = json.loads(line)
    assert entry["label"] is None and entry["lane"] == "left" and entry["elapsed_s"] == 12.35
    crop = cv2.imread(str(tmp_path / entry["crop"]))
    assert crop is not None and crop.shape == CROP_SHAPE
    saved = cv2.imread(str(tmp_path / entry["frame"]))
    assert saved is not None and np.array_equal(saved, frame)  # PNG keeps the exact BGR frame


def test_recorder_ignores_units_when_not_recording(frame, balloon):
    assert recorder.FightPackRecorder().log_units(frame, [balloon]) == 0
    recorder.log_units(frame, [balloon])  # no active pack: no-op, no error


def test_play_smart_card_snapshots_units_while_recording(frame, monkeypatch):
    seen = []
    monkeypatch.setattr(fight.card_detection, "bridge_iar", frame)
    monkeypatch.setattr(fight, "log_units", lambda iar, units, elapsed=None: seen.append(units))
    strategy = fight.BattleStrategy()
    strategy.start_battle()
    brain = BattleBrain(clock=_Clock(), rng=random.Random(0))
    assert fight.play_smart_card(_FakeEmulator(frame), _FakeLogger(), True, strategy, brain) is True
    assert len(seen) == 1 and len(seen[0]) == 1


# ---- dataset -------------------------------------------------------------------


def test_collect_then_train_round_trip(frame, balloon, tmp_path):
    pack = tmp_path / "recordings" / "20261015-120000-abcdef"
    rec = recorder.FightPackRecorder()
    rec.dir = str(pack)
    rec.log_units(frame, [balloon, *(EnemyUnit(x, y, 0) for x, y in BACKGROUND_SPOTS[:2])])
    dataset = tmp_path / "dataset"

    assert unit_dataset.collect(tmp_path / "recordings", dataset) == 3
    assert unit_dataset.collect(tmp_path / "recordings", dataset) == 0  # idempotent
    unlabeled = sorted((dataset / unit_dataset.UNLABELED).glob("*.png"))
    assert len(unlabeled) == 3

    # A human sorts the crops into label folders.
    for path, label in zip(unlabeled, ["balloon", NOT_A_UNIT, NOT_A_UNIT], strict=True):
        (dataset / label).mkdir(exist_ok=True)
        path.rename(dataset / label / path.name)
    report = unit_dataset.train(dataset, tmp_path / "model.npz")
    assert report.samples == 3 and report.per_label == {"balloon": 1, NOT_A_UNIT: 2}
    assert report.unknown_labels == []
    model = NearestNeighborUnitClassifier.load(tmp_path / "model.npz")
    guess = model.classify(_crop(frame, balloon))
    assert guess is not None and guess.label == "balloon"


def test_known_labels():
    assert unit_dataset.known_label("minion_horde")
    assert unit_dataset.known_label("evo_valkyrie")
    assert unit_dataset.known_label("air_swarm")
    assert unit_dataset.known_label(NOT_A_UNIT)
    assert not unit_dataset.known_label("mystery_blob")


# ---- brain: defending by unit type --------------------------------------------


def _brain(clock=None, seed=0) -> BattleBrain:
    return BattleBrain(clock=clock or _Clock(), rng=random.Random(seed))


def test_air_push_gets_an_anti_air_defender():
    hand = {0: "knight", 1: "musketeer"}
    unknown = _brain().choose(hand, [EnemyUnit(110, 330, 40)], elapsed_s=40)
    assert unknown is not None and unknown.card_id == "knight"  # cheapest, ground-only
    air = _brain().choose(hand, [EnemyUnit(110, 330, 40, kind="balloon")], elapsed_s=40)
    assert air is not None and air.card_id == "musketeer" and air.defensive


def test_air_push_with_no_anti_air_uses_a_spell():
    play = _brain().choose({0: "knight", 1: "fireball"}, [EnemyUnit(110, 330, 40, kind="balloon")], elapsed_s=40)
    assert play is not None and play.card_id == "fireball"


def test_splash_troop_never_gets_a_swarm_card():
    for seed in range(8):
        play = _brain(seed=seed).choose(
            {0: "skeleton_army", 1: "knight"}, [EnemyUnit(110, 330, 40, kind="bowler")], elapsed_s=40
        )
        assert play is not None and play.card_id == "knight"


def test_recognized_swarm_is_spelled_even_as_one_marker():
    play = _brain().choose({0: "arrows", 1: "musketeer"}, [EnemyUnit(110, 330, 40, kind="minion_horde")], elapsed_s=40)
    assert play is not None and play.card_id == "arrows"


def test_tank_gets_a_tank_killer():
    hand = {0: "knight", 1: "mini_pekka", 2: "musketeer"}
    plain = _brain().choose(hand, [EnemyUnit(110, 330, 40)], elapsed_s=40)
    assert plain is not None and plain.card_id == "knight"
    tank = _brain().choose(hand, [EnemyUnit(110, 330, 40, kind="golem")], elapsed_s=40)
    assert tank is not None and tank.card_id == "mini_pekka"


def test_fixture_balloon_is_answered_with_anti_air(frame, balloon, monkeypatch):
    model = _model(frame, balloon)
    monkeypatch.setattr(unit_classifier, "default_classifier", lambda: model)
    monkeypatch.setattr(fight.card_detection, "bridge_iar", frame)
    emulator = _FakeEmulator(frame)
    logger = _FakeLogger()
    strategy = fight.BattleStrategy()
    strategy.start_battle()
    brain = _brain()
    assert fight.play_smart_card(emulator, logger, False, strategy, brain) is True
    # Lumberjack and Bowler can't hit the balloon; Ram Rider is a win condition -> Ice Wizard.
    assert emulator.clicks[0] == tuple(HAND_CARDS_COORDS[2])
    assert brain.opponent.deck.cards == ["balloon"]
    assert brain.opponent.archetype() == "Balloon"


# ---- OpponentDeck --------------------------------------------------------------


def _seen(deck: OpponentDeck, clock: _Clock, *kinds: str, step: float = 10.0) -> None:
    for kind in kinds:
        deck.observe([EnemyUnit(100, 330, 30, kind=kind)])
        clock.now += step


def test_same_unit_seen_again_is_one_play():
    clock = _Clock()
    deck = OpponentDeck(clock)
    _seen(deck, clock, "hog", "hog", step=2.0)
    assert [c for _, c in deck.plays] == ["hog"]
    clock.now += 10
    _seen(deck, clock, "hog")
    assert [c for _, c in deck.plays] == ["hog", "hog"] and deck.cards == ["hog"]


def test_categories_are_not_cards():
    clock = _Clock()
    deck = OpponentDeck(clock)
    _seen(deck, clock, "air_swarm", "ground_tank")
    assert deck.cards == [] and deck.plays == []


def test_cycle_estimate():
    clock = _Clock()
    deck = OpponentDeck(clock)
    _seen(deck, clock, "hog", step=1.0)
    assert deck.ready("hog") is False  # just played
    assert deck.ready("balloon") is None  # never seen
    _seen(deck, clock, "knight", "musketeer", "ice_golem", "skeletons", step=1.0)
    assert deck.ready("hog") is True  # 4 more plays: back in hand
    clock2 = _Clock()
    deck2 = OpponentDeck(clock2)
    _seen(deck2, clock2, "hog", step=CYCLE_FORGET_S + 1)
    assert deck2.ready("hog") is True  # too long ago to trust the cycle


def test_likely_hand_once_all_eight_are_known():
    clock = _Clock()
    deck = OpponentDeck(clock)
    cards = ["hog", "knight", "musketeer", "ice_golem", "skeletons", "valkyrie", "cannon", "bats"]
    _seen(deck, clock, *cards[:7], step=1.0)
    assert deck.likely_hand() is None
    _seen(deck, clock, cards[7], step=1.0)
    assert deck.likely_hand() == cards[:4]  # the last 4 played are cycling


@pytest.mark.parametrize(
    ("cards", "name"),
    [
        (["balloon", "lumberjack"], "Lumberloon"),
        (["balloon"], "Balloon"),
        (["golem", "baby_dragon"], "Golem beatdown"),
        (["hog", "valkyrie"], "Hog cycle"),
        (["battle_ram", "bandit"], "Bridge spam"),
        (["pekka", "battle_ram"], "P.E.K.K.A bridge spam"),
        (["xbow"], "X-Bow"),
        (["evo_valkyrie", "musketeer"], None),
    ],
)
def test_archetype_from_cards(cards, name):
    clock = _Clock()
    brain = _brain(clock)
    _seen(brain.opponent.deck, clock, *cards)
    named = brain.opponent.deck.named_archetype()
    assert (named[0] if named else None) == name
    if name is None:
        assert brain.opponent.archetype() == "balanced"  # falls back to the push pattern


def test_reserve_reacts_to_their_deck_and_cycle():
    clock = _Clock()
    brain = _brain(clock)
    assert brain.reserve() == DEFENSE_RESERVE
    _seen(brain.opponent.deck, clock, "lumberjack", step=1.0)
    _seen(brain.opponent.deck, clock, "balloon", step=1.0)
    assert brain.opponent.style() == ARCHETYPE_PRESSURE
    assert brain.reserve() == PUNISH_DEFENSE_RESERVE  # their balloon was just played
    clock.now += CYCLE_FORGET_S
    assert brain.reserve() == PRESSURE_DEFENSE_RESERVE


# ---- fakes ---------------------------------------------------------------------


class _FakeEmulator:
    def __init__(self, image):
        self.image = image
        self.clicks: list[tuple[int, int]] = []

    def screenshot(self):
        return self.image

    def click(self, x, y, *_a, **_k):
        self.clicks.append((x, y))


class _FakeLogger:
    def change_status(self, _status):
        pass

    def log(self, _msg):
        pass

    def add_card_played(self):
        pass
