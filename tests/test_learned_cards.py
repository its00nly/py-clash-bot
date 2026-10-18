"""Learned hand-card fingerprints (card_detection.LearnedCardStore) for deck-specific art.

The Hero Balloon frame and evo frames have no static fingerprint and read UNKNOWN. Verified
plays teach the matcher their art; a card only becomes matchable once 2 labelled samples agree.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pyclashbot.bot import card_detection as cd

YELLOW_ARENA_DAMAGED = Path(__file__).parent / "fixtures" / "arena_yellow_damaged_downscaled.png"


def _hists(seed: int, scale: int = 0) -> np.ndarray:
    """A (4, 16) corner-histogram sample; ``scale`` moves counts between bins (L1 = 8 * scale)."""
    rng = np.random.default_rng(seed)
    base = np.zeros((4, 16), dtype=np.int32)
    for corner in range(4):  # a few dominant colours per corner, like real card art
        base[corner] = np.bincount(rng.choice(16, size=891, p=rng.dirichlet(np.full(16, 0.3))), minlength=16)
    if scale:
        for corner in range(4):
            src = int(np.argmax(base[corner]))
            base[corner, src] -= scale
            base[corner, (src + 1) % 16] += scale
    return base


def test_store_discards_older_format(tmp_path):
    path = tmp_path / "learned_cards.json"
    path.write_text(json.dumps({"ice_spirit": [_hists(1).tolist(), _hists(1, scale=20).tolist()]}), encoding="utf-8")
    assert cd.LearnedCardStore(str(path)).samples == {}


def test_store_round_trip(tmp_path):
    path = tmp_path / "learned_cards.json"
    store = cd.LearnedCardStore(str(path))
    store.add("hero_balloon", _hists(1))
    store.add("hero_balloon", _hists(1, scale=20))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == cd.LEARNED_FORMAT_VERSION
    assert list(data["cards"]) == ["hero_balloon"] and len(data["cards"]["hero_balloon"]) == 2
    again = cd.LearnedCardStore(str(path))
    assert again.is_active("hero_balloon")
    assert np.array_equal(again.samples["hero_balloon"][0], _hists(1))


def test_corrupt_file_is_an_empty_store(tmp_path):
    path = tmp_path / "learned_cards.json"
    path.write_text("{not json", encoding="utf-8")
    store = cd.LearnedCardStore(str(path))
    assert store.samples == {} and store.active_entries() == []


def test_card_activates_after_two_consistent_samples():
    store = cd.LearnedCardStore(None)
    assert store.add("hero_balloon", _hists(1)) is False
    assert store.active_entries() == []  # one sample is never matched
    assert store.add("hero_balloon", _hists(1, scale=50)) is True  # L1 400 <= 1000: active
    assert {card for card, _ in store.active_entries()} == {"hero_balloon"}
    assert store.add("hero_balloon", _hists(1, scale=60)) is False  # already active: no second "learned"


def test_inconsistent_samples_never_activate():
    store = cd.LearnedCardStore(None)
    store.add("furnace", _hists(1))
    store.add("furnace", _hists(2))  # a different crop: far more than 1000 apart
    assert cd._hist_distance(_hists(1), _hists(2)) > cd.LEARNED_AGREE_DISTANCE
    assert not store.is_active("furnace") and store.active_entries() == []


def test_samples_are_capped_and_disagreeing_ones_go_first():
    store = cd.LearnedCardStore(None)
    store.add("lumberjack", _hists(1))
    store.add("lumberjack", _hists(1, scale=10))
    for seed in (5, 6, 7):
        store.add("lumberjack", _hists(seed))
    assert len(store.samples["lumberjack"]) == cd.LEARNED_MAX_SAMPLES
    assert store.is_active("lumberjack")  # the agreeing pair survived


def test_sample_close_to_another_cards_art_is_rejected():
    store = cd.LearnedCardStore(None)
    store.add("hero_balloon", _hists(1))
    store.add("hero_balloon", _hists(1, scale=10))
    store.add("furnace", _hists(1, scale=20))
    assert "furnace" not in store.samples


def test_greyed_slot_is_never_learned():
    assert cd.learn_card_sample("hero_balloon", _hists(1), chroma=5.0) is False
    assert cd.learn_card_sample("hero_balloon", _hists(1), chroma=cd.CARD_SOFT_MIN_CHROMA - 0.1) is False
    assert cd.learned_store().samples == {}
    assert cd.learn_card_sample("UNKNOWN", _hists(1), chroma=80.0) is False


def test_matcher_returns_a_learned_card():
    frame = cv2.imread(str(YELLOW_ARENA_DAMAGED))
    assert frame is not None
    before = {s: c for s, c, _b, _s2 in cd.identify_hand_cards_scored(frame)}
    assert before[0] == "UNKNOWN" and before[1] == "ice_spirit"
    art = cd.slot_histograms(frame, 0)
    store = cd.LearnedCardStore(None)
    store.add("hero_balloon", art)
    cd.set_learned_store(store)
    assert {s: c for s, c, _b, _s2 in cd.identify_hand_cards_scored(frame)}[0] == "UNKNOWN"  # 1 sample
    nudged = art.copy()
    nudged[0, int(np.argmax(nudged[0]))] -= 5
    nudged[0, int(np.argmin(nudged[0]))] += 5
    store.add("hero_balloon", nudged)
    after = {s: (c, best) for s, c, best, _s2 in cd.identify_hand_cards_scored(frame)}
    assert after[0][0] == "hero_balloon" and after[0][1] <= cd.CARD_MATCH_THRESHOLD
    assert after[1][0] == "ice_spirit"  # static matches are untouched


@pytest.mark.parametrize("value", ["off", "0", ""])
def test_env_can_turn_persistence_off(monkeypatch, value):
    monkeypatch.setattr(cd, "LEARNED_CARDS_PATH", None)
    monkeypatch.setenv("PYCLASHBOT_LEARNED_CARDS", value)
    assert cd.learned_cards_path() is None


def test_default_path_is_under_app_data(monkeypatch):
    monkeypatch.setattr(cd, "LEARNED_CARDS_PATH", None)
    monkeypatch.delenv("PYCLASHBOT_LEARNED_CARDS", raising=False)
    path = cd.learned_cards_path()
    assert path is not None and path.endswith(cd.LEARNED_CARDS_FILE) and "py-clash-bot" in path
