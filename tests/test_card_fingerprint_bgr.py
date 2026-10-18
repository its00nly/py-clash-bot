"""Regression: fingerprints and bot both use BGR emulator screenshots."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from pyclashbot.bot import card_detection as cd

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "card_hand_bgr_regression.png"

EXPECTED = {
    0: "lumberjack",
    1: "ram_rider",
    2: "ice_wizard",
    3: "hero_bowler",
}


@pytest.fixture(autouse=True)
def _restore_module_state(monkeypatch):
    # identify_hand_cards_scored / check_which_cards_are_available set these module globals.
    monkeypatch.setattr(cd, "battle_iar", cd.battle_iar)
    monkeypatch.setattr(cd, "play_side", cd.play_side)


class _BgrEmulator:
    def __init__(self, bgr: np.ndarray) -> None:
        self._bgr = bgr

    def screenshot(self) -> np.ndarray:
        return self._bgr


def _load_bgr() -> np.ndarray:
    if not FIXTURE.is_file():
        raise AssertionError(f"missing fixture: {FIXTURE}")
    rgb = np.array(Image.open(FIXTURE).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def test_bot_uses_raw_bgr_screenshot() -> None:
    bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr[0, 0] = [10, 20, 30]
    out = _BgrEmulator(bgr).screenshot()
    assert out[0, 0].tolist() == [10, 20, 30]


def test_bgr_path_identifies_hand() -> None:
    bgr = _load_bgr()
    cd.check_which_cards_are_available(_BgrEmulator(bgr))
    for slot, expected in EXPECTED.items():
        corners = cd.get_all_pixel_data(None, slot)
        raw = cd.find_closest_card(corners)
        got = cd.CARD_DETECTION_ALIASES.get(raw, raw)
        assert got == expected, f"slot {slot}: expected {expected}, got {got} (raw {raw})"


# ---- scored matcher: raised / dimmed / greyed hands -------------------------------
def _raised(bgr: np.ndarray, px: int) -> np.ndarray:
    out = bgr.copy()
    for x, y in cd.toplefts:
        out[y - px : y - px + cd.TOTAL_HEIGHT, x : x + cd.TOTAL_WIDTH] = bgr[
            y : y + cd.TOTAL_HEIGHT, x : x + cd.TOTAL_WIDTH
        ]
    return out


def _gain(bgr: np.ndarray, g: float) -> np.ndarray:
    return np.clip(bgr.astype(np.float32) * g, 0, 255).astype(np.uint8)


def _grey(bgr: np.ndarray, amount: float = 1.0, g: float = 1.0) -> np.ndarray:
    grey = cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR).astype(np.float32)
    return np.clip(((1 - amount) * bgr + amount * grey) * g, 0, 255).astype(np.uint8)


def _scored(frame: np.ndarray, monkeypatch) -> dict[int, str]:
    saved = []
    monkeypatch.setattr(cd, "_save_unknown_crop", lambda *a: saved.append(a[1]))
    out = {slot: card for slot, card, _best, _second in cd.identify_hand_cards_scored(frame)}
    # Only UNKNOWN slots still in colour are queued; greyed (unaffordable) ones are skipped.
    in_colour = {s for s in out if cd._slot_chroma(frame, s) >= cd.CARD_SOFT_MIN_CHROMA}
    assert sorted(saved) == sorted(s for s, c in out.items() if c == "UNKNOWN" and s in in_colour)
    return out


def test_scored_identifies_normal_hand() -> None:
    bgr = _load_bgr()
    results = cd.identify_hand_cards_scored(bgr)
    assert cd.battle_iar is bgr
    assert {slot: card for slot, card, _b, _s in results} == EXPECTED
    for _slot, _card, best, second in results:
        assert best == 0 and second > best


def test_scored_subset_of_slots() -> None:
    results = cd.identify_hand_cards_scored(_load_bgr(), slots=(2,))
    assert [(s, c) for s, c, _b, _s in results] == [(2, "ice_wizard")]


def test_raised_cards_still_identified(monkeypatch) -> None:
    for px in (8, 12):
        assert _scored(_raised(_load_bgr(), px), monkeypatch) == EXPECTED, px


def test_identify_hand_cards_uses_search() -> None:
    cd.battle_iar = _raised(_load_bgr(), 8)
    assert [cd.identify_hand_cards(None, s) for s in range(4)] == [EXPECTED[s] for s in range(4)]


def test_brightness_090_still_identified(monkeypatch) -> None:
    assert _scored(_gain(_load_bgr(), 0.9), monkeypatch) == EXPECTED


def test_degraded_hand_is_unknown_never_wrong(monkeypatch) -> None:
    bgr = _load_bgr()
    assert set(_scored(_gain(bgr, 0.8), monkeypatch).values()) == {"UNKNOWN"}
    assert set(_scored(_grey(bgr), monkeypatch).values()) == {"UNKNOWN"}
    for amount in (0.0, 0.25, 0.5, 0.75, 1.0):
        for g in (0.9, 0.8, 0.7):
            for px in (0, 8):
                frame = _grey(bgr, amount, g)
                if px:
                    frame = _raised(frame, px)
                for slot, card in _scored(frame, monkeypatch).items():
                    assert card in {EXPECTED[slot], "UNKNOWN"}, (amount, g, px, slot, card)


# ---- frame predicates ----------------------------------------------------------
def test_hero_ability_ready_pure() -> None:
    bgr = _load_bgr()
    assert cd.hero_ability_ready(bgr) is False
    lit = bgr.copy()
    for (r, c), color in zip(cd._ABILITY_PIXELS, cd._ABILITY_COLORS, strict=True):
        lit[r, c] = color
    assert cd.hero_ability_ready(lit) is True
    assert cd.is_hero_champion_ability_visible(_BgrEmulator(lit)) is True


def test_cards_available_matches_emulator_wrapper() -> None:
    bgr = _load_bgr()
    cd.battle_iar = None
    assert cd.check_which_cards_are_available(_BgrEmulator(bgr)) == cd.cards_available(bgr)
    assert cd.battle_iar is bgr


def test_switch_side_ignores_noise(monkeypatch) -> None:
    bgr = _load_bgr()
    monkeypatch.setattr(cd, "bridge_iar", bgr)
    monkeypatch.setattr(cd, "battle_iar", bgr.copy())
    assert cd.switch_side()[0] == 0
    rng = np.random.default_rng(0)
    noisy = np.clip(bgr.astype(np.int16) + rng.integers(-3, 4, size=bgr.shape), 0, 255).astype(np.uint8)
    monkeypatch.setattr(cd, "battle_iar", noisy)
    assert cd.switch_side()[0] < 5000  # fight.py WAIT_THRESHOLD; uint8 wraparound used to exceed it
    busy = noisy.copy()
    busy[200:260, 100:140] = 255 - busy[200:260, 100:140]
    monkeypatch.setattr(cd, "battle_iar", busy)
    offset, side = cd.switch_side()
    assert offset > 5000 and side == "left"


def test_check_side_sets_play_side(monkeypatch) -> None:
    bgr = _load_bgr()
    busy = bgr.copy()
    busy[200:260, 275:315] = 255 - busy[200:260, 275:315]
    monkeypatch.setattr(cd, "bridge_iar", bgr)
    monkeypatch.setattr(cd, "play_side", "left")
    cd.check_which_cards_are_available(_BgrEmulator(busy), check_side=True)
    assert cd.play_side == "right"
