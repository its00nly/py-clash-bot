"""Frame-taking battle predicates agree with their emulator-taking wrappers."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from pyclashbot.bot import card_detection, state_detect

FIXTURE = Path(__file__).parent / "fixtures" / "card_hand_bgr_regression.png"


class _Emu:
    def __init__(self, image):
        self.image = image

    def screenshot(self):
        return self.image


@pytest.fixture
def frame():
    image = cv2.imread(str(FIXTURE))
    assert image is not None
    return image


def _synthetic(in_battle: bool = True, elixir: int = 0, ability: bool = False) -> np.ndarray:
    iar = np.zeros((633, 419, 3), dtype=np.uint8)
    if in_battle:
        for y, x in [(528, 49), (532, 77), (546, 52), (546, 77)]:
            iar[y, x] = (255, 255, 255)
        iar[618, 115] = (220, 50, 220)  # scoreboard purple (same in BGR and RGB)
    for y, x in state_detect.ELIXIR_COORDS[:elixir]:
        iar[y, x] = state_detect.ELIXIR_COLOR
    if ability:
        iar[462, 324] = (215, 28, 223)
    return iar


def test_fixture_is_in_battle(frame):
    assert state_detect.in_battle_from_frame(frame) is True
    assert state_detect.check_if_in_battle(_Emu(frame)) is True


def test_none_frame_is_not_in_battle():
    assert state_detect.in_battle_from_frame(None) is False
    assert state_detect.hero_ability_ready(None) is False


@pytest.mark.parametrize("in_battle", [True, False])
def test_in_battle_wrapper_matches_frame_predicate(in_battle):
    iar = _synthetic(in_battle=in_battle)
    assert state_detect.in_battle_from_frame(iar) is in_battle
    assert state_detect.check_if_in_battle(_Emu(iar)) is in_battle


def test_elixir_reads_match_wrapper(frame):
    assert state_detect.read_elixir(frame) == 5
    for n in range(1, 11):
        assert state_detect.count_elixir(_Emu(frame), n) == state_detect.elixir_at_least(frame, n)


@pytest.mark.parametrize("elixir", [0, 1, 4, 7, 10])
def test_read_elixir_on_synthetic_frames(elixir):
    assert state_detect.read_elixir(_synthetic(elixir=elixir)) == elixir


@pytest.mark.parametrize("ability", [True, False])
def test_hero_ability_ready_matches_card_detection(ability):
    iar = _synthetic(ability=ability)
    assert state_detect.hero_ability_ready(iar) is ability
    assert card_detection.is_hero_champion_ability_visible(_Emu(iar)) is ability


def test_hero_ability_ready_matches_card_detection_on_fixture(frame):
    expected = bool(card_detection.is_hero_champion_ability_visible(_Emu(frame)))
    assert state_detect.hero_ability_ready(frame) is expected
