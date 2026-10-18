"""find_enemy_units merges marker fragments into one unit and drops lone specks."""

import numpy as np

from pyclashbot.bot import find

BADGE_BGR = (35, 5, 110)


def _arena() -> np.ndarray:
    return np.full((633, 419, 3), (60, 200, 200), dtype=np.uint8)  # yellow-ish grass, no markers


def test_fragments_merge_into_one_unit():
    iar = _arena()
    iar[300:307, 100:107] = BADGE_BGR  # the badge
    iar[310, 112] = BADGE_BGR  # an occlusion fragment of the same unit
    units = find.find_enemy_units(iar)
    assert len(units) == 1
    assert abs(units[0].x - 103) <= 2 and units[0].size > 81


def test_lone_speck_is_ignored():
    iar = _arena()
    iar[350, 250] = BADGE_BGR  # one stray pixel (~9 px once dilated)
    assert find.find_enemy_units(iar) == []


def test_two_separate_units_stay_separate():
    iar = _arena()
    iar[300:307, 100:107] = BADGE_BGR
    iar[400:407, 300:307] = BADGE_BGR
    assert len(find.find_enemy_units(iar)) == 2
