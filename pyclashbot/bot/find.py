"""Pure screen-detection helpers.

Imports only from pyclashbot.detection.*, pyclashbot.utils.*, and pyclashbot.bot.coords.
Never imports from other pyclashbot.bot.* modules, with one exception:
detect_upgradable_cards reuses pixel_indicates_upgradable from state_detect (a leaf
predicate module that does not import from find).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from pyclashbot.bot.coords import ENEMY_UNIT_SCAN_REGION_LTRB, LANE_SPLIT_X, UPGRADE_POINTS
from pyclashbot.bot.state_detect import pixel_indicates_upgradable
from pyclashbot.detection.image_rec import find_image, pixel_is_equal


def find_war_battle_icon(emulator):
    """Find the war battle icon on the clan war page.

    Returns (x, y) of the icon, or None if not found.
    """
    image = emulator.screenshot()
    return find_image(image, "war_battle_icon", tolerance=0.8, show_image=False)


def find_fight_mode_icon(emulator, mode: str):
    expected_mode_types = ["Classic 1v1", "Classic 2v2", "Trophy Road"]

    # Check if the mode is valid
    if mode not in expected_mode_types:
        print(f'[!] Fatal error: Mode "{mode}" is not a valid mode type. Expected one of {expected_mode_types}.')
        return None

    mode2folder = {
        "Classic 1v1": "fight_mode_1v1",
        "Classic 2v2": "fight_mode_2v2",
        "Trophy Road": "fight_mode_trophy_road",
    }

    look_folder = mode2folder[mode]

    image = emulator.screenshot()

    fight_mode_1v1_button_location = find_image(
        image,
        look_folder,
        tolerance=0.9,
        show_image=False,
    )
    if fight_mode_1v1_button_location is not None:
        return fight_mode_1v1_button_location
    return None


def find_post_battle_button(emulator):
    """Find and return coordinates for post-battle exit/OK button.

    Tries multiple detection methods in order:
    1. Pixel-based detection (fastest)
    2. Image recognition for OK button
    3. Image recognition for exit button

    Returns:
        tuple[int, int] | None: Button coordinates (x, y) or None if not found
    """
    iar = emulator.screenshot()

    pixels = [
        iar[545][178],
        iar[547][239],
        iar[553][214],
        iar[554][201],
    ]
    colors = [
        [255, 187, 104],
        [255, 187, 104],
        [255, 255, 255],
        [255, 255, 255],
    ]

    pixel_match = True
    for i, p in enumerate(pixels):
        if not pixel_is_equal(p, colors[i], tol=20):
            pixel_match = False
            break

    if pixel_match:
        return (200, 550)

    coord = find_image(iar, "ok_post_battle_button", tolerance=0.85)
    if coord is not None:
        return coord

    coord = find_image(iar, "exit_battle_button", tolerance=0.9)
    if coord is not None:
        return coord

    return None


def locate_free_shop_offer_icon(emulator) -> tuple[int, int] | None:
    """Find the free daily shop offer icon anywhere on the current screen.

    Returns (x, y) of the match, or None if not found.
    """
    image = emulator.screenshot()
    return find_image(image, "daily_free_shop_offer_icon", tolerance=0.9)


def detect_upgradable_cards(emulator):
    img = emulator.screenshot()
    upgradable = []

    for i, (x, y) in enumerate(UPGRADE_POINTS, start=1):
        if pixel_indicates_upgradable(img[y][x]):
            upgradable.append(i)

    return upgradable


# Enemy units carry a dark-maroon level badge and a pink HP bar (ours are blue).
# BGR bounds, measured on tests/fixtures/card_hand_bgr_regression.png.
_ENEMY_BADGE_BGR = (np.array([22, 0, 85]), np.array([48, 14, 130]))
_ENEMY_HP_BAR_BGR = (np.array([120, 65, 235]), np.array([165, 110, 255]))
_MIN_ENEMY_MARKER_AREA = 8
# The badge floats above the unit; shift down so the point lands on the troop itself.
_BADGE_TO_BODY_DY = 15


@dataclass(frozen=True)
class EnemyUnit:
    """An enemy troop seen on our half of the arena (screen coords)."""

    x: int
    y: int
    size: int  # marker pixel count — a rough proxy for how big/healthy the unit is

    @property
    def lane(self) -> str:
        return "left" if self.x < LANE_SPLIT_X else "right"


def find_enemy_units(iar: np.ndarray | None) -> list[EnemyUnit]:
    """Locate enemy troops on our side from a BGR battle frame, deepest (closest to us) first."""
    if iar is None:
        return []
    left, top, right, bottom = ENEMY_UNIT_SCAN_REGION_LTRB
    crop = iar[top:bottom, left:right]
    if crop.size == 0:
        return []
    mask = cv2.bitwise_or(cv2.inRange(crop, *_ENEMY_BADGE_BGR), cv2.inRange(crop, *_ENEMY_HP_BAR_BGR))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    units = [
        EnemyUnit(
            x=int(centroids[i][0]) + left,
            y=int(centroids[i][1]) + top + _BADGE_TO_BODY_DY,
            size=int(stats[i][cv2.CC_STAT_AREA]),
        )
        for i in range(1, count)
        if stats[i][cv2.CC_STAT_AREA] >= _MIN_ENEMY_MARKER_AREA
    ]
    return sorted(units, key=lambda u: u.y, reverse=True)
