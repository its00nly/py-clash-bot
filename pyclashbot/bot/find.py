"""Pure screen-detection helpers.

Imports only from pyclashbot.detection.*, pyclashbot.utils.*, and pyclashbot.bot.coords.
Never imports from other pyclashbot.bot.* modules, with one exception:
detect_upgradable_cards reuses pixel_indicates_upgradable from state_detect (a leaf
predicate module that does not import from find).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from pyclashbot.bot.coords import (
    ENEMY_TOWER_HP_BAR_Y_BAND,
    ENEMY_TOWER_HP_BARS,
    ENEMY_UNIT_CROP_OFFSETS_LTRB,
    ENEMY_UNIT_SCAN_REGION_LTRB,
    FRIENDLY_UNIT_SCAN_REGION_LTRB,
    LANE_SPLIT_X,
    OWN_TOWER_HP_BAR_Y_BAND,
    OWN_TOWER_HP_BARS,
    UPGRADE_POINTS,
)
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
# Marker blobs this close belong to one unit (badge + HP bar + occlusion fragments): merged.
_ENEMY_MERGE_PX = 22
# A merged unit smaller than this is a speck (one or two matching pixels, ~9 px after dilation):
# on the overnight run 60% of "defensive" plays answered such specks, mostly on pink troop bodies.
_MIN_ENEMY_UNIT_AREA = 20
# The badge floats above the unit; shift down so the point lands on the troop itself.
_BADGE_TO_BODY_DY = 15


@dataclass(frozen=True)
class EnemyUnit:
    """An enemy troop seen on our half of the arena (screen coords)."""

    x: int
    y: int
    size: int  # marker pixel count — a rough proxy for how big/healthy the unit is
    # What the unit is (a card id or a coarse unit_classifier category), once recognized.
    kind: str | None = None

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
    blobs = sorted(
        (
            (int(stats[i][cv2.CC_STAT_AREA]), int(centroids[i][0]) + left, int(centroids[i][1]) + top)
            for i in range(1, count)
            if stats[i][cv2.CC_STAT_AREA] >= _MIN_ENEMY_MARKER_AREA
        ),
        reverse=True,
    )
    # Greedy merge, biggest blob first: it anchors the unit's position, fragments add to its size.
    clusters: list[list[int]] = []  # [x, y, total_area]
    for area, x, y in blobs:
        for cluster in clusters:
            if abs(cluster[0] - x) <= _ENEMY_MERGE_PX and abs(cluster[1] - y) <= _ENEMY_MERGE_PX:
                cluster[2] += area
                break
        else:
            clusters.append([x, y, area])
    units = [
        EnemyUnit(x=x, y=y + _BADGE_TO_BODY_DY, size=area) for x, y, area in clusters if area >= _MIN_ENEMY_UNIT_AREA
    ]
    return sorted(units, key=lambda u: u.y, reverse=True)


def crop_unit(iar: np.ndarray | None, unit: EnemyUnit) -> np.ndarray | None:
    """Fixed-size BGR crop around a detected unit (badge + body), edge-padded at the screen border.

    Always the same shape so crops from anywhere on the board are comparable.
    """
    if iar is None:
        return None
    dl, dt, dr, db = ENEMY_UNIT_CROP_OFFSETS_LTRB
    pad = max(-dl, -dt, dr, db)
    padded = cv2.copyMakeBorder(iar, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    x, y = unit.x + pad, unit.y + pad
    return padded[y + dt : y + db, x + dl : x + dr].copy()


# HP-bar fill colors (BGR) for princess towers: enemy pink-red, ours light blue (top rows)
# and a darker blue (lower rows). Measured on the battle fixture and the yellow arena.
_ENEMY_TOWER_FILLS_BGR = (np.array([94, 38, 226]),)
_OWN_TOWER_FILLS_BGR = (np.array([255, 210, 114]), np.array([231, 158, 0]))
_TOWER_FILL_TOLERANCE = 40
# A bar drains from the right, so a live bar always has fill in its first few pixels.
_TOWER_BAR_ANCHOR_PX = 4
# Gaps this wide inside the fill (anti-aliasing, a stray pixel) don't end the run.
_TOWER_BAR_MAX_GAP = 2
# A bar row needs this many fill pixels, on this many consecutive rows (the bar is ~3 rows thick).
_TOWER_BAR_MIN_PX = 2
_TOWER_BAR_MIN_ROWS = 2
# Rows read from the bar's top: the HP number overlaps the lower rows.
_TOWER_BAR_READ_ROWS = 3


@dataclass(frozen=True)
class TowerHealth:
    """Princess-tower HP as a 0..1 fraction per lane; None = no bar found (covered or destroyed)."""

    enemy: dict[str, float | None]
    own: dict[str, float | None]


def _fill_run(row: np.ndarray, fills: tuple[np.ndarray, ...]) -> int:
    """Length of the fill run from the bar's left end (0 if the left end isn't fill)."""
    pixels = row.astype(int)
    mask = np.zeros(len(pixels), dtype=bool)
    for fill in fills:
        mask |= np.all(np.abs(pixels - fill) <= _TOWER_FILL_TOLERANCE, axis=-1)
    if not mask[:_TOWER_BAR_ANCHOR_PX].any():
        return 0
    end, gap = 0, 0
    for i, filled in enumerate(mask):
        if filled:
            end, gap = i + 1, 0
        else:
            gap += 1
            if gap > _TOWER_BAR_MAX_GAP:
                break
    return end


def _bar_fraction(
    iar: np.ndarray, span: tuple[int, int], band: tuple[int, int], fills: tuple[np.ndarray, ...]
) -> float | None:
    """Find the bar's row inside the y-band by color, then measure its fill; None if no bar."""
    x_left, x_right = span
    top, bottom = max(0, band[0]), min(iar.shape[0], band[1])
    width = x_right - x_left + 1
    runs = [_fill_run(iar[y, x_left : x_right + 1], fills) for y in range(top, bottom)]
    for i in range(len(runs) - _TOWER_BAR_MIN_ROWS + 1):
        if all(run >= _TOWER_BAR_MIN_PX for run in runs[i : i + _TOWER_BAR_MIN_ROWS]):
            return min(1.0, max(runs[i : i + _TOWER_BAR_READ_ROWS]) / width)
    return None


def read_tower_health(iar: np.ndarray | None) -> TowerHealth | None:
    """Read princess-tower HP bars from a BGR battle frame (searches each side's y-band)."""
    if iar is None:
        return None
    return TowerHealth(
        enemy={
            lane: _bar_fraction(iar, (x_left, x_right), ENEMY_TOWER_HP_BAR_Y_BAND, _ENEMY_TOWER_FILLS_BGR)
            for lane, (x_left, x_right, _y) in ENEMY_TOWER_HP_BARS.items()
        },
        own={
            lane: _bar_fraction(iar, (x_left, x_right), OWN_TOWER_HP_BAR_Y_BAND, _OWN_TOWER_FILLS_BGR)
            for lane, (x_left, x_right, _y) in OWN_TOWER_HP_BARS.items()
        },
    )


# Our troops' level badge + HP bar are light blue (BGR). Only the wide bar shape counts:
# tower decorations share the color but are small squares. A bar shows once a troop is
# fighting, which is exactly what "already being handled" needs.
_FRIENDLY_BAR_BGR = (np.array([215, 170, 74]), np.array([255, 250, 154]))
_FRIENDLY_BAR_MIN_W = 14
_FRIENDLY_BAR_MAX_H = 10
_TOWER_BAR_MARGIN = 6


def _on_own_tower_bar(x: int, y: int) -> bool:
    return any(
        x_left - _TOWER_BAR_MARGIN <= x <= x_right + _TOWER_BAR_MARGIN and abs(y - bar_y) <= _TOWER_BAR_MARGIN
        for x_left, x_right, bar_y in OWN_TOWER_HP_BARS.values()
    )


def find_friendly_units(iar: np.ndarray | None) -> list[EnemyUnit]:
    """Locate our own troops that are in a fight (their HP bar is showing)."""
    if iar is None:
        return []
    left, top, right, bottom = FRIENDLY_UNIT_SCAN_REGION_LTRB
    crop = iar[top:bottom, left:right]
    if crop.size == 0:
        return []
    mask = cv2.dilate(cv2.inRange(crop, *_FRIENDLY_BAR_BGR), np.ones((3, 3), np.uint8))
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    units = []
    for i in range(1, count):
        width, height = stats[i][cv2.CC_STAT_WIDTH], stats[i][cv2.CC_STAT_HEIGHT]
        if width < _FRIENDLY_BAR_MIN_W or height > _FRIENDLY_BAR_MAX_H:
            continue
        x, y = int(centroids[i][0]) + left, int(centroids[i][1]) + top
        if _on_own_tower_bar(x, y):
            continue
        units.append(EnemyUnit(x=x, y=y + _BADGE_TO_BODY_DY, size=int(stats[i][cv2.CC_STAT_AREA])))
    return units
