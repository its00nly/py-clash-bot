"""Fingerprint DB integrity and matcher margin rules (offline)."""

from __future__ import annotations

import numpy as np

from pyclashbot.bot import card_detection as cd

CORNER_PIXELS = cd.HALF_WIDTH * cd.HALF_HEIGHT


def test_every_fingerprint_is_4x16_and_sums_to_corner_pixels() -> None:
    assert CORNER_PIXELS == 891
    for name, corners in cd.card_color_data.items():
        arr = np.asarray(corners)
        assert arr.shape == (4, len(cd.COLORS)), name
        assert (arr.sum(axis=1) == CORNER_PIXELS).all(), f"{name}: corner sums {arr.sum(axis=1).tolist()}"


def test_percentage_fingerprints_are_disabled() -> None:
    disabled = cd._DISABLED_PERCENT_FINGERPRINTS
    assert len(disabled) == 11
    for name, corners in disabled.items():
        assert name not in cd.card_color_data
        assert all(sum(corner.values()) == 100 for corner in corners), name


def test_alias_targets_exist() -> None:
    known = set(cd.card_color_data) | set(cd.CARD_TO_GROUP)
    for key, target in cd.CARD_DETECTION_ALIASES.items():
        assert target in known, f"{key} -> {target}"
    assert cd.CARD_DETECTION_ALIASES["evo_furnace"] == "furnace"


def _accepts(best: int, second: int) -> bool:
    return best <= cd.CARD_MATCH_THRESHOLD or (
        best <= cd.CARD_MATCH_SOFT_THRESHOLD and second >= cd.CARD_MATCH_MIN_RATIO * best
    )


def test_leave_one_out_at_most_one_false_accept() -> None:
    names, db, bases = cd._get_db()
    false_accepts = []
    for i, name in enumerate(names):
        keep = bases != bases[i]  # drop the card and every variant of it
        dists = np.abs(db[keep] - db[i][None]).sum(axis=(1, 2))
        kept_bases = bases[keep]
        j = int(np.argmin(dists))
        best = int(dists[j])
        second = int(dists[kept_bases != kept_bases[j]].min())
        if _accepts(best, second):
            false_accepts.append(name)
    assert len(false_accepts) <= 1, false_accepts


def _sibling_pairs(names: list[str]) -> list[tuple[str, str]]:
    pairs = [("spirit_empress_air", "spirit_empress_ground"), ("evo_furnace", "furnace")]
    pairs += [(n, n[len("evo_") :]) for n in names if n.startswith("evo_") and n[len("evo_") :] in names]
    return [(a, b) for a, b in pairs if a in names and b in names]


def test_runner_up_never_shares_base_id() -> None:
    names, _db, _bases = cd._get_db()
    pairs = _sibling_pairs(names)
    assert ("spirit_empress_air", "spirit_empress_ground") in pairs
    assert len(pairs) > 5
    for a, b in pairs + [(b, a) for a, b in pairs]:
        assert cd.base_card_id(a) == cd.base_card_id(b), (a, b)
        x = np.asarray(cd.card_color_data[a])
        card, best, second, top3 = cd.find_closest_card_scored(x)
        assert (card, best) == (a, 0)
        assert len(top3) == 3 and top3[0] == (a, 0)
        expected = min(
            int(np.abs(np.asarray(cd.card_color_data[n]) - x).sum())
            for n in names
            if cd.base_card_id(n) != cd.base_card_id(a)
        )
        assert second == expected, (a, b)


def test_find_closest_card_wrapper_matches_scored() -> None:
    corners = cd.card_color_data["lumberjack"]
    dicts = [dict(zip(cd.COLORS_KEYS, map(int, row), strict=True)) for row in corners]
    assert cd.find_closest_card(dicts) == "lumberjack"
    assert cd.find_closest_card_scored(dicts)[0] == "lumberjack"


def test_soft_accept_requires_margin() -> None:
    assert _accepts(1000, 1000)
    assert _accepts(1500, int(1500 * cd.CARD_MATCH_MIN_RATIO))
    assert not _accepts(1500, int(1500 * cd.CARD_MATCH_MIN_RATIO) - 1)
    assert not _accepts(cd.CARD_MATCH_SOFT_THRESHOLD + 1, 10**6)
