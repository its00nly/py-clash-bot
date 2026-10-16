"""Smart battle AI: decide which card to play and where from what's on the board.

The default fight loop waits for a random elixir amount and plays a random card.
This module adds the basics a human does to win more:

* **Defend pushes.** Enemy troops crossing onto our half (see ``find_enemy_units``)
  get answered right away with a defensive troop, building, or spell placed on them.
* **Counter-push.** After defending a lane, offense follows up in that same lane so
  the surviving defenders support the push.
* **Card roles.** Win conditions are saved for offense; buildings and defensive
  spells are held for defense instead of being thrown away on an empty board.
* **No elixir leak.** At full elixir something is always played.

Pure decision logic over (hand, enemy units, time) so it is unit-testable offline;
``fight.py`` owns the screenshots and clicks.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyclashbot.bot.card_detection import calculate_play_coords, get_card_group
from pyclashbot.bot.coords import (
    DEFENSE_BUILDING_PULL_COORDS,
    LANE_SPLIT_X,
    OWN_HALF_DROP_Y_RANGE,
    OWN_TOWER_FOOTPRINTS_LTRB,
    PLAYABLE_PLAY_REGION_LTRB,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pyclashbot.bot.find import EnemyUnit

ROLE_WIN = "win"
ROLE_TROOP = "troop"
ROLE_SUPPORT = "support"
ROLE_BUILDING = "building"
ROLE_SPELL = "spell"
ROLE_OWN_SIDE_SPELL = "own_side_spell"
ROLE_UTILITY = "utility"

# Cards whose job is to damage towers; played on offense, not as defenders.
WIN_CONDITIONS = frozenset(
    {
        "balloon",
        "battle_ram",
        "electro_giant",
        "elixir_golem",
        "giant",
        "goblin_barrel",
        "goblin_drill",
        "goblin_giant",
        "golem",
        "graveyard",
        "hog",
        "lava_hound",
        "miner",
        "mortar",
        "ram_rider",
        "royal_giant",
        "royal_hogs",
        "rune_giant",
        "skeleton_barrel",
        "suspicious_bush",
        "wall_breakers",
        "xbow",
    }
)
# Damage/control spells: cast on an enemy push, or at a tower to chip.
DAMAGE_SPELLS = frozenset(
    {
        "arrows",
        "earthquake",
        "fireball",
        "freeze",
        "gob_curse",
        "lightning",
        "log",
        "poison",
        "rocket",
        "snowball",
        "tornado",
        "vines",
        "void",
        "zap",
    }
)
# Spell-like cards that can only be dropped on our own side — great on defense only.
OWN_SIDE_SPELLS = frozenset({"barb_barrel", "royal_delivery"})
# Cards with no standalone value on an empty board or against a push.
UTILITY_CARDS = frozenset({"clone", "elixir_collector", "heal_spirit", "mirror", "rage"})
BUILDINGS = frozenset({"barb_hut", "furnace", "goblin_hut", "tombstone"})
# Placement groups for ranged/support troops that should sit behind the action.
_SUPPORT_GROUPS = frozenset({"back_support", "king_lane", "princess"})

_DEFENSE_PRIORITY = (ROLE_TROOP, ROLE_SUPPORT, ROLE_BUILDING, ROLE_SPELL, ROLE_OWN_SIDE_SPELL)
_OFFENSE_PRIORITY = (ROLE_WIN, ROLE_SUPPORT, ROLE_TROOP, ROLE_SPELL)
# Only played on an empty board to avoid leaking elixir.
_LEAK_PRIORITY = (ROLE_BUILDING, ROLE_UTILITY, ROLE_OWN_SIDE_SPELL)

# Melee/tank defenders go just behind the enemy so it walks into them; ranged support further back.
_TROOP_DEFENSE_DY = 45
_SUPPORT_DEFENSE_DY = 85
# Spells land a little ahead of the enemy (it keeps walking toward our towers).
_SPELL_LEAD_DY = 12
# After defending a lane, ignore further pings there for this long unless the push gets deeper.
DEFENSE_COOLDOWN_S = 4.0
_DEEPER_PUSH_DY = 40
# How long offense keeps favoring the lane we just defended.
COUNTER_PUSH_WINDOW_S = 12.0


def _base_card_id(card_id: str) -> str:
    for prefix in ("evo_", "hero_"):
        if card_id.startswith(prefix):
            return card_id[len(prefix) :]
    return card_id


def card_role(card_id: str | None) -> str:
    """Classify a detected card id into the role the brain plays it as."""
    if not card_id or card_id == "UNKNOWN":
        return ROLE_TROOP
    base = _base_card_id(card_id)
    group = get_card_group(card_id)
    if base in WIN_CONDITIONS:
        return ROLE_WIN
    if base in UTILITY_CARDS:
        return ROLE_UTILITY
    if base in OWN_SIDE_SPELLS:
        return ROLE_OWN_SIDE_SPELL
    if base in DAMAGE_SPELLS:
        return ROLE_SPELL
    if base in BUILDINGS or group == "defense_building":
        return ROLE_BUILDING
    if group in _SUPPORT_GROUPS:
        return ROLE_SUPPORT
    return ROLE_TROOP


def lane_of(x: float) -> str:
    return "left" if x < LANE_SPLIT_X else "right"


def _clamp(value: float, lo: float, hi: float) -> int:
    return int(max(lo, min(hi, value)))


def _avoid_own_towers(x: int, y: int) -> tuple[int, int]:
    """Troops can't be dropped on a tower; nudge the drop to just in front of it."""
    for left, top, right, bottom in OWN_TOWER_FOOTPRINTS_LTRB:
        if left <= x <= right and top <= y <= bottom:
            return x, max(OWN_HALF_DROP_Y_RANGE[0], top - 8)
    return x, y


def defensive_drop(role: str, threat: EnemyUnit) -> tuple[int, int]:
    """Where to put a card of ``role`` to answer ``threat``."""
    left, top, right, bottom = PLAYABLE_PLAY_REGION_LTRB
    y_min, y_max = OWN_HALF_DROP_Y_RANGE
    if role == ROLE_BUILDING:
        return DEFENSE_BUILDING_PULL_COORDS[threat.lane]
    if role in (ROLE_SPELL, ROLE_OWN_SIDE_SPELL):
        spell_top = y_min if role == ROLE_OWN_SIDE_SPELL else top
        return _clamp(threat.x, left, right), _clamp(threat.y + _SPELL_LEAD_DY, spell_top, bottom)
    dy = _SUPPORT_DEFENSE_DY if role == ROLE_SUPPORT else _TROOP_DEFENSE_DY
    x = _clamp(threat.x, left, right)
    y = _clamp(threat.y + dy, y_min, y_max)
    return _avoid_own_towers(x, y)


@dataclass(frozen=True)
class Play:
    index: int
    card_id: str
    role: str
    coord: tuple[int, int]
    lane: str
    defensive: bool
    reason: str


class BattleBrain:
    """Per-battle state: which lane we attack, recent plays, defense cooldowns."""

    def __init__(self, clock=time.time, rng: random.Random | None = None) -> None:
        self._clock = clock
        self._rng = rng or random.Random()
        self.attack_lane = self._rng.choice(("left", "right"))
        self._counter_push_lane: str | None = None
        self._counter_push_until = 0.0
        self._last_defense: dict[str, tuple[float, int]] = {}  # lane -> (time, threat y)
        self.recent: list[int] = []

    def _pick(self, candidates: list[tuple[int, str, str]]) -> tuple[int, str, str]:
        """Prefer cards not played recently (keeps the cycle moving), then random."""
        fresh = [c for c in candidates if c[0] not in self.recent]
        return self._rng.choice(fresh or candidates)

    # ---- lanes ---------------------------------------------------------------

    def offense_lane(self) -> str:
        if self._counter_push_lane and self._clock() < self._counter_push_until:
            return self._counter_push_lane
        return self.attack_lane

    def should_defend_now(self, threats: Sequence[EnemyUnit]) -> EnemyUnit | None:
        """Return the threat to answer immediately, or None if it's already handled."""
        now = self._clock()
        for threat in threats:  # deepest first
            last = self._last_defense.get(threat.lane)
            if last is None:
                return threat
            when, defended_y = last
            if now - when >= DEFENSE_COOLDOWN_S or threat.y > defended_y + _DEEPER_PUSH_DY:
                return threat
        return None

    # ---- decisions -----------------------------------------------------------

    def choose(
        self,
        hand: Mapping[int, str],
        threats: Sequence[EnemyUnit],
        elapsed_s: float,
        elixir_full: bool = False,
    ) -> Play | None:
        """Pick a play from the playable ``hand`` (index -> card id), or None to hold."""
        if not hand:
            return None
        cards = [(index, card_id, card_role(card_id)) for index, card_id in hand.items()]

        threat = self.should_defend_now(threats) if threats else None
        if threat is not None:
            for role in _DEFENSE_PRIORITY:
                options = [c for c in cards if c[2] == role]
                if options:
                    index, card_id, picked = self._pick(options)
                    return Play(
                        index,
                        card_id,
                        picked,
                        defensive_drop(picked, threat),
                        threat.lane,
                        True,
                        f"defending the {threat.lane} lane",
                    )
            # Nothing that defends — pressure the other lane instead.
            lane = "left" if threat.lane == "right" else "right"
        else:
            lane = self.offense_lane()

        priorities = _OFFENSE_PRIORITY + _LEAK_PRIORITY if elixir_full else _OFFENSE_PRIORITY
        for role in priorities:
            options = [c for c in cards if c[2] == role]
            if not options:
                continue
            index, card_id, picked = self._pick(options)
            coord = self._offense_drop(card_id, picked, lane, elapsed_s)
            if coord is None:
                continue
            leak = picked in _LEAK_PRIORITY
            reason = "spending elixir before it leaks" if leak else f"pushing the {lane} lane"
            return Play(index, card_id, picked, coord, lane, False, reason)
        return None

    def _offense_drop(self, card_id: str, role: str, lane: str, elapsed_s: float) -> tuple[int, int] | None:
        if role == ROLE_BUILDING:
            return DEFENSE_BUILDING_PULL_COORDS[lane]
        if role == ROLE_OWN_SIDE_SPELL:
            # Drop at the bridge on our side so its troops walk into the lane.
            return (DEFENSE_BUILDING_PULL_COORDS[lane][0] - (60 if lane == "left" else -60), OWN_HALF_DROP_Y_RANGE[0])
        coord = calculate_play_coords(get_card_group(card_id), lane, elapsed_s)
        return tuple(coord) if coord is not None else None

    def record(self, play: Play) -> None:
        """Remember a play we made (call after the clicks went through)."""
        self.recent = [*self.recent[-2:], play.index]
        if play.defensive:
            now = self._clock()
            self._last_defense[play.lane] = (now, play.coord[1])
            self._counter_push_lane = play.lane
            self._counter_push_until = now + COUNTER_PUSH_WINDOW_S
