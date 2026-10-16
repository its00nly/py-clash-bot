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

import collections
import random
import statistics
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyclashbot.bot.card_detection import calculate_play_coords, get_card_group
from pyclashbot.bot.coords import (
    DEFENSE_BUILDING_PULL_COORDS,
    LANE_SPLIT_X,
    OPENING_BACK_COORDS,
    OWN_HALF_DROP_Y_RANGE,
    OWN_TOWER_FOOTPRINTS_LTRB,
    PLAYABLE_PLAY_REGION_LTRB,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pyclashbot.bot.find import EnemyUnit, TowerHealth

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
_OFFENSE_PRIORITY = (ROLE_WIN, ROLE_SUPPORT, ROLE_TROOP)
# Right after a win condition goes down: back it up instead of stacking another one.
_FOLLOW_UP_PRIORITY = (ROLE_SUPPORT, ROLE_TROOP, ROLE_WIN)
# Opening play: something cheap at the back, never a naked win condition at the bridge.
_OPENING_PRIORITY = (ROLE_SUPPORT, ROLE_TROOP)
# Only played when elixir is about to leak (spells also when they can finish a tower / chip late).
_LEAK_PRIORITY = (ROLE_SPELL, ROLE_BUILDING, ROLE_UTILITY, ROLE_OWN_SIDE_SPELL)

# Melee/tank defenders go just behind the enemy so it walks into them; ranged support further back.
_TROOP_DEFENSE_DY = 45
_SUPPORT_DEFENSE_DY = 85
# Spells land a little ahead of the enemy (it keeps walking toward our towers).
_SPELL_LEAD_DY = 12
# After defending a lane, ignore further pings there for this long unless the push gets deeper.
DEFENSE_COOLDOWN_S = 4.0
# ...shorter when that lane's tower is already low.
WEAK_TOWER_DEFENSE_COOLDOWN_S = 2.0
_DEEPER_PUSH_DY = 40
# How long offense keeps favoring the lane we just defended.
COUNTER_PUSH_WINDOW_S = 12.0
# How long after a win condition we prefer backing it up with support.
SUPPORT_FOLLOW_UP_S = 8.0
# The opening play happens within this window (later first plays are just normal offense).
OPENING_WINDOW_S = 25.0
# From here on (overtime-ish) spells are worth throwing at towers for chip damage.
SPELL_CHIP_AFTER_S = 150.0

# Tower health (0..1 fractions of the HP bar, median of recent reads).
WEAK_TOWER_HP = 0.4
FINISH_TOWER_HP = 0.25
LANE_SWITCH_HP_MARGIN = 0.15
_TOWER_HISTORY = 5


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


def _other(lane: str) -> str:
    return "left" if lane == "right" else "right"


class BattleBrain:
    """Per-battle state: which lane we attack, recent plays, defense cooldowns, tower health."""

    def __init__(self, clock=time.time, rng: random.Random | None = None) -> None:
        self._clock = clock
        self._rng = rng or random.Random()
        self.attack_lane = self._rng.choice(("left", "right"))
        self._counter_push_lane: str | None = None
        self._counter_push_until = 0.0
        self._last_defense: dict[str, tuple[float, int]] = {}  # lane -> (time, threat y)
        self._last_win: tuple[str, float] | None = None  # lane, time of our last win-condition drop
        self._played_any = False
        self._enemy_hp: dict[str, collections.deque[float]] = {
            lane: collections.deque(maxlen=_TOWER_HISTORY) for lane in ("left", "right")
        }
        self._own_hp: dict[str, collections.deque[float]] = {
            lane: collections.deque(maxlen=_TOWER_HISTORY) for lane in ("left", "right")
        }
        self.recent: list[int] = []

    def _pick(self, candidates: list[tuple[int, str, str]]) -> tuple[int, str, str]:
        """Prefer cards not played recently (keeps the cycle moving), then random."""
        fresh = [c for c in candidates if c[0] not in self.recent]
        return self._rng.choice(fresh or candidates)

    # ---- towers --------------------------------------------------------------

    def observe_towers(self, health: TowerHealth | None) -> None:
        """Feed one tower-HP read; a median over recent reads smooths troops covering the bars."""
        if health is None:
            return
        for lane in ("left", "right"):
            self._enemy_hp[lane].append(health.enemy.get(lane, 0.0))
            self._own_hp[lane].append(health.own.get(lane, 0.0))

    def enemy_tower_hp(self, lane: str) -> float | None:
        reads = self._enemy_hp[lane]
        return statistics.median(reads) if reads else None

    def own_tower_hp(self, lane: str) -> float | None:
        reads = self._own_hp[lane]
        return statistics.median(reads) if reads else None

    def _own_tower_weak(self, lane: str) -> bool:
        hp = self.own_tower_hp(lane)
        return hp is not None and hp < WEAK_TOWER_HP

    # ---- lanes ---------------------------------------------------------------

    def offense_lane(self) -> str:
        if self._counter_push_lane and self._clock() < self._counter_push_until:
            return self._counter_push_lane
        # Pile onto the weaker enemy tower once one side is clearly ahead.
        left, right = self.enemy_tower_hp("left"), self.enemy_tower_hp("right")
        if left is not None and right is not None:
            if left + LANE_SWITCH_HP_MARGIN < right:
                return "left"
            if right + LANE_SWITCH_HP_MARGIN < left:
                return "right"
        return self.attack_lane

    def should_defend_now(self, threats: Sequence[EnemyUnit]) -> EnemyUnit | None:
        """Return the threat to answer immediately, or None if it's already handled.

        Threats against a low tower come first; otherwise the deepest one.
        """
        now = self._clock()
        ordered = sorted(threats, key=lambda t: (not self._own_tower_weak(t.lane), -t.y))
        for threat in ordered:
            last = self._last_defense.get(threat.lane)
            if last is None:
                return threat
            when, defended_y = last
            cooldown = WEAK_TOWER_DEFENSE_COOLDOWN_S if self._own_tower_weak(threat.lane) else DEFENSE_COOLDOWN_S
            if now - when >= cooldown or threat.y > defended_y + _DEEPER_PUSH_DY:
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
            lane = _other(threat.lane)
            priorities, reason = _OFFENSE_PRIORITY, f"pressuring the {lane} lane"
        elif self._is_opening(elapsed_s) and any(c[2] in _OPENING_PRIORITY for c in cards):
            lane = self.offense_lane()
            index, card_id, picked = self._pick([c for c in cards if c[2] in _OPENING_PRIORITY])
            return Play(index, card_id, picked, OPENING_BACK_COORDS[lane], lane, False, "opening at the back")
        elif self._last_win is not None and self._clock() - self._last_win[1] < SUPPORT_FOLLOW_UP_S:
            lane = self._last_win[0]
            priorities, reason = _FOLLOW_UP_PRIORITY, f"backing up the push in the {lane} lane"
        else:
            lane = self.offense_lane()
            priorities, reason = _OFFENSE_PRIORITY, f"pushing the {lane} lane"

        enemy_hp = self.enemy_tower_hp(lane)
        finishing = enemy_hp is not None and 0 < enemy_hp <= FINISH_TOWER_HP
        spell_worth_it = finishing or elapsed_s >= SPELL_CHIP_AFTER_S
        if spell_worth_it:
            priorities = (*priorities, ROLE_SPELL)
        if elixir_full:
            priorities = (*priorities, *_LEAK_PRIORITY)

        for role in dict.fromkeys(priorities):  # de-dupe, keep order
            options = [c for c in cards if c[2] == role]
            if not options:
                continue
            index, card_id, picked = self._pick(options)
            coord = self._offense_drop(card_id, picked, lane, elapsed_s)
            if coord is None:
                continue
            if picked == ROLE_SPELL and spell_worth_it:
                why = f"finishing the {lane} tower" if finishing else "chipping the tower"
            elif picked in _LEAK_PRIORITY:
                why = "spending elixir before it leaks"
            else:
                why = reason
            return Play(index, card_id, picked, coord, lane, False, why)
        return None

    def _is_opening(self, elapsed_s: float) -> bool:
        return not self._played_any and elapsed_s < OPENING_WINDOW_S

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
        self._played_any = True
        self.recent = [*self.recent[-2:], play.index]
        now = self._clock()
        if play.defensive:
            self._last_defense[play.lane] = (now, play.coord[1])
            self._counter_push_lane = play.lane
            self._counter_push_until = now + COUNTER_PUSH_WINDOW_S
        elif play.role == ROLE_WIN:
            self._last_win = (play.lane, now)
