"""Smart battle AI: decide which card to play and where from what's on the board.

The default fight loop waits for a random elixir amount and plays a random card.
This module plays more like a person:

* **Defend pushes cheaply.** Enemy troops crossing onto our half (``find_enemy_units``)
  are grouped into a push per lane. Swarms get a small spell the moment they cross,
  a lone troop gets the cheapest card that handles it, and a push our troops are
  already fighting is left alone (no Arrows on Minions our Phoenix is shooting).
* **Stuns on purpose.** Zap / Electro Spirit / E-Wiz / E-Dragon / Freeze are kept to
  reset a lone heavy hitter that reaches our tower.
* **Safer placement.** Defenders go where they pull the enemy into tower range,
  ranged support sits off to the side, and swarm cards aren't thrown into multi-unit
  pushes (splash like Bowler/Valkyrie/Wizard eats them).
* **Elixir discipline.** Knows card costs and the current elixir: keeps a defense
  reserve, never dumps troops at the bridge just because, cycles cheap cards to
  bring the win condition back, and never leaks at 10.
* **Real pushes.** Heavy tanks (Golem, Giant, Lava Hound...) start at the back when
  elixir is high and get support behind them; counter-pushes follow a good defense.
* **Memory.** Tracks our own deck and rotation and the opponent's push pattern
  (bridge spam / beatdown / balanced) and adapts reserve and aggression to it.
* **Knows what it's fighting** (once ``unit_classifier`` has a trained model): air
  pushes get anti-air defenders, splash troops never get swarm cards thrown at them,
  tanks get high-DPS answers, and ``OpponentDeck`` remembers the opponent's cards and
  rough cycle to name their deck (Lumberloon, Hog cycle, Golem...) and punish the
  window right after their win condition was played.

Pure decision logic over (hand, units, elixir, time) so it is unit-testable offline;
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
    BACK_TANK_COORDS,
    DEFENSE_BUILDING_PULL_COORDS,
    LANE_SPLIT_X,
    OPENING_BACK_COORDS,
    OWN_HALF_DROP_Y_RANGE,
    OWN_TOWER_FOOTPRINTS_LTRB,
    PLAYABLE_PLAY_REGION_LTRB,
)
from pyclashbot.bot.find import EnemyUnit
from pyclashbot.bot.unit_classifier import CATEGORY_TRAITS, UnitTraits

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pyclashbot.bot.find import TowerHealth

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

# Elixir cost by base card id (evo_/hero_ variants share it). Unknown cards count as 4.
ELIXIR_COST: dict[str, int] = {
    "skeletons": 1, "ice_spirit": 1, "electro_spirit": 1, "fire_spirit": 1, "heal_spirit": 1, "mirror": 1,
    "goblins": 2, "spear_goblins": 2, "bats": 2, "bomber": 2, "ice_golem": 2, "log": 2, "zap": 2,
    "snowball": 2, "barb_barrel": 2, "rage": 2, "wall_breakers": 2, "gob_curse": 2, "berserker": 2,
    "suspicious_bush": 2,
    "archers": 3, "arrows": 3, "knight": 3, "minions": 3, "cannon": 3, "goblin_barrel": 3,
    "skeleton_barrel": 3, "skeleton_army": 3, "tornado": 3, "royal_ghost": 3, "bandit": 3, "guards": 3,
    "dart_goblin": 3, "firecracker": 3, "royal_delivery": 3, "fisherman": 3, "ice_wizard": 3,
    "princess": 3, "miner": 3, "earthquake": 3, "mega_minion": 3, "vines": 3, "void": 3, "clone": 3,
    "tombstone": 3, "elixir_golem": 3, "little_prince": 3, "goblin_gang": 3,
    "valkyrie": 4, "mini_pekka": 4, "musketeer": 4, "hog": 4, "battle_ram": 4, "fireball": 4,
    "baby_dragon": 4, "dark_prince": 4, "lumberjack": 4, "inferno_dragon": 4, "hunter": 4,
    "magic_archer": 4, "night_witch": 4, "mother_witch": 4, "flying_machine": 4, "zappies": 4,
    "battle_healer": 4, "golden_knight": 4, "skeleton_king": 4, "mighty_miner": 4, "phoenix": 4,
    "bomb_tower": 4, "furnace": 4, "freeze": 4, "poison": 4, "tesla": 4, "electro_wizard": 4,
    "goblin_drill": 4, "goblin_cage": 4, "skeleton_dragons": 4, "mortar": 4, "rune_giant": 4,
    "goblin_demolisher": 4,
    "goblin_hut": 5, "cannon_cart": 5, "electro_dragon": 5, "monk": 5, "goblinstein": 5,
    "archer_queen": 5, "royal_hogs": 5, "rascals": 5, "wizard": 5, "witch": 5, "giant": 5,
    "ram_rider": 5, "barbarians": 5, "bowler": 5, "minion_horde": 5, "inferno_tower": 5, "prince": 5,
    "executioner": 5, "balloon": 5, "graveyard": 5,
    "sparky": 6, "giant_skeleton": 6, "goblin_giant": 6, "rocket": 6, "lightning": 6,
    "elite_barbarians": 6, "royal_giant": 6, "elixir_collector": 6, "xbow": 6, "barb_hut": 6,
    "boss_bandit": 6, "spirit_empress": 6,
    "pekka": 7, "lava_hound": 7, "mega_knight": 7, "royal_recruits": 7, "electro_giant": 7,
    "golem": 8, "three_musketeers": 9,
}  # fmt: skip
_DEFAULT_COST = 4

# Many weak bodies: great on single targets, food for splash.
SWARM_CARDS = frozenset(
    {
        "bats",
        "barbarians",
        "elite_barbarians",
        "goblin_gang",
        "goblins",
        "guards",
        "minion_horde",
        "minions",
        "rascals",
        "royal_recruits",
        "skeleton_army",
        "skeletons",
        "spear_goblins",
    }
)
# Cheap area spells that clear swarms — cast the instant a swarm crosses, not when it's dying.
SMALL_SPELLS = frozenset(
    {"arrows", "barb_barrel", "fireball", "gob_curse", "log", "poison", "snowball", "tornado", "zap"}
)
# Rolling spells travel up the lane from where they land: drop them just on our side of the target.
_ROLLING_SPELLS = frozenset({"log", "barb_barrel"})
# Stun / reset: interrupts charges, Inferno beams, Sparky shots, and units diving the tower.
STUN_CARDS = frozenset({"zap", "electro_spirit", "electro_wizard", "electro_dragon", "freeze", "lightning"})
# Tanks worth starting a push with from behind our towers.
BACK_TANKS = frozenset({"golem", "giant", "lava_hound", "electro_giant", "goblin_giant", "elixir_golem", "rune_giant"})

# ---- what enemy units are (used once unit_classifier recognizes them) ----------
AIR_UNITS = frozenset(
    {
        "baby_dragon", "balloon", "bats", "electro_dragon", "flying_machine", "inferno_dragon", "lava_hound",
        "lava_pups", "mega_minion", "minion_horde", "minions", "phoenix", "skeleton_dragons",
    }
)  # fmt: skip
SPLASH_UNITS = frozenset(
    {
        "baby_dragon", "bomb_tower", "bomber", "bowler", "dark_prince", "electro_dragon", "executioner",
        "firecracker", "giant_skeleton", "goblin_demolisher", "hunter", "ice_wizard", "magic_archer",
        "mega_knight", "monk", "princess", "skeleton_dragons", "sparky", "valkyrie", "witch", "wizard",
    }
)  # fmt: skip
TANK_UNITS = frozenset(
    {
        "electro_giant", "elixir_golem", "giant", "giant_skeleton", "goblin_giant", "goblinstein", "golem",
        "lava_hound", "mega_knight", "pekka", "royal_giant", "rune_giant",
    }
)  # fmt: skip
ENEMY_BUILDINGS = BUILDINGS | frozenset(
    {"bomb_tower", "cannon", "elixir_collector", "goblin_cage", "inferno_tower", "mortar", "tesla", "xbow"}
)
# Our defenders that can't hit air — useless against Balloon / Minions / Lava Hound.
GROUND_ONLY = frozenset(
    {
        "bandit", "barb_hut", "barbarians", "battle_healer", "berserker", "bomb_tower", "bomber", "bowler", "cannon",
        "dark_prince", "elite_barbarians", "fisherman", "giant_skeleton", "goblin_cage", "goblin_demolisher",
        "goblins", "golden_knight", "guards", "ice_golem", "knight", "lumberjack", "mega_knight", "mighty_miner",
        "mini_pekka", "monk", "pekka", "prince", "royal_ghost", "royal_recruits", "skeleton_army",
        "skeleton_king", "skeletons", "tombstone", "valkyrie",
    }
)  # fmt: skip
# High single-target DPS (or a pull building) — what you want on a tank.
TANK_KILLERS = frozenset(
    {
        "cannon", "elite_barbarians", "goblin_cage", "guards", "hunter", "inferno_dragon", "inferno_tower",
        "lumberjack", "mighty_miner", "mini_pekka", "pekka", "prince", "skeleton_army", "tesla",
    }
)  # fmt: skip

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

# Push grouping: enemy units within this distance of the lead count as one push.
_PUSH_RADIUS = 60
# 3+ markers bunched up = a swarm (Minion Horde, Skarmy, Gob Gang, Bats...).
_SWARM_COUNT = 3
# A friendly troop this close to the lead enemy means it's already being dealt with.
_HANDLED_RADIUS = 55
# A lone enemy this deep is hitting (or about to hit) our tower — stun/reset it.
_TOWER_DIVE_Y = 375
STUN_COOLDOWN_S = 6.0
# Defenders engage here or deeper so our tower helps (don't meet the enemy at the bridge).
_TOWER_PULL_Y = 365
# Ranged support sits this far toward the middle so splash on the lane misses it.
_SUPPORT_SIDE_DX = 35
# Elixir kept back for defense after an offensive play (more vs bridge spam).
DEFENSE_RESERVE = 3
PRESSURE_DEFENSE_RESERVE = 4
# Tanks start at the back only with this much elixir.
BACK_TANK_MIN_ELIXIR = 9
BACK_TANK_FOLLOW_UP_S = 14.0
# Bridge plays without a reason (no win condition, no follow-up) only near a leak.
NEAR_LEAK_ELIXIR = 9
# Cheap cards cycled at the back to bring the win condition back around.
_CYCLE_MAX_COST = 3
_RECENT_PENALTY = 1.5
CYCLE_MIN_ELIXIR = 7

# Opponent pattern detection.
_PUSH_GAP_S = 3.0
_PATTERN_WINDOW_S = 60.0
ARCHETYPE_BALANCED = "balanced"
ARCHETYPE_PRESSURE = "bridge spam"
ARCHETYPE_BEATDOWN = "beatdown"

# Named decks from recognized cards: (name, signature cards, how many of them, play style).
# First match wins, so the more specific signatures come first.
_CARD_ARCHETYPES: tuple[tuple[str, frozenset[str], int, str], ...] = (
    ("Lumberloon", frozenset({"balloon", "lumberjack"}), 2, ARCHETYPE_PRESSURE),
    ("Lava Hound", frozenset({"lava_hound"}), 1, ARCHETYPE_BEATDOWN),
    ("Golem beatdown", frozenset({"golem", "elixir_golem"}), 1, ARCHETYPE_BEATDOWN),
    ("Electro Giant", frozenset({"electro_giant"}), 1, ARCHETYPE_BEATDOWN),
    ("Goblin Giant", frozenset({"goblin_giant"}), 1, ARCHETYPE_BEATDOWN),
    ("Giant beatdown", frozenset({"giant", "rune_giant"}), 1, ARCHETYPE_BEATDOWN),
    ("Royal Giant", frozenset({"royal_giant"}), 1, ARCHETYPE_BEATDOWN),
    ("X-Bow", frozenset({"xbow"}), 1, ARCHETYPE_BALANCED),
    ("Mortar", frozenset({"mortar"}), 1, ARCHETYPE_BALANCED),
    ("P.E.K.K.A bridge spam", frozenset({"pekka"}), 1, ARCHETYPE_PRESSURE),
    ("Bridge spam", frozenset({"battle_ram", "bandit", "ram_rider", "royal_ghost", "dark_prince", "prince",
                               "elite_barbarians"}), 2, ARCHETYPE_PRESSURE),
    ("Hog cycle", frozenset({"hog"}), 1, ARCHETYPE_PRESSURE),
    ("Royal Hogs", frozenset({"royal_hogs"}), 1, ARCHETYPE_PRESSURE),
    ("Balloon", frozenset({"balloon"}), 1, ARCHETYPE_PRESSURE),
    ("Miner control", frozenset({"miner"}), 1, ARCHETYPE_PRESSURE),
    ("Log bait", frozenset({"princess", "goblin_gang", "dart_goblin"}), 2, ARCHETYPE_PRESSURE),
    ("Mega Knight", frozenset({"mega_knight"}), 1, ARCHETYPE_BALANCED),
)  # fmt: skip
# The same card seen again within this long is the same deployment, not a new play.
SAME_DEPLOY_S = 8.0
# A played card is back in the opponent's hand after 4 more plays; we miss plays we can't
# see (spells, troops that stay on their half), so also assume it's back after this long.
CYCLE_FORGET_S = 20.0
_CYCLE_DEPTH = 4
# Their win condition was just played and can't be back yet: a smaller reserve is enough.
PUNISH_DEFENSE_RESERVE = 2


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


def unit_traits(kind: str | None) -> UnitTraits | None:
    """What a recognized enemy unit is (``kind`` = card id or coarse category), or None if unknown."""
    if not kind:
        return None
    if kind in CATEGORY_TRAITS:
        return CATEGORY_TRAITS[kind]
    base = _base_card_id(kind)
    return UnitTraits(
        air=base in AIR_UNITS,
        swarm=base in SWARM_CARDS,
        tank=base in TANK_UNITS,
        splash=base in SPLASH_UNITS,
        building=base in ENEMY_BUILDINGS,
        win_condition=base in WIN_CONDITIONS,
    )


def card_cost(card_id: str | None) -> int:
    if not card_id or card_id == "UNKNOWN":
        return _DEFAULT_COST
    return ELIXIR_COST.get(_base_card_id(card_id), _DEFAULT_COST)


def _is(card_id: str, group: frozenset[str]) -> bool:
    return _base_card_id(card_id) in group


def lane_of(x: float) -> str:
    return "left" if x < LANE_SPLIT_X else "right"


def _clamp(value: float, lo: float, hi: float) -> int:
    return int(max(lo, min(hi, value)))


def _avoid_own_towers(x: int, y: int) -> tuple[int, int]:
    """Troops can't be dropped on a tower: princess towers -> step beside it toward the
    middle (keeps the defender at the same depth, inside tower range); king tower -> in front."""
    for left, top, right, bottom in OWN_TOWER_FOOTPRINTS_LTRB:
        if left <= x <= right and top <= y <= bottom:
            if right < LANE_SPLIT_X:
                return right + 8, y
            if left > LANE_SPLIT_X:
                return left - 8, y
            return x, max(OWN_HALF_DROP_Y_RANGE[0], top - 8)
    return x, y


def defensive_drop(role: str, threat: EnemyUnit, card_id: str | None = None) -> tuple[int, int]:
    """Where to put a card of ``role`` to answer ``threat``.

    Troops never land on top of the enemy (they'd eat its splash/burst before doing
    anything): melee goes behind it, deep enough that our tower helps; ranged support
    stays further back and off toward the middle.
    """
    left, top, right, bottom = PLAYABLE_PLAY_REGION_LTRB
    y_min, y_max = OWN_HALF_DROP_Y_RANGE
    if role == ROLE_BUILDING:
        return DEFENSE_BUILDING_PULL_COORDS[threat.lane]
    if role in (ROLE_SPELL, ROLE_OWN_SIDE_SPELL):
        spell_top = y_min if role == ROLE_OWN_SIDE_SPELL else top
        lead = 30 if card_id and _is(card_id, _ROLLING_SPELLS) else _SPELL_LEAD_DY
        return _clamp(threat.x, left, right), _clamp(threat.y + lead, spell_top, bottom)
    x = _clamp(threat.x, left, right)
    if role == ROLE_SUPPORT:
        toward_middle = _SUPPORT_SIDE_DX if threat.lane == "left" else -_SUPPORT_SIDE_DX
        # ...but never across into the other lane.
        lo, hi = (left, LANE_SPLIT_X - 1) if threat.lane == "left" else (LANE_SPLIT_X, right)
        x = _clamp(threat.x + toward_middle, lo, hi)
        y = _clamp(max(threat.y + _SUPPORT_DEFENSE_DY, _TOWER_PULL_Y), y_min, y_max)
    else:
        y = _clamp(max(threat.y + _TROOP_DEFENSE_DY, _TOWER_PULL_Y), y_min, y_max)
    return _avoid_own_towers(x, y)


@dataclass(frozen=True)
class EnemyPush:
    """Enemy units in one lane on our half, summarized."""

    lane: str
    lead: EnemyUnit  # deepest unit
    center: tuple[int, int]
    count: int
    handled: bool  # one of our troops is already fighting it
    kinds: tuple[str, ...] = ()  # recognized unit types in the push (unit_classifier)

    def _traits(self) -> list[UnitTraits]:
        return [t for t in map(unit_traits, self.kinds) if t is not None]

    @property
    def swarm(self) -> bool:
        return self.count >= _SWARM_COUNT or any(t.swarm for t in self._traits())

    @property
    def air(self) -> bool:
        return any(t.air for t in self._traits())

    @property
    def splash(self) -> bool:
        return any(t.splash for t in self._traits())

    @property
    def tank(self) -> bool:
        return any(t.tank for t in self._traits())


def summarize_pushes(threats: Sequence[EnemyUnit], friends: Sequence[EnemyUnit] = ()) -> dict[str, EnemyPush]:
    pushes: dict[str, EnemyPush] = {}
    for lane in ("left", "right"):
        units = sorted((t for t in threats if t.lane == lane), key=lambda t: -t.y)
        if not units:
            continue
        lead = units[0]
        group = [u for u in units if abs(u.x - lead.x) <= _PUSH_RADIUS and abs(u.y - lead.y) <= _PUSH_RADIUS]
        center = (sum(u.x for u in group) // len(group), sum(u.y for u in group) // len(group))
        handled = any(abs(f.x - lead.x) <= _HANDLED_RADIUS and abs(f.y - lead.y) <= _HANDLED_RADIUS for f in friends)
        kinds = tuple(dict.fromkeys(u.kind for u in group if u.kind))
        pushes[lane] = EnemyPush(lane, lead, center, len(group), handled, kinds)
    return pushes


class DeckTracker:
    """Our own 8-card deck and its rotation, learned from the hand during the battle."""

    def __init__(self) -> None:
        self.deck: list[str] = []
        self._order: list[str] = []  # played cards, oldest first (= next to come back)

    def observe_hand(self, hand: Sequence[str]) -> None:
        for card in hand:
            if card and card != "UNKNOWN" and card not in self.deck and len(self.deck) < 8:
                self.deck.append(card)

    def played(self, card_id: str) -> None:
        self.observe_hand([card_id])
        self._order = [c for c in self._order if c != card_id] + [card_id]

    def queue(self, hand: Sequence[str]) -> list[str]:
        """Cards not in hand, in the order they'll come back (first = next card)."""
        return [c for c in self._order if c not in hand]

    def plays_until(self, card_id: str, hand: Sequence[str]) -> int | None:
        """How many plays until ``card_id`` is back in hand (0 = in hand, None = unknown)."""
        if card_id in hand:
            return 0
        queue = self.queue(hand)
        return queue.index(card_id) + 1 if card_id in queue else None

    def win_conditions(self) -> list[str]:
        return [c for c in self.deck if card_role(c) == ROLE_WIN]


class OpponentDeck:
    """The opponent's cards as recognized on our half, and an estimate of their cycle.

    Clash Royale sends a played card to the back of a 4-card queue, so a card is out of
    the opponent's hand until 4 more plays happen. We only see troops that cross onto
    our half (no spells, nothing that stays on their side), so the play log undercounts;
    ``CYCLE_FORGET_S`` caps how long a card is assumed to be out.
    """

    def __init__(self, clock) -> None:
        self._clock = clock
        self.cards: list[str] = []  # first-seen order, at most 8
        self.plays: list[tuple[float, str]] = []  # (time, card), oldest first
        self._last_seen: dict[str, float] = {}

    def observe(self, units: Sequence[EnemyUnit]) -> None:
        now = self._clock()
        for kind in dict.fromkeys(u.kind for u in units if u.kind):
            if kind in CATEGORY_TRAITS:
                continue  # a coarse category isn't a card
            if now - self._last_seen.get(kind, -1e9) > SAME_DEPLOY_S:
                self.plays.append((now, kind))
                if kind not in self.cards and len(self.cards) < 8:
                    self.cards.append(kind)
            self._last_seen[kind] = now

    def ready(self, card: str) -> bool | None:
        """Could ``card`` be in the opponent's hand? None if we've never seen it."""
        if card not in self.cards:
            return None
        now = self._clock()
        recent = self.plays[-_CYCLE_DEPTH:]
        return not any(c == card and now - t < CYCLE_FORGET_S for t, c in recent)

    def likely_hand(self) -> list[str] | None:
        """With all 8 cards known: the ones that could be in hand right now."""
        if len(self.cards) < 8:
            return None
        return [c for c in self.cards if self.ready(c)]

    def known(self, group: frozenset[str]) -> list[str]:
        return [c for c in self.cards if _base_card_id(c) in group]

    def any_ready(self, group: frozenset[str]) -> bool:
        return any(self.ready(c) for c in self.known(group))

    def win_condition_out_of_cycle(self) -> bool:
        """Their known win conditions were all just played — a window to push."""
        wins = self.known(WIN_CONDITIONS)
        return bool(wins) and not any(self.ready(c) for c in wins)

    def named_archetype(self) -> tuple[str, str] | None:
        """(deck name, play style) from the recognized cards, or None."""
        bases = {_base_card_id(c) for c in self.cards}
        for name, signature, needed, style in _CARD_ARCHETYPES:
            if len(bases & signature) >= needed:
                return name, style
        return None


class OpponentModel:
    """What the opponent does: how often and how big they push, where, and with which cards."""

    def __init__(self, clock) -> None:
        self._clock = clock
        self.pushes: list[tuple[float, str, int]] = []  # time, lane, unit count
        self._last_seen: dict[str, float] = {}
        self.deck = OpponentDeck(clock)

    def observe(self, pushes: Mapping[str, EnemyPush], units: Sequence[EnemyUnit] = ()) -> None:
        now = self._clock()
        for lane, push in pushes.items():
            if now - self._last_seen.get(lane, -1e9) > _PUSH_GAP_S:
                self.pushes.append((now, lane, push.count))
            self._last_seen[lane] = now
        self.deck.observe(units)

    def archetype(self) -> str:
        """Deck name when the cards give it away (Lumberloon, Hog cycle...), else the push-pattern style."""
        named = self.deck.named_archetype()
        return named[0] if named else self._pattern()

    def style(self) -> str:
        """bridge spam / beatdown / balanced — what the reserve and back-tank logic react to."""
        named = self.deck.named_archetype()
        return named[1] if named else self._pattern()

    def _pattern(self) -> str:
        now = self._clock()
        recent = [p for p in self.pushes if now - p[0] <= _PATTERN_WINDOW_S]
        if len(recent) >= 4 and sum(p[2] for p in recent) / len(recent) <= 2.0:
            return ARCHETYPE_PRESSURE  # many small pushes: bridge spam, cycle, Lumberloon-style pairs
        if recent and len(recent) <= 2 and max(p[2] for p in recent) >= _SWARM_COUNT:
            return ARCHETYPE_BEATDOWN  # few, big pushes
        return ARCHETYPE_BALANCED

    def favorite_lane(self) -> str | None:
        lanes = [p[1] for p in self.pushes[-6:]]
        if len(lanes) < 3:
            return None
        left = lanes.count("left")
        if left * 3 >= len(lanes) * 2:
            return "left"
        if (len(lanes) - left) * 3 >= len(lanes) * 2:
            return "right"
        return None


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
    """Per-battle state: lanes, recent plays, cooldowns, tower health, deck rotation, opponent pattern."""

    def __init__(self, clock=time.time, rng: random.Random | None = None) -> None:
        self._clock = clock
        self._rng = rng or random.Random()
        self.attack_lane = self._rng.choice(("left", "right"))
        self._counter_push_lane: str | None = None
        self._counter_push_until = 0.0
        self._last_defense: dict[str, tuple[float, int]] = {}  # lane -> (time, threat y)
        # Our last win-condition drop: lane, time, how long to keep backing it up.
        self._last_win: tuple[str, float, float] | None = None
        self._last_stun = -1e9
        self._played_any = False
        self._enemy_hp: dict[str, collections.deque[float]] = {
            lane: collections.deque(maxlen=_TOWER_HISTORY) for lane in ("left", "right")
        }
        self._own_hp: dict[str, collections.deque[float]] = {
            lane: collections.deque(maxlen=_TOWER_HISTORY) for lane in ("left", "right")
        }
        self.recent: list[int] = []
        self.deck = DeckTracker()
        self.opponent = OpponentModel(clock)
        self.situation = ""

    def _pick(self, candidates: list[tuple[int, str, str]]) -> tuple[int, str, str]:
        """Prefer cards not played recently (keeps the cycle moving), then random."""
        fresh = [c for c in candidates if c[0] not in self.recent]
        return self._rng.choice(fresh or candidates)

    def _cheapest(self, candidates: list[tuple[int, str, str]], target: int = 0) -> tuple[int, str, str]:
        """The card whose cost is closest to ``target`` (0 = just the cheapest).

        Recently played slots get a penalty so one card isn't spammed every time.
        """

        def score(card: tuple[int, str, str]) -> float:
            return abs(card_cost(card[1]) - target) + (_RECENT_PENALTY if card[0] in self.recent else 0)

        best = min(score(c) for c in candidates)
        return self._rng.choice([c for c in candidates if score(c) == best])

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

    def reserve(self) -> int:
        """Elixir to keep for defense after an offensive play."""
        if self.opponent.deck.win_condition_out_of_cycle():
            return PUNISH_DEFENSE_RESERVE
        return PRESSURE_DEFENSE_RESERVE if self.opponent.style() == ARCHETYPE_PRESSURE else DEFENSE_RESERVE

    def should_defend_now(self, threats: Sequence[EnemyUnit], friends: Sequence[EnemyUnit] = ()) -> EnemyUnit | None:
        """Return the threat to answer immediately, or None if it's already handled.

        Threats against a low tower come first; otherwise the deepest one. A small push
        our troops are already fighting is left to them (no wasted spell on a dying unit).
        """
        now = self._clock()
        pushes = summarize_pushes(threats, friends)
        ordered = sorted(threats, key=lambda t: (not self._own_tower_weak(t.lane), -t.y))
        for threat in ordered:
            push = pushes.get(threat.lane)
            if push and push.handled and not push.swarm and not self._own_tower_weak(threat.lane):
                continue
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
        elixir: int | None = None,
        friends: Sequence[EnemyUnit] = (),
    ) -> Play | None:
        """Pick a play from the playable ``hand`` (index -> card id), or None to hold.

        ``elixir`` (0-10) turns on elixir discipline; without it the brain only knows
        whether elixir is full.
        """
        pushes = summarize_pushes(threats, friends)
        self.opponent.observe(pushes, threats)
        self.deck.observe_hand(list(hand.values()))
        if elixir is not None and elixir >= 10:
            elixir_full = True
        self.situation = self._describe(elixir, pushes)
        if not hand:
            return None
        cards = [(index, card_id, card_role(card_id)) for index, card_id in hand.items()]

        threat = self.should_defend_now(threats, friends) if threats else None
        if threat is not None:
            push = pushes.get(threat.lane) or EnemyPush(threat.lane, threat, (threat.x, threat.y), 1, False)
            play = self._defend(cards, push, threat)
            if play is not None:
                return play
            # Nothing that defends — pressure the other lane instead.
            lane = _other(threat.lane)
            priorities, reason = _OFFENSE_PRIORITY, f"pressuring the {lane} lane"
        elif self._is_opening(elapsed_s) and any(c[2] in _OPENING_PRIORITY for c in cards):
            lane = self.offense_lane()
            options = [c for c in cards if c[2] in _OPENING_PRIORITY]
            index, card_id, picked = self._cheapest(options) if elixir is not None else self._pick(options)
            return Play(index, card_id, picked, OPENING_BACK_COORDS[lane], lane, False, "opening at the back")
        elif self._following_up():
            assert self._last_win is not None
            lane = self._last_win[0]
            priorities, reason = _FOLLOW_UP_PRIORITY, f"backing up the push in the {lane} lane"
        else:
            lane = self.offense_lane()
            priorities, reason = _OFFENSE_PRIORITY, f"pushing the {lane} lane"
            tank = self._back_tank(cards, lane, elixir, elixir_full)
            if tank is not None:
                return tank
            if elixir is not None and not elixir_full:
                cycle = self._cycle(cards, lane, elixir)
                if cycle is not None:
                    return cycle

        enemy_hp = self.enemy_tower_hp(lane)
        finishing = enemy_hp is not None and 0 < enemy_hp <= FINISH_TOWER_HP
        spell_worth_it = finishing or elapsed_s >= SPELL_CHIP_AFTER_S
        if spell_worth_it:
            priorities = (*priorities, ROLE_SPELL)
        if elixir_full:
            priorities = (*priorities, *_LEAK_PRIORITY)

        for role in dict.fromkeys(priorities):  # de-dupe, keep order
            options = [c for c in cards if c[2] == role and self._affordable_on_offense(c, elixir, elixir_full)]
            if role == ROLE_WIN and elixir is not None and not elixir_full:
                # Back tanks wait for the full-elixir push (see _back_tank), never a naked bridge drop.
                options = [c for c in options if not _is(c[1], BACK_TANKS)]
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

    def _defend(self, cards: list[tuple[int, str, str]], push: EnemyPush, threat: EnemyUnit) -> Play | None:
        lane = threat.lane

        def play(card: tuple[int, str, str], coord: tuple[int, int], why: str) -> Play:
            return Play(card[0], card[1], card[2], coord, lane, True, why)

        # 1. A lone heavy hitter at our tower: stun it (resets Inferno/Sparky/charges, knocks back divers).
        if push.count <= 2 and threat.y >= _TOWER_DIVE_Y and self._clock() - self._last_stun >= STUN_COOLDOWN_S:
            stuns = [c for c in cards if _is(c[1], STUN_CARDS)]
            if stuns:
                card = self._cheapest(stuns)
                role = ROLE_SPELL if card[2] == ROLE_SPELL else ROLE_TROOP
                return play(card, defensive_drop(role, threat, card[1]), f"stunning the {lane} lane push to reset it")

        # 2. Swarm: small spell on the group the moment it crosses.
        if push.swarm:
            spells = [c for c in cards if c[2] in (ROLE_SPELL, ROLE_OWN_SIDE_SPELL) and _is(c[1], SMALL_SPELLS)]
            if spells:
                card = self._cheapest(spells)
                center = EnemyUnit(push.center[0], push.center[1], threat.size)
                return play(card, defensive_drop(card[2], center, card[1]), f"spelling the swarm in the {lane} lane")

        # 3. Cheapest troop/building that handles it. Swarm cards stay out of multi-unit pushes
        #    and away from splash (Bowler/Valkyrie/Wizard wipes them) — also when a splash card
        #    of theirs we've seen could be back in hand.
        target = 3 if push.count == 1 else 4 if push.count == 2 else 5
        defenders = [c for c in cards if c[2] in (ROLE_TROOP, ROLE_SUPPORT, ROLE_BUILDING)]
        if push.air:
            # Ground-only defenders can't touch Balloon/Minions; with no anti-air, fall through to spells.
            defenders = [c for c in defenders if not _is(c[1], GROUND_ONLY)]
        if push.count >= 2 or push.splash or self.opponent.deck.any_ready(SPLASH_UNITS):
            defenders = [c for c in defenders if not _is(c[1], SWARM_CARDS)] or defenders
        if push.tank:
            killers = [c for c in defenders if _is(c[1], TANK_KILLERS) or c[2] == ROLE_BUILDING]
            if killers:
                defenders, target = killers, 5
        if defenders:
            card = self._cheapest(defenders, target)
            return play(card, defensive_drop(card[2], threat, card[1]), f"defending the {lane} lane")

        # 4. Any spell / own-side spell left.
        for role in (ROLE_SPELL, ROLE_OWN_SIDE_SPELL):
            options = [c for c in cards if c[2] == role]
            if options:
                card = self._cheapest(options)
                return play(card, defensive_drop(role, threat, card[1]), f"defending the {lane} lane")
        return None

    def _affordable_on_offense(self, card: tuple[int, str, str], elixir: int | None, elixir_full: bool) -> bool:
        """Elixir discipline: offense keeps a defense reserve unless backed by a reason."""
        if elixir is None or elixir_full:
            return True
        if card[2] in (ROLE_SPELL, ROLE_BUILDING, ROLE_UTILITY, ROLE_OWN_SIDE_SPELL):
            return True  # gated by the priority lists already
        counter_pushing = self._counter_push_lane is not None and self._clock() < self._counter_push_until
        if card[2] == ROLE_WIN:
            return counter_pushing or elixir - card_cost(card[1]) >= self.reserve()
        # Support/troops: only to back up a push, counter-push, or avoid a leak.
        if self._following_up() or counter_pushing:
            return elixir - card_cost(card[1]) >= 1  # committing to a live push is worth it
        return elixir >= NEAR_LEAK_ELIXIR

    def _back_tank(
        self, cards: list[tuple[int, str, str]], lane: str, elixir: int | None, elixir_full: bool
    ) -> Play | None:
        tanks = [c for c in cards if _is(c[1], BACK_TANKS)]
        if not tanks or elixir is None:
            return None
        if self.opponent.style() == ARCHETYPE_PRESSURE and not elixir_full:
            return None  # a slow push at the back gets punished by bridge spam in the other lane
        if not elixir_full and elixir < max(BACK_TANK_MIN_ELIXIR, *(card_cost(c[1]) for c in tanks)):
            return None
        index, card_id, picked = self._pick(tanks)
        return Play(
            index,
            card_id,
            picked,
            BACK_TANK_COORDS[lane],
            lane,
            False,
            f"starting a push at the back in the {lane} lane",
        )

    def _cycle(self, cards: list[tuple[int, str, str]], lane: str, elixir: int) -> Play | None:
        """No win condition in hand: spend a cheap card at the back to bring one around."""
        if elixir < CYCLE_MIN_ELIXIR or any(c[2] == ROLE_WIN for c in cards):
            return None
        hand = [c[1] for c in cards]
        waits = [(n, w) for w in self.deck.win_conditions() if (n := self.deck.plays_until(w, hand))]
        if not waits:
            return None
        cheap = [c for c in cards if c[2] in (ROLE_TROOP, ROLE_SUPPORT) and card_cost(c[1]) <= _CYCLE_MAX_COST]
        if not cheap:
            return None
        plays_away, win = min(waits)
        index, card_id, picked = self._cheapest(cheap)
        return Play(
            index, card_id, picked, OPENING_BACK_COORDS[lane], lane, False,
            f"cycling toward {win} ({plays_away} play{'s' if plays_away > 1 else ''} away)",
        )  # fmt: skip

    def _following_up(self) -> bool:
        return self._last_win is not None and self._clock() - self._last_win[1] < self._last_win[2]

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

    def _describe(self, elixir: int | None, pushes: Mapping[str, EnemyPush]) -> str:
        """One-line read of the game right now, for the status bar."""
        parts = [f"{elixir} elixir"] if elixir is not None else []
        for lane, push in pushes.items():
            kind = "swarm" if push.swarm else f"{push.count} unit{'s' if push.count > 1 else ''}"
            parts.append(f"enemy {kind} {lane}{' (ours on it)' if push.handled else ''}")
        if not pushes:
            parts.append("no enemies on our side")
        for lane, push in pushes.items():
            if push.kinds:
                parts.append(f"{lane}: {', '.join(push.kinds)}")
        parts.append(f"opponent: {self.opponent.archetype()}")
        if self.opponent.deck.cards:
            parts.append(f"seen {len(self.opponent.deck.cards)}/8 of their cards")
        return " · ".join(parts)

    def record(self, play: Play) -> None:
        """Remember a play we made (call after the clicks went through)."""
        self._played_any = True
        self.recent = [*self.recent[-2:], play.index]
        self.deck.played(play.card_id)
        now = self._clock()
        if play.reason.startswith("stunning"):
            self._last_stun = now
        if play.defensive:
            self._last_defense[play.lane] = (now, play.coord[1])
            self._counter_push_lane = play.lane
            self._counter_push_until = now + COUNTER_PUSH_WINDOW_S
        elif play.role == ROLE_WIN:
            window = BACK_TANK_FOLLOW_UP_S if play.coord == BACK_TANK_COORDS[play.lane] else SUPPORT_FOLLOW_UP_S
            self._last_win = (play.lane, now, window)
