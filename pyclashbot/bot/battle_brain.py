"""Smart battle AI: decide which card to play and where from what's on the board.

The default fight loop waits for a random elixir amount and plays a random card.
This module plays more like a person:

* **Defend pushes cheaply, and only as hard as needed.** Enemy troops crossing onto our
  half (``find_enemy_units``) are grouped into a push per lane. A real swarm (4+ markers,
  or a recognized Minion Horde / Skarmy...) gets an area spell, a lone troop gets the
  cheapest known card that handles it, and a push our troops already fight is left
  alone. ``defense_budget`` sizes the answer: short on elixir far from the tower we
  wait instead of throwing in a card that can't win the fight.
* **Spells are precious.** Arrows/Snowball are kept for swarms; tower damage only when
  the spell actually finishes the tower (``SPELL_TOWER_FRACTION``), plus a tiebreak chip
  in the last seconds of overtime. The last swarm answer in the cycle is never thrown.
* **Stuns on purpose.** Zap / Electro Spirit / E-Wiz / E-Dragon reset a lone heavy hitter
  that reaches our tower.
* **Per-card placement** (``card_knowledge.CardInfo.def_place``): cycle cards land on the
  threat, tanks in front of it, buildings in the centre pull spot.
* **Elixir discipline by match phase.** Leak / back-tank / cycle thresholds drop in
  double elixir and overtime (``MatchClock`` / ``match_phase``).
* **Real pushes.** Tanks (Golem, Giant, Giant Skeleton when it is the deck's only win
  condition...) start at the back; follow-ups wait until the tank reaches the bridge and
  then drop behind it. Counter-pushes only happen while a troop of ours survived the
  defense, with a spend cap.
* **Memory.** Tracks our own deck and rotation (resolving an unreadable slot from the
  rotation), infers the win condition, remembers threats for a couple of seconds when a
  frame misses them, and models the opponent's push pattern and cards.
* **Hero card hint.** ``BattleBrain(hero_card="hero_balloon")`` (the user's deck setting) names
  the card the reader has no fingerprint for; an UNKNOWN slot the rotation can't name becomes
  it when the slot's cost bounds allow. Tank-led win conditions (``TANK_LED_WINS``: Balloon)
  never defend: they go in behind a back tank as it nears the bridge, or right behind a
  Lumberjack at the bridge (Lumberloon), and alone only near a leak or at a low tower.

Pure decision logic over (hand, units, elixir, time) so it is unit-testable offline;
``fight.py`` owns the screenshots and clicks.
"""

from __future__ import annotations

import collections
import logging
import random
import statistics
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from pyclashbot.bot import card_knowledge as ck
from pyclashbot.bot.card_detection import calculate_play_coords, get_card_group
from pyclashbot.bot.coords import (
    BACK_TANK_COORDS,
    DEFENSE_BUILDING_PULL_COORDS,
    ENEMY_PRINCESS_TOWER_CENTERS,
    LANE_BRIDGE_X,
    LANE_SPLIT_X,
    OPENING_BACK_COORDS,
    OWN_HALF_DROP_Y_RANGE,
    OWN_TOWER_FOOTPRINTS_LTRB,
    PLAYABLE_PLAY_REGION_LTRB,
)
from pyclashbot.bot.find import EnemyUnit
from pyclashbot.bot.unit_classifier import CATEGORY_TRAITS, UnitTraits

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from pyclashbot.bot.find import TowerHealth

_log = logging.getLogger(__name__)

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
# Damage/control spells: cast on an enemy push, or at a tower to finish it.
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
STUN_CARDS = frozenset({"zap", "electro_spirit", "electro_wizard", "electro_dragon"})
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

# Melee swarm cards: they fight like troops (on the threat), not like ranged support at the back.
MELEE_SWARMS = frozenset({"skeletons", "skeleton_army", "goblins"})
# Buildings that make value on their own: worth placing on an empty board.
SPAWNER_BUILDINGS = frozenset({"furnace", "goblin_hut"})
# Spells that only hit ground units, and spells that don't kill anything by themselves.
GROUND_ONLY_SPELLS = frozenset({"log", "barb_barrel", "earthquake"})
NO_KILL_SPELLS = frozenset({"tornado"})
# Area spells big enough for a Minion Horde / Skarmy / Goblin Gang.
BIG_SWARM_SPELLS = frozenset({"arrows", "fireball", "poison"})
# Recognized swarms a spell actually kills...
SPELLABLE_SWARMS = frozenset(
    {"bats", "goblin_gang", "goblins", "minion_horde", "minions", "skeleton_army", "skeletons", "spear_goblins"}
)
# ...of which a single marker already means a big group.
_ONE_MARKER_SWARMS = frozenset({"goblin_gang", "minion_horde", "skeleton_army"})
# ...and the cheap ones (1-2 elixir) only worth a spell when there are plenty of them.
_CHEAP_SWARMS = frozenset({"bats", "goblins", "skeletons", "spear_goblins"})
# Fraction of a princess tower's HP a spell removes (rough, level-agnostic): finish only below this.
SPELL_TOWER_FRACTION: dict[str, float] = {
    "arrows": 0.04, "snowball": 0.04, "zap": 0.04, "log": 0.04, "gob_curse": 0.04, "barb_barrel": 0.03,
    "fireball": 0.10, "poison": 0.10, "earthquake": 0.08, "void": 0.08, "lightning": 0.12, "rocket": 0.30,
}  # fmt: skip

_DEFENSE_PRIORITY = (ROLE_TROOP, ROLE_SUPPORT, ROLE_BUILDING, ROLE_SPELL, ROLE_OWN_SIDE_SPELL)
# Opening play: something cheap at the back, never a naked win condition at the bridge.
_OPENING_PRIORITY = (ROLE_SUPPORT, ROLE_TROOP)
_LEAK_PRIORITY = (ROLE_SPELL, ROLE_BUILDING, ROLE_UTILITY, ROLE_OWN_SIDE_SPELL)

# Melee/tank defenders go just behind the enemy so it walks into them; ranged support further back.
_TROOP_DEFENSE_DY = 45
_SUPPORT_DEFENSE_DY = 85
# card_knowledge placements: cycle cards right on the threat, tanks just in front of it.
_ON_THREAT_DY = 10
_FRONT_OF_THREAT_DY = 25
# Spells land a little ahead of the enemy (it keeps walking toward our towers).
_SPELL_LEAD_DY = 12
# After defending a lane, ignore further pings there for this long unless the push grows,
# gets deeper or our tower starts taking damage.
DEFENSE_COOLDOWN_S = 5.0
WEAK_TOWER_DEFENSE_COOLDOWN_S = 2.0
DEFENSE_LOCK_S = 5.0
_DEEPER_PUSH_DY = 40
_LOCK_HP_DROP = 0.08
_LOCK_GROWTH_READS = 2
# Growth reads closer together than this are the same frame (observe + choose both ask).
_LOCK_GROWTH_MIN_GAP_S = 0.15
# How long offense keeps favoring the lane we just defended, and how much it may spend there.
COUNTER_PUSH_WINDOW_S = 12.0
COUNTER_PUSH_SPEND_CAP = 4
# A surviving defender this far up (y below it) makes a counter-push worth it.
_COUNTER_PUSH_FRIEND_Y = 330
# How long after a win condition we prefer backing it up with support.
SUPPORT_FOLLOW_UP_S = 8.0
# The opening play happens within this window (later first plays are just normal offense).
OPENING_WINDOW_S = 25.0

# Match phases (seconds on the match clock).
PHASE_SINGLE = "single"
PHASE_DOUBLE = "double"
PHASE_OT2 = "ot2"
PHASE_OT3 = "ot3"
_OVERTIME = (PHASE_OT2, PHASE_OT3)
# Last 15 s of overtime: chip whatever tower is lowest to win the tiebreak.
OT_TIEBREAK_CHIP_S = 285.0

# Tower health (0..1 fractions of the HP bar, median of the reads from the last few seconds).
WEAK_TOWER_HP = 0.4
FINISH_TOWER_HP = 0.25
LANE_SWITCH_HP_MARGIN = 0.15
TOWER_WINDOW_S = 3.0
_TOWER_HISTORY = 50
# A weak tower whose bar stays unreadable this long is treated as destroyed (hp 0).
TOWER_DESTROYED_MISS_S = 4.0

# Push grouping: enemy units within this distance of the lead count as one push.
_PUSH_RADIUS = 60
# With unrecognized units, this many bunched markers is a swarm worth a spell (3 is often
# a tank + support, e.g. Golem + Night Witch + Baby Dragon).
_SWARM_MARKERS = 4
# 3+ units in one push = a big push (opponent pattern detection).
_SWARM_COUNT = 3
# A friendly troop this close to the lead enemy means it's already being dealt with.
_HANDLED_RADIUS = 55
# A lone enemy this deep is hitting (or about to hit) our tower — stun/reset it.
_TOWER_DIVE_Y = 375
STUN_COOLDOWN_S = 6.0
# Short on elixir: still answer (with a 1-2 cost card) once the push is this close to the tower.
_AT_TOWER_Y = 335
_SHORT_DEFENSE_MAX_COST = 2
# Defenders engage here or deeper so our tower helps (don't meet the enemy at the bridge).
_TOWER_PULL_Y = 365
# Ranged support sits this far toward the middle so splash on the lane misses it.
_SUPPORT_SIDE_DX = 35
# Unknown-kind lone pushes might be air (Balloon, Minions): ground-only defenders score worse.
_AIR_HEDGE_PENALTY = 0.6
_UNKNOWN_PENALTY = 0.5
# Elixir kept back for defense after an offensive play (more vs bridge spam).
DEFENSE_RESERVE = 3
PRESSURE_DEFENSE_RESERVE = 4
# Phase thresholds: bridge plays without a reason, tanks at the back, cycling.
NEAR_LEAK_BY_PHASE = {PHASE_SINGLE: 9, PHASE_DOUBLE: 8, PHASE_OT2: 8, PHASE_OT3: 7}
BACK_TANK_MIN_BY_PHASE = {PHASE_SINGLE: 9, PHASE_DOUBLE: 7, PHASE_OT2: 7, PHASE_OT3: 7}
CYCLE_MIN_BY_PHASE = {PHASE_SINGLE: 7, PHASE_DOUBLE: 6, PHASE_OT2: 6, PHASE_OT3: 5}
# A tank-led win condition (Balloon) in hand behind a back tank (Giant Skeleton): start the
# tank at 8 rather than waiting for 9, so the win condition is affordable while it walks.
TANK_LED_BACK_TANK_MIN = 8
# Once one of our princess towers is below this, every new push keeps LOW_TOWER_RESERVE back.
OWN_TOWER_LOW_HP = 0.5
LOW_TOWER_RESERVE = 4
NEAR_LEAK_ELIXIR = NEAR_LEAK_BY_PHASE[PHASE_SINGLE]
BACK_TANK_MIN_ELIXIR = BACK_TANK_MIN_BY_PHASE[PHASE_SINGLE]
CYCLE_MIN_ELIXIR = CYCLE_MIN_BY_PHASE[PHASE_SINGLE]
BACK_TANK_FOLLOW_UP_S = 14.0
# After a tank goes down at the back, troop follow-ups wait for it to reach the bridge.
FOLLOW_UP_HOLD_S = 7.5
TANK_SPEED_PX_S = 22.0
_FOLLOW_BEHIND_DY = 35
_TANK_AT_BRIDGE_Y = 340
# Threat memory: a push seen this recently still counts when a frame misses it.
THREAT_MEMORY_S = 2.0
SPELL_BLOCK_MEMORY_S = 3.0
# Cheap cards cycled at the back to bring the win condition back around.
_CYCLE_MAX_COST = 3
_RECENT_PENALTY = 1.5
# Our own deck: the slot the game couldn't read, and its label once 7 cards are known.
UNKNOWN_CARD = "UNKNOWN"
DECK_X = "deck_x"
_INFER_WIN_KNOWN = 6
_INFER_WIN_PLAYS = 8
# A played card is back in hand after this many more deployed plays (4-card queue).
_ROTATION_DEPTH = 4

# Tank-led win conditions: they only hit buildings, so they never defend and go in behind a
# tank (Giant Skeleton at the back) or a Lumberjack at the bridge, not alone.
TANK_LED_WINS = frozenset({"balloon"})
# Bridge leaders for the Lumberloon-style pair (leader at the bridge, win condition right behind).
PAIR_LEADERS = frozenset({"lumberjack"})
PAIR_MIN_BY_PHASE = {PHASE_SINGLE: 8, PHASE_DOUBLE: 7, PHASE_OT2: 7, PHASE_OT3: 7}
# How long the win condition may follow its bridge leader (elixir catch-up included).
PAIR_FOLLOW_S = 4.0
# ~1 tile behind the tank / leader (an arena tile is ~16 px at this resolution).
_TANK_LED_BEHIND_DY = 18
# After a tank-led push goes in, spells stay in hand for the swarms that defend it.
TANK_LED_PUSH_S = 12.0
# Tank-led offense prefers the weaker enemy tower once the reads differ by this much.
_WIN_LANE_HP_MARGIN = 0.05
_PAIR_REASON = "lumberloon"
_TANK_LED_REASON = "tank-led"

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
    """Base card id: evo_/hero_ prefixes and detection aliases removed (see ``card_knowledge``)."""
    return ck.base_card_id(card_id)


def card_role(card_id: str | None) -> str:
    """Classify a detected card id into the role the brain plays it as."""
    if not card_id or card_id in (UNKNOWN_CARD, DECK_X):
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
    if base in MELEE_SWARMS:
        return ROLE_TROOP
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
    if not card_id or card_id in (UNKNOWN_CARD, DECK_X):
        return _DEFAULT_COST
    base = _base_card_id(card_id)
    if base in ELIXIR_COST:
        return ELIXIR_COST[base]
    info = ck.card_info(base)
    return info.cost if info is not None else _DEFAULT_COST


def _is(card_id: str, group: frozenset[str]) -> bool:
    return _base_card_id(card_id) in group


def _known(card_id: str | None) -> bool:
    return bool(card_id) and card_id not in (UNKNOWN_CARD, DECK_X)


def normalize_hero_card(card_id: object) -> str | None:
    """The deck hero-card setting as a card id, or None for blank / "None" / "auto"."""
    if not card_id:
        return None
    text = str(card_id).strip()
    if not text or text.lower() in ("none", "auto", "none/auto") or text in (UNKNOWN_CARD, DECK_X):
        return None
    return text


def is_tank_led_win(card_id: str | None) -> bool:
    """Balloon-style win condition: building-only, played behind a tank, never on defense."""
    return _known(card_id) and _base_card_id(str(card_id)) in TANK_LED_WINS


def lane_of(x: float) -> str:
    return "left" if x < LANE_SPLIT_X else "right"


def _clamp(value: float, lo: float, hi: float) -> int:
    return int(max(lo, min(hi, value)))


def match_phase(elapsed_s: float) -> str:
    """single (0-2:00) / double (2:00-3:00) / ot2 (first minute of overtime) / ot3 (after)."""
    if elapsed_s < 120:
        return PHASE_SINGLE
    if elapsed_s < 180:
        return PHASE_DOUBLE
    if elapsed_s < 240:
        return PHASE_OT2
    return PHASE_OT3


class MatchClock:
    """Seconds since the match started, on an injectable clock."""

    def __init__(self, start: float | None = None, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self.start = clock() if start is None else start

    def elapsed(self) -> float:
        return self._clock() - self.start

    def phase(self) -> str:
        return match_phase(self.elapsed())


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

    Troops never land on top of a big enemy (they'd eat its splash/burst before doing
    anything): melee goes behind it, deep enough that our tower helps; ranged support
    stays further back and off toward the middle. With a ``card_id`` that
    ``card_knowledge`` knows, its ``def_place`` decides instead (cycle cards ON the
    threat, tanks in FRONT of it, buildings at the PULL spot).
    """
    left, top, right, bottom = PLAYABLE_PLAY_REGION_LTRB
    y_min, y_max = OWN_HALF_DROP_Y_RANGE
    if role == ROLE_BUILDING:
        return DEFENSE_BUILDING_PULL_COORDS[threat.lane]
    if role in (ROLE_SPELL, ROLE_OWN_SIDE_SPELL):
        spell_top = y_min if role == ROLE_OWN_SIDE_SPELL else top
        lead = 30 if card_id and _is(card_id, _ROLLING_SPELLS) else _SPELL_LEAD_DY
        return _clamp(threat.x, left, right), _clamp(threat.y + lead, spell_top, bottom)
    info = ck.card_info(card_id)
    if info is not None:
        if info.def_place == ck.PULL:
            return DEFENSE_BUILDING_PULL_COORDS[threat.lane]
        if info.def_place in (ck.ON, ck.FRONT):
            dy = _ON_THREAT_DY if info.def_place == ck.ON else _FRONT_OF_THREAT_DY
            y = _clamp(max(threat.y + dy, _TOWER_PULL_Y), y_min, y_max)
            return _avoid_own_towers(_clamp(threat.x, left, right), y)
        if info.def_place == ck.BACK:
            role = ROLE_SUPPORT
        elif info.def_place == ck.BEHIND:
            role = ROLE_TROOP
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
    first_count: int = 0  # unit count when this push was first seen (tracked by the brain)

    def _traits(self) -> list[UnitTraits]:
        return [t for t in map(unit_traits, self.kinds) if t is not None]

    def _bases(self) -> set[str]:
        return {_base_card_id(k) for k in self.kinds}

    @property
    def swarm(self) -> bool:
        """Many weak bodies: unknown kinds need 4+ markers; known kinds need a swarm card
        with 2+ markers (or 3+ when first seen, or one that is a big group by itself)."""
        traits = self._traits()
        if not traits:
            return self.count >= _SWARM_MARKERS
        if not any(t.swarm for t in traits):
            return False
        return self.count >= 2 or self.first_count >= _SWARM_COUNT or bool(self._bases() & _ONE_MARKER_SWARMS)

    @property
    def spellable(self) -> bool:
        """A swarm an area spell actually kills (not Barbarians / Guards / Recruits)."""
        if not self.kinds:
            return self.count >= _SWARM_MARKERS
        bases = self._bases() & SPELLABLE_SWARMS
        if not bases:
            return self.count >= _SWARM_MARKERS and self.swarm
        if bases & _ONE_MARKER_SWARMS:
            return True
        need = _SWARM_MARKERS if bases <= _CHEAP_SWARMS else 2
        return max(self.count, self.first_count) >= need

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
    """Our own 8-card deck and its rotation, learned from the hand during the battle.

    Card ids are kept as base ids (``evo_furnace`` and ``furnace`` are one card). Plays are
    logged with their hand slot: the card that refills slot ``s`` after a play is the one
    played 4 deployed plays earlier, which names a slot the card reader can't identify.

    ``hero_card`` (the user's deck setting, e.g. ``hero_balloon``) names a card the reader
    has no fingerprint for: it counts as one of the 8 deck cards from the start, and an
    UNKNOWN slot the rotation can't name becomes it when the slot's cost bounds allow it.
    """

    def __init__(self, hero_card: str | None = None) -> None:
        self.deck: list[str] = []
        self._order: list[str] = []  # played cards, oldest first (= next to come back)
        self.plays: list[tuple[int | None, str, bool]] = []  # (slot, card, deployed)
        # Elixir cost bounds (lo, hi) of the UNKNOWN card in each slot, from affordability.
        self.cost_bounds: dict[int, tuple[int, int]] = {}
        self._warned_unknown = False
        self.hero_card = normalize_hero_card(hero_card)
        self.hero_base = _base_card_id(self.hero_card) if self.hero_card else None
        # UNKNOWN slots that might be the hero card but couldn't be pinned down (2+ candidates).
        self.maybe_hero_slots: set[int] = set()
        if self.hero_base:
            self.deck.append(self.hero_base)

    def _label(self, card_id: str | None) -> str | None:
        """Base id for a known card, ``deck_x`` for the unreadable 8th card, else None."""
        if not card_id:
            return None
        if card_id == UNKNOWN_CARD:
            return DECK_X if len(self.deck) >= 7 else None
        return _base_card_id(card_id)

    def _hand_labels(self, hand: Sequence[str]) -> set[str]:
        return {label for label in map(self._label, hand) if label}

    def observe_hand(self, hand: Sequence[str]) -> None:
        for card in hand:
            if not _known(card):
                continue
            base = _base_card_id(card)
            if base not in self.deck and len(self.deck) < 8:
                self.deck.append(base)
        if UNKNOWN_CARD in hand and not self._warned_unknown:
            self._warned_unknown = True
            _log.warning("A hand slot reads as UNKNOWN — playing it last; check the card fingerprints")

    def observe_costs(self, full_hand: Mapping[int, str], affordable: set[int], elixir: int) -> None:
        """Narrow the UNKNOWN slots' cost: affordable at ``elixir`` means cost <= elixir, else > elixir."""
        for slot, card in full_hand.items():
            if card != UNKNOWN_CARD:
                continue
            lo, hi = self.cost_bounds.get(slot, (1, 10))
            if slot in affordable:
                hi = min(hi, elixir)
            else:
                lo = max(lo, elixir + 1)
            self.cost_bounds[slot] = (lo, max(lo, hi))

    def resolve_unknown(self, slot: int) -> str | None:
        """The card now in ``slot``: the one played 4 deployed plays before that slot's last play."""
        deployed = [(s, c) for s, c, d in self.plays if d]
        for i in range(len(deployed) - 1, -1, -1):
            if deployed[i][0] == slot:
                j = i - 4
                card = deployed[j][1] if j >= 0 else None
                return _base_card_id(card) if card is not None and _known(card) else None
        return None

    def _plays_since(self, base: str) -> int | None:
        """Deployed plays after the last deployed ``base`` play (None if it was never played)."""
        deployed = [c for _, c, d in self.plays if d]
        for i in range(len(deployed) - 1, -1, -1):
            if _known(deployed[i]) and _base_card_id(deployed[i]) == base:
                return len(deployed) - 1 - i
        return None

    def could_be_hero(self, slot: int, hand: Mapping[int, str]) -> bool:
        """The UNKNOWN in ``slot`` may be the hero card: its cost fits the slot's affordability
        bounds, no other slot shows it, and it isn't still in the queue from a recent play."""
        if self.hero_card is None or self.hero_base is None:
            return False
        lo, hi = self.cost_bounds.get(slot, (1, 10))
        if not lo <= card_cost(self.hero_card) <= hi:
            return False
        if any(i != slot and _known(c) and _base_card_id(c) == self.hero_base for i, c in hand.items()):
            return False
        since = self._plays_since(self.hero_base)
        return since is None or since >= _ROTATION_DEPTH

    def resolve_hand(self, hand: Mapping[int, str]) -> dict[int, str]:
        """Name the UNKNOWN slots: the rotation first, then the hero-card hint where it fits.

        With two or more UNKNOWN slots that could be the hero, the one with the highest cost
        floor gets it when that floor is unique; otherwise they all stay UNKNOWN and are listed in
        ``maybe_hero_slots`` (the brain plays none of them).
        """
        named = {i: (self.resolve_unknown(i) or c) if c == UNKNOWN_CARD else c for i, c in hand.items()}
        self.maybe_hero_slots = set()
        if self.hero_card is None:
            return named
        open_slots = [i for i, c in named.items() if c == UNKNOWN_CARD and self.could_be_hero(i, named)]
        if not open_slots:
            return named
        floors = {i: self.cost_bounds.get(i, (1, 10))[0] for i in open_slots}
        best = max(floors.values())
        top = [i for i in open_slots if floors[i] == best]
        if len(top) == 1:
            named[top[0]] = self.hero_card  # only one copy: the other UNKNOWNs aren't it
        else:
            self.maybe_hero_slots = set(open_slots)
        return named

    def played(self, card_id: str, slot: int | None = None, deployed: bool = True) -> None:
        """Log a play from ``slot``; ``deployed=False`` = tapped but the card never went down."""
        if card_id == UNKNOWN_CARD and slot is not None:
            card_id = self.resolve_unknown(slot) or card_id
        self.plays.append((slot, card_id, deployed))
        if not deployed:
            return
        if slot is not None:
            self.cost_bounds.pop(slot, None)  # a new card fills the slot
        self.observe_hand([card_id])
        label = self._label(card_id)
        if label:
            self._order = [c for c in self._order if c != label] + [label]

    def queue(self, hand: Sequence[str]) -> list[str]:
        """Cards not in hand, in the order they'll come back (first = next card)."""
        in_hand = self._hand_labels(hand)
        return [c for c in self._order if c not in in_hand]

    def plays_until(self, card_id: str, hand: Sequence[str]) -> int | None:
        """How many plays until ``card_id`` is back in hand (0 = in hand, None = unknown)."""
        label = self._label(card_id)
        if not label:
            return None
        if label in self._hand_labels(hand):
            return 0
        queue = self.queue(hand)
        return queue.index(label) + 1 if label in queue else None

    def inferred_win(self) -> str | None:
        """No real win condition in the deck: the most expensive tank / bridge card carries it
        (Giant Skeleton in a deck with no Hog/Giant/Balloon...). Needs enough of the deck seen."""
        if any(card_role(c) == ROLE_WIN for c in self.deck):
            return None
        deployed = sum(1 for *_, d in self.plays if d)
        if len(self.deck) < _INFER_WIN_KNOWN and deployed < _INFER_WIN_PLAYS:
            return None
        candidates = [
            c
            for c in self.deck
            if card_role(c) not in (ROLE_SPELL, ROLE_OWN_SIDE_SPELL)
            and (ck.has_role(c, ck.TANK, ck.WIN2) or get_card_group(c) == "bridge_line")
        ]
        return max(candidates, key=card_cost) if candidates else None

    def win_conditions(self) -> list[str]:
        wins = [c for c in self.deck if card_role(c) == ROLE_WIN]
        inferred = self.inferred_win()
        return [*wins, inferred] if inferred and inferred not in wins else wins


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
    threat_y: int | None = None  # y of the threat a defensive play answered


@dataclass
class _DefenseCommit:
    """A lane we just defended: locked until the push grows, gets deeper or hurts the tower."""

    at: float
    threat_y: int
    count: int
    own_hp: float | None
    growth: set[float]


Card = tuple[int, str, str]  # (hand slot, card id, role)


def _other(lane: str) -> str:
    return "left" if lane == "right" else "right"


class BattleBrain:
    """Per-battle state: lanes, recent plays, cooldowns, tower health, deck rotation, opponent pattern."""

    def __init__(self, clock=time.time, rng: random.Random | None = None, hero_card: str | None = None) -> None:
        self._clock = clock
        # The user's deck setting for the card the reader can't see (e.g. hero_balloon), or None.
        self.hero_card = normalize_hero_card(hero_card)
        self._rng = rng or random.Random()
        self.attack_lane = self._rng.choice(("left", "right"))
        self._counter_push_lane: str | None = None
        self._counter_push_until = 0.0
        self._counter_push_spent = 0
        self._last_defense: dict[str, tuple[float, int]] = {}  # lane -> (time, threat y)
        self._commits: dict[str, _DefenseCommit] = {}
        # Our last win-condition drop: lane, time, how long to keep backing it up.
        self._last_win: tuple[str, float, float] | None = None
        self._back_tank_drop: tuple[str, float] | None = None
        self._last_stun = -1e9
        self._played_any = False
        self._enemy_hp: dict[str, collections.deque[tuple[float, float]]] = {
            lane: collections.deque(maxlen=_TOWER_HISTORY) for lane in ("left", "right")
        }
        self._own_hp: dict[str, collections.deque[tuple[float, float]]] = {
            lane: collections.deque(maxlen=_TOWER_HISTORY) for lane in ("left", "right")
        }
        # (side, lane) -> time the bar first went unreadable, while it was weak.
        self._tower_miss_since: dict[tuple[str, str], float] = {}
        self._threat_memory: dict[str, tuple[float, EnemyUnit]] = {}
        self._first_count: dict[str, int] = {}
        self._pushes: dict[str, EnemyPush] = {}
        self._friends: list[EnemyUnit] = []
        self._phase = PHASE_SINGLE
        self._hold_reason = ""
        self.recent: list[int] = []
        self.deck = DeckTracker(self.hero_card)
        # Tank-led win condition in hand but not affordable yet (from the full hand).
        self._waiting_win: str | None = None
        # The whole hand with UNKNOWN slots named by the deck tracker (all 4 slots when known).
        self._named_hand: dict[int, str] = {}
        # Lumberloon-style pair: (win card, lane, drop spot, until) after the leader went down.
        self._pending_win: tuple[str, str, tuple[int, int], float] | None = None
        self._tank_led_until = 0.0
        self.opponent = OpponentModel(clock)
        self.situation = ""

    def _pick(self, candidates: list[Card]) -> Card:
        """Prefer known cards, then cards not played recently (keeps the cycle moving), then random."""
        known = [c for c in candidates if _known(c[1])] or candidates
        fresh = [c for c in known if c[0] not in self.recent]
        return self._rng.choice(fresh or known)

    def _cheapest(
        self, candidates: list[Card], target: int = 0, penalty: Callable[[Card], float] | None = None
    ) -> Card:
        """The card whose cost is closest to ``target`` (0 = just the cheapest).

        Recently played slots and the unreadable card get a penalty so one card isn't spammed
        every time and a known card wins a tie.
        """

        def score(card: Card) -> float:
            value = abs(card_cost(card[1]) - target) + (_RECENT_PENALTY if card[0] in self.recent else 0)
            if not _known(card[1]):
                value += _UNKNOWN_PENALTY
            return value + (penalty(card) if penalty else 0)

        best = min(score(c) for c in candidates)
        return self._rng.choice([c for c in candidates if score(c) == best])

    def _role(self, card_id: str) -> str:
        """Offense role: the inferred win condition (e.g. Giant Skeleton) plays as ROLE_WIN."""
        inferred = self.deck.inferred_win()
        if inferred and _known(card_id) and _base_card_id(card_id) == inferred:
            return ROLE_WIN
        return card_role(card_id)

    def _is_back_tank(self, card_id: str) -> bool:
        if not _known(card_id):
            return False
        return _is(card_id, BACK_TANKS) or ck.has_role(card_id, ck.WIN2) or card_id == self.deck.inferred_win()

    def _win_lane(self, lane: str) -> str:
        """With a tank-led win condition in the deck: the weaker live enemy tower's lane."""
        if not any(c in TANK_LED_WINS for c in self.deck.deck):
            return lane
        left, right = self.enemy_tower_hp("left"), self.enemy_tower_hp("right")
        if left is None or right is None or left <= 0 or right <= 0:
            return lane
        if left + _WIN_LANE_HP_MARGIN <= right:
            return "left"
        if right + _WIN_LANE_HP_MARGIN <= left:
            return "right"
        return lane

    def _lone_win_ok(self, cards: list[Card], lane: str, elixir: int | None, elixir_full: bool, mode: str) -> bool:
        """A tank-led win condition without a tank: behind a live push of ours, at a low enemy
        tower, or about to leak with no tank in hand (never as pressure while we're defending)."""
        if mode in ("follow", "counter"):
            return True
        hp = self.enemy_tower_hp(lane)
        if hp is not None and 0 < hp <= WEAK_TOWER_HP:
            return True
        if mode != "normal":
            return False
        near_leak = elixir_full or (elixir is not None and elixir >= NEAR_LEAK_BY_PHASE[self._phase])
        tank_in_hand = any(self._is_back_tank(c[1]) or ck.has_role(c[1], ck.TANK) for c in cards if _known(c[1]))
        return near_leak and not tank_in_hand

    # ---- towers --------------------------------------------------------------

    def observe_towers(self, health: TowerHealth | None) -> None:
        """Feed one tower-HP read; a median over the last few seconds smooths troops covering the bars.

        Missing or None values (unreadable bar) are skipped, not counted as a dead tower — unless
        the tower was already weak and its bar stays missing for TOWER_DESTROYED_MISS_S, which
        is what a destroyed tower looks like (then it reads 0).
        """
        if health is None:
            return
        now = self._clock()
        for side, store in (("enemy", self._enemy_hp), ("own", self._own_hp)):
            values = getattr(health, side, None)
            if not values:
                continue
            for lane in ("left", "right"):
                value = values.get(lane)
                key = (side, lane)
                if value is not None:
                    self._tower_miss_since.pop(key, None)
                    store[lane].append((now, float(value)))
                    continue
                last = self._median(store[lane])
                if last is None or last <= 0 or last > WEAK_TOWER_HP:
                    continue
                since = self._tower_miss_since.setdefault(key, now)
                if now - since >= TOWER_DESTROYED_MISS_S:
                    store[lane].clear()
                    store[lane].append((now, 0.0))
                    self._tower_miss_since.pop(key, None)

    @staticmethod
    def _median(reads: collections.deque[tuple[float, float]]) -> float | None:
        if not reads:
            return None
        latest = reads[-1][0]
        return statistics.median(hp for t, hp in reads if latest - t <= TOWER_WINDOW_S)

    def enemy_tower_hp(self, lane: str) -> float | None:
        return self._median(self._enemy_hp[lane])

    def own_tower_hp(self, lane: str) -> float | None:
        return self._median(self._own_hp[lane])

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
        """Elixir to keep for defense after an offensive play (at least 4 once a tower of ours is low)."""
        if self.opponent.deck.win_condition_out_of_cycle():
            base = PUNISH_DEFENSE_RESERVE
        else:
            base = PRESSURE_DEFENSE_RESERVE if self.opponent.style() == ARCHETYPE_PRESSURE else DEFENSE_RESERVE
        return max(base, LOW_TOWER_RESERVE) if self._own_tower_low() else base

    def _own_tower_low(self) -> bool:
        """One of our princess towers is below OWN_TOWER_LOW_HP (or destroyed)."""
        return any((hp := self.own_tower_hp(lane)) is not None and hp < OWN_TOWER_LOW_HP for lane in ("left", "right"))

    def _push_reserve_ok(self, cost: int, elixir: int | None, elixir_full: bool = False) -> bool:
        """Starting a push with a low tower of ours: keep LOW_TOWER_RESERVE back for defense."""
        if elixir is None or elixir_full or not self._own_tower_low():
            return True
        return elixir - cost >= LOW_TOWER_RESERVE

    # ---- defense sizing, locks, memory -----------------------------------------

    def _summarize(
        self, threats: Sequence[EnemyUnit], friends: Sequence[EnemyUnit], track: bool = False
    ) -> dict[str, EnemyPush]:
        """summarize_pushes plus the count each push had when first seen."""
        pushes = summarize_pushes(threats, friends)
        if track:
            self._first_count = {lane: self._first_count.get(lane, p.count) for lane, p in pushes.items()}
        return {lane: replace(p, first_count=self._first_count.get(lane, p.count)) for lane, p in pushes.items()}

    def defense_budget(self, threat: EnemyPush | EnemyUnit) -> int:
        """Elixir a defense against ``threat`` needs: 3 for a lone troop, 4 for two, 5 for a big
        push or a tank (3 for a swarm a spell clears). Units use their lane's last push."""
        push = threat if isinstance(threat, EnemyPush) else self._pushes.get(threat.lane)
        if push is None:
            return 3
        if push.spellable:
            return 3
        if push.tank or push.count >= 3:
            return 5
        return 4 if push.count == 2 else 3

    def note_threat(self, threat: EnemyPush | EnemyUnit, now: float | None = None) -> None:
        """Remember the latest threat per lane, so a frame that misses it doesn't flip to offense."""
        unit = threat.lead if isinstance(threat, EnemyPush) else threat
        self._threat_memory[unit.lane] = (self._clock() if now is None else now, unit)

    def _remembered(self, within_s: float, exclude: Mapping[str, EnemyPush] | None = None) -> list[EnemyUnit]:
        now = self._clock()
        return [
            unit
            for lane, (at, unit) in self._threat_memory.items()
            if now - at <= within_s and not (exclude and lane in exclude)
        ]

    def _lane_unlocked(self, threat: EnemyUnit, push: EnemyPush | None, now: float) -> bool:
        commit = self._commits.get(threat.lane)
        if commit is None:
            return True
        cooldown = WEAK_TOWER_DEFENSE_COOLDOWN_S if self._own_tower_weak(threat.lane) else DEFENSE_COOLDOWN_S
        if now - commit.at >= max(cooldown, DEFENSE_LOCK_S):
            return True
        if threat.y > commit.threat_y + _DEEPER_PUSH_DY:
            return True
        if push is not None and push.count > commit.count:
            if not commit.growth or now - max(commit.growth) >= _LOCK_GROWTH_MIN_GAP_S:
                commit.growth.add(now)
            if len(commit.growth) >= _LOCK_GROWTH_READS:
                return True
        hp = self.own_tower_hp(threat.lane)
        return commit.own_hp is not None and hp is not None and commit.own_hp - hp > _LOCK_HP_DROP

    def should_defend_now(self, threats: Sequence[EnemyUnit], friends: Sequence[EnemyUnit] = ()) -> EnemyUnit | None:
        """Return the threat to answer immediately, or None if it's already handled.

        Threats against a low tower come first (the lowest tower in overtime); otherwise the
        deepest one. A push our troops already outnumber is left to them, and a lane we just
        defended stays locked unless the push grows, gets deeper or hurts the tower.
        """
        now = self._clock()
        pushes = self._summarize(threats, friends)
        self._pushes = pushes
        if self._phase in _OVERTIME:
            ordered = sorted(threats, key=lambda t: (self.own_tower_hp(t.lane) or 1.0, -t.y))
        else:
            ordered = sorted(threats, key=lambda t: (not self._own_tower_weak(t.lane), -t.y))
        for threat in ordered:
            push = pushes.get(threat.lane)
            if push and push.handled:
                lead = push.lead
                near = sum(
                    1 for f in friends if abs(f.x - lead.x) <= _HANDLED_RADIUS and abs(f.y - lead.y) <= _HANDLED_RADIUS
                )
                if (not push.swarm and not self._own_tower_weak(threat.lane)) or near >= push.count:
                    continue
            if self._lane_unlocked(threat, push, now):
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
        clock: MatchClock | float | None = None,
        full_hand: Mapping[int, str] | None = None,
        affordable: set[int] | None = None,
    ) -> Play | None:
        """Pick a play from the playable ``hand`` (index -> card id), or None to hold.

        ``elixir`` (0-10) turns on elixir discipline; without it the brain only knows
        whether elixir is full. ``clock`` (a MatchClock or seconds since the match started)
        overrides ``elapsed_s`` for the match phase. ``full_hand`` (all 4 slots) lets the
        deck tracker narrow the cost of an unreadable card.
        """
        if isinstance(clock, MatchClock):
            elapsed_s = clock.elapsed()
        elif clock is not None:
            elapsed_s = float(clock)
        self._phase = match_phase(elapsed_s)
        self._friends = list(friends)
        # Cost bounds first: they decide whether an UNKNOWN slot can be the hero card.
        if full_hand is not None and elixir is not None:
            self.deck.observe_costs(full_hand, set(hand) if affordable is None else affordable, elixir)
        named = self.deck.resolve_hand(full_hand if full_hand is not None else hand)
        hand = {
            i: c for i, c in hand.items() if not (c == UNKNOWN_CARD and i in self.deck.maybe_hero_slots)
        }  # an UNKNOWN that might be the hero (2+ candidates) is never played blind
        hand = {i: named.get(i, c) if c == UNKNOWN_CARD else c for i, c in hand.items()}
        self._waiting_win = next((c for i, c in named.items() if i not in hand and is_tank_led_win(c)), None)
        self._named_hand = dict(named)
        pushes = self._summarize(threats, friends, track=True)
        self._pushes = pushes
        self.opponent.observe(pushes, threats)
        self.deck.observe_hand(list(hand.values()))
        if elixir is not None and elixir >= 10:
            elixir_full = True
        self.situation = self._describe(elixir, pushes)
        now = self._clock()
        for push in pushes.values():
            self.note_threat(push, now)
        if not hand:
            return None
        cards = [(index, card_id, card_role(card_id)) for index, card_id in hand.items()]
        self._hold_reason = ""

        threat = self.should_defend_now(threats, friends) if threats else None
        if threat is not None:
            push = pushes.get(threat.lane) or EnemyPush(threat.lane, threat, (threat.x, threat.y), 1, False)
            play = self._defend(cards, push, threat, elixir)
            if play is not None or self._hold_reason:
                return play
            # Nothing that defends — pressure the other lane instead (never with a spell).
            return self._offense(cards, _other(threat.lane), "pressure", elapsed_s, elixir, elixir_full, pushes)

        # A push seen a moment ago that this frame missed: answer it or wait, never attack.
        missed = self._remembered(THREAT_MEMORY_S, exclude=pushes)
        if missed:
            remembered = self.should_defend_now(missed, friends)
            if remembered is None:
                return None
            push = EnemyPush(remembered.lane, remembered, (remembered.x, remembered.y), 1, False)
            return self._defend(cards, push, remembered, elixir, allow_spells=False)

        if self._is_opening(elapsed_s) and any(c[2] in _OPENING_PRIORITY for c in cards):
            lane = self.offense_lane()
            options = [c for c in cards if c[2] in _OPENING_PRIORITY]
            options = [c for c in options if _known(c[1])] or options
            index, card_id, picked = self._cheapest(options) if elixir is not None else self._pick(options)
            return Play(index, card_id, picked, OPENING_BACK_COORDS[lane], lane, False, "opening at the back")
        for tank_led in (self._pair_follow_up, self._tank_led_follow_up):
            play = tank_led(cards, elixir, elixir_full)
            if play is not None or self._hold_reason:
                return play
        if self._following_up():
            assert self._last_win is not None
            return self._offense(cards, self._last_win[0], "follow", elapsed_s, elixir, elixir_full, pushes)
        mode = "counter" if self._counter_pushing() else "normal"
        return self._offense(cards, self.offense_lane(), mode, elapsed_s, elixir, elixir_full, pushes)

    def _defend(
        self,
        cards: list[Card],
        push: EnemyPush,
        threat: EnemyUnit,
        elixir: int | None = None,
        allow_spells: bool = True,
    ) -> Play | None:
        """Answer ``threat``; None with ``self._hold_reason`` set means "wait", not "can't"."""
        lane = threat.lane
        self._hold_reason = ""
        # Air answers matter when the push is led by (or made only of) air units — a Baby
        # Dragon behind a Golem doesn't make Lumberjack useless.
        lead_traits = unit_traits(threat.kind)
        air_threat = (
            lead_traits.air if lead_traits is not None else bool(push.kinds) and all(t.air for t in push._traits())
        )

        def play(card: Card, coord: tuple[int, int], why: str) -> Play:
            return Play(card[0], card[1], card[2], coord, lane, True, why, threat.y)

        # 1. A lone heavy hitter at our tower: stun it (resets Inferno/Sparky/charges, knocks back divers).
        if push.count <= 2 and threat.y >= _TOWER_DIVE_Y and self._clock() - self._last_stun >= STUN_COOLDOWN_S:
            stuns = [c for c in cards if _is(c[1], STUN_CARDS) and (allow_spells or c[2] != ROLE_SPELL)]
            if stuns:
                card = self._cheapest(stuns)
                role = ROLE_SPELL if card[2] == ROLE_SPELL else ROLE_TROOP
                return play(card, defensive_drop(role, threat, card[1]), f"stunning the {lane} lane push to reset it")

        # 2. A real swarm: an area spell on the group the moment it crosses (a big one for a big swarm).
        if push.spellable and allow_spells:
            spells = [
                c
                for c in cards
                if c[2] in (ROLE_SPELL, ROLE_OWN_SIDE_SPELL)
                and _is(c[1], SMALL_SPELLS | BIG_SWARM_SPELLS)
                and not _is(c[1], NO_KILL_SPELLS)
                and not (push.air and _is(c[1], GROUND_ONLY_SPELLS))
            ]
            big = [c for c in spells if _is(c[1], BIG_SWARM_SPELLS)]
            center = EnemyUnit(push.center[0], push.center[1], threat.size)
            why = f"spelling the swarm in the {lane} lane"
            if big:
                card = self._cheapest(big)
                return play(card, defensive_drop(card[2], center, card[1]), why)
            # No big spell: an air-capable defender beats a small spell that leaves half the swarm.
            hitters = [
                c
                for c in cards
                if c[2] in (ROLE_TROOP, ROLE_SUPPORT, ROLE_BUILDING)
                and not _is(c[1], SWARM_CARDS)
                and (info := ck.card_info(c[1])) is not None
                and info.targets_air
            ]
            if hitters and (not push.kinds or push.air):
                card = self._cheapest(hitters)
                return play(card, defensive_drop(card[2], threat, card[1]), f"defending the {lane} lane swarm")
            if spells:
                card = self._cheapest(spells)
                return play(card, defensive_drop(card[2], center, card[1]), why)

        # 3. Size the defense: short on elixir far from the tower, wait; at the tower, only a cheap card.
        budget = self.defense_budget(push)
        short = elixir is not None and elixir < budget
        if short and threat.y < _AT_TOWER_Y:
            self._hold_reason = f"saving {budget} elixir for the {lane} lane push"
            return None

        # 4. Cheapest troop/building that handles it. Swarm cards stay out of multi-unit pushes
        #    and away from splash (Bowler/Valkyrie/Wizard wipes them) — also when a splash card
        #    of theirs we've seen could be back in hand.
        target = 3 if push.count == 1 or (push.count == 3 and not push.kinds) else 4 if push.count == 2 else 5
        defenders = [c for c in cards if c[2] in (ROLE_TROOP, ROLE_SUPPORT, ROLE_BUILDING)]
        any_troop = bool(defenders)
        if short:
            defenders = [c for c in defenders if card_cost(c[1]) <= _SHORT_DEFENSE_MAX_COST]
        defenders = [c for c in defenders if _known(c[1])] or defenders
        if air_threat:
            # Ground-only defenders can't touch Balloon/Minions; with no anti-air, fall through to spells.
            defenders = [c for c in defenders if not _is(c[1], GROUND_ONLY)]
        if push.count >= 2 or push.splash or self.opponent.deck.any_ready(SPLASH_UNITS):
            defenders = [c for c in defenders if not _is(c[1], SWARM_CARDS)] or defenders
        if push.tank:
            killers = [c for c in defenders if _is(c[1], TANK_KILLERS) or c[2] == ROLE_BUILDING]
            if killers:
                defenders, target = killers, 5
        if defenders:
            hedge = not push.kinds and push.count <= 2  # might be air: prefer cards that hit air

            def air_hedge(card: Card) -> float:
                return _AIR_HEDGE_PENALTY if hedge and _is(card[1], GROUND_ONLY) else 0.0

            card = self._cheapest(defenders, target, air_hedge)
            return play(card, defensive_drop(card[2], threat, card[1]), f"defending the {lane} lane")

        # 5. Spells on a 1-2 unit push only to save a weak tower, against a recognized air unit
        #    nothing in hand can hit, or at the tower with no troop to play.
        spells = [c for c in cards if c[2] in (ROLE_SPELL, ROLE_OWN_SIDE_SPELL)]
        if not spells:
            return None
        desperate = self._own_tower_weak(lane) or air_threat or (threat.y >= _TOWER_DIVE_Y and not any_troop)
        if allow_spells and (push.swarm or desperate):
            card = self._cheapest(spells)
            return play(card, defensive_drop(card[2], threat, card[1]), f"defending the {lane} lane")
        self._hold_reason = f"not wasting a spell on the {lane} lane push"
        return None

    # ---- offense ---------------------------------------------------------------

    def _counter_window_open(self) -> bool:
        return self._counter_push_lane is not None and self._clock() < self._counter_push_until

    def _counter_pushing(self) -> bool:
        """A counter-push needs a defender of ours that survived and is heading up the lane."""
        if not self._counter_window_open():
            return False
        return any(f.lane == self._counter_push_lane and f.y < _COUNTER_PUSH_FRIEND_Y for f in self._friends)

    def _following_up(self) -> bool:
        return self._last_win is not None and self._clock() - self._last_win[1] < self._last_win[2]

    def _back_tank_hold(self) -> bool:
        """The back tank is still walking: troop follow-ups would arrive alone."""
        if self._back_tank_drop is None or not self._following_up():
            return False
        lane, at = self._back_tank_drop
        if self._clock() - at >= FOLLOW_UP_HOLD_S:
            return False
        return not any(f.lane == lane and f.y < _TANK_AT_BRIDGE_Y for f in self._friends)

    def _behind_push(self, lane: str, dy: int = _FOLLOW_BEHIND_DY) -> tuple[int, int] | None:
        """``dy`` px behind our most advanced troop in ``lane``, or where the back tank should be by now."""
        y_min, y_max = OWN_HALF_DROP_Y_RANGE
        ours = [f for f in self._friends if f.lane == lane and f.y < y_max]
        if ours:
            lead = min(ours, key=lambda f: f.y)
            return _avoid_own_towers(lead.x, _clamp(lead.y + dy, y_min, y_max))
        if self._back_tank_drop is not None and self._back_tank_drop[0] == lane:
            back_x, back_y = BACK_TANK_COORDS[lane]
            tank_y = back_y - TANK_SPEED_PX_S * (self._clock() - self._back_tank_drop[1])
            progress = max(0.0, min(1.0, (back_y - tank_y) / (back_y - y_min)))
            x = int(back_x + (LANE_BRIDGE_X[lane] - back_x) * progress)
            return _avoid_own_towers(x, _clamp(tank_y + dy, y_min, y_max))
        return None

    def _pair_follow_up(self, cards: list[Card], elixir: int | None, elixir_full: bool) -> Play | None:
        """Right after the bridge leader (Lumberjack): the win condition just behind it, same lane."""
        if self._pending_win is None:
            return None
        win_id, lane, coord, until = self._pending_win
        if self._clock() >= until:
            self._pending_win = None
            return None
        wins = [c for c in cards if is_tank_led_win(c[1])]
        if wins:
            card = wins[0]
            reason = f"{_PAIR_REASON}: {_base_card_id(card[1])} right behind the lumberjack in the {lane} lane"
            return Play(card[0], card[1], ROLE_WIN, coord, lane, False, reason)
        if self._waiting_win is not None or (elixir is not None and elixir < card_cost(win_id)):
            if not elixir_full:
                self._hold_reason = f"saving elixir for the {_base_card_id(win_id)} behind the lumberjack"
            return None
        self._pending_win = None  # it isn't in hand any more
        return None

    def _tank_led_follow_up(self, cards: list[Card], elixir: int | None, elixir_full: bool) -> Play | None:
        """A back tank is walking up: the tank-led win condition goes ~1 tile behind it once it
        nears the bridge; until then (and until it's affordable) the elixir is saved for it."""
        if self._back_tank_drop is None or not self._following_up():
            return None
        lane = self._back_tank_drop[0]
        wins = [c for c in cards if is_tank_led_win(c[1])]
        if not wins and self._waiting_win is None:
            return None
        name = _base_card_id(wins[0][1] if wins else str(self._waiting_win))
        if self._back_tank_hold() or not wins:
            if not elixir_full:
                self._hold_reason = f"holding the {name} until the tank nears the bridge"
            return None
        coord = self._behind_push(lane, _TANK_LED_BEHIND_DY)
        if coord is None:
            return None
        card = wins[0]
        reason = f"{_TANK_LED_REASON}: {name} behind the tank in the {lane} lane"
        return Play(card[0], card[1], ROLE_WIN, coord, lane, False, reason)

    def _pair(self, cards: list[Card], lane: str, elixir: int | None, elixir_full: bool) -> Play | None:
        """Lumberloon: with lots of elixir, the Lumberjack at the bridge and the win condition
        right behind it on the next tick (see ``_pair_follow_up``)."""
        if elixir is None:
            return None
        if not elixir_full and elixir < PAIR_MIN_BY_PHASE[self._phase]:
            return None
        wins = [c for c in cards if is_tank_led_win(c[1])]
        leaders = [
            c
            for c in cards
            if _known(c[1]) and _is(c[1], PAIR_LEADERS) and self._push_reserve_ok(card_cost(c[1]), elixir, elixir_full)
        ]
        if not wins or not leaders:
            return None
        lane = self._win_lane(lane)
        bridge = (LANE_BRIDGE_X[lane], OWN_HALF_DROP_Y_RANGE[0])
        index, card_id, picked = self._pick(leaders)
        reason = f"{_PAIR_REASON}: lumberjack at the {lane} bridge, {_base_card_id(wins[0][1])} next"
        return Play(index, card_id, picked, bridge, lane, False, reason)

    def _last_swarm_answer(self, card_id: str, hand_ids: Sequence[str]) -> bool:
        """``card_id`` is the only swarm spell in hand and none comes back within 2 plays."""
        if not _is(card_id, SMALL_SPELLS):
            return False
        base = _base_card_id(card_id)
        if any(_known(c) and _is(c, SMALL_SPELLS) and _base_card_id(c) != base for c in hand_ids):
            return False
        for other in self.deck.deck:
            if other != base and other in SMALL_SPELLS:
                n = self.deck.plays_until(other, hand_ids)
                if n is not None and n <= 2:
                    return False
        return True

    def _finishing_spell(self, cards: list[Card]) -> Play | None:
        """A spell that kills an enemy princess tower outright (0 < hp <= what the spell removes)."""
        best: tuple[float, Card, str] | None = None
        for card in cards:
            if card[2] != ROLE_SPELL:
                continue
            fraction = SPELL_TOWER_FRACTION.get(_base_card_id(card[1]), 0.0)
            for lane in ("left", "right"):
                hp = self.enemy_tower_hp(lane)
                if hp is not None and 0 < hp <= fraction and (best is None or hp < best[0]):
                    best = (hp, card, lane)
        if best is None:
            return None
        _, card, lane = best
        coord = ENEMY_PRINCESS_TOWER_CENTERS[lane]
        return Play(card[0], card[1], card[2], coord, lane, False, f"finishing the {lane} tower")

    def _tiebreak_chip(self, cards: list[Card], elapsed_s: float) -> Play | None:
        """Last seconds of overtime: any tower damage can decide the game."""
        if self._phase not in _OVERTIME or elapsed_s < OT_TIEBREAK_CHIP_S:
            return None
        spells = [c for c in cards if c[2] == ROLE_SPELL and _base_card_id(c[1]) in SPELL_TOWER_FRACTION]
        alive = [(hp, lane) for lane in ("left", "right") if (hp := self.enemy_tower_hp(lane)) is not None and hp > 0]
        if not spells or not alive:
            return None
        lane = min(alive)[1]
        card = max(spells, key=lambda c: SPELL_TOWER_FRACTION[_base_card_id(c[1])])
        coord = ENEMY_PRINCESS_TOWER_CENTERS[lane]
        return Play(card[0], card[1], card[2], coord, lane, False, f"tiebreak chip on the {lane} tower")

    def _offense_tiers(self, mode: str) -> list[Callable[[Card], bool]]:
        """Card filters in play order for this offense mode."""

        def win(c: Card) -> bool:
            return self._role(c[1]) == ROLE_WIN

        def tank(c: Card) -> bool:
            return self._role(c[1]) != ROLE_WIN and ck.has_role(c[1], ck.WIN2, ck.TANK)

        def support(c: Card) -> bool:
            return self._role(c[1]) == ROLE_SUPPORT and not ck.has_role(c[1], ck.CY)

        def cycle(c: Card) -> bool:
            return _known(c[1]) and self._role(c[1]) in (ROLE_TROOP, ROLE_SUPPORT) and ck.has_role(c[1], ck.CY)

        def troop(c: Card) -> bool:
            return _known(c[1]) and self._role(c[1]) == ROLE_TROOP and not tank(c) and not cycle(c)

        def unknown(c: Card) -> bool:
            return not _known(c[1])

        if mode == "follow":
            # Back the push up instead of stacking a second win condition on it.
            return [troop, support, cycle, unknown, tank, win]
        if mode == "counter":
            return [win, tank, troop, support, cycle, unknown]
        return [win, tank, support, cycle, troop, unknown]

    def _offense(
        self,
        cards: list[Card],
        lane: str,
        mode: str,
        elapsed_s: float,
        elixir: int | None,
        elixir_full: bool,
        pushes: Mapping[str, EnemyPush],
    ) -> Play | None:
        finish = self._finishing_spell(cards)
        if finish is not None:
            return finish
        spells_blocked = bool(pushes) or bool(self._remembered(SPELL_BLOCK_MEMORY_S))
        # A tank-led push is in: spells stay for the swarms that defend it, not the tower.
        spells_blocked = spells_blocked or self._clock() < self._tank_led_until
        if not spells_blocked:
            chip = self._tiebreak_chip(cards, elapsed_s)
            if chip is not None:
                return chip
        if mode == "normal":
            for special in (
                self._back_tank(cards, self._win_lane(lane), elixir, elixir_full),
                self._pair(cards, lane, elixir, elixir_full),
                self._spawner(cards, lane, elixir, pushes),
                self._cycle(cards, lane, elixir) if elixir is not None and not elixir_full else None,
            ):
                if special is not None:
                    return special

        reason = {
            "pressure": f"pressuring the {lane} lane",
            "follow": f"backing up the push in the {lane} lane",
            "counter": f"counter-pushing in the {lane} lane",
        }.get(mode, f"pushing the {lane} lane")
        hold = mode == "follow" and self._back_tank_hold()
        for tier in self._offense_tiers(mode):
            options = [
                c
                for c in cards
                if c[2] not in (ROLE_SPELL, ROLE_BUILDING, ROLE_UTILITY, ROLE_OWN_SIDE_SPELL)
                and tier(c)
                and self._affordable_on_offense(c, elixir, elixir_full, mode)
                and (not is_tank_led_win(c[1]) or self._lone_win_ok(cards, lane, elixir, elixir_full, mode))
            ]
            if elixir is not None and not elixir_full:
                # Back tanks wait for the full-elixir push (see _back_tank), never a naked bridge drop.
                options = [c for c in options if not self._is_back_tank(c[1])]
            if hold:
                options = [
                    c for c in options if self._role(c[1]) in (ROLE_WIN, ROLE_SUPPORT) and not is_tank_led_win(c[1])
                ]
            if not options:
                continue
            index, card_id, _ = self._pick(options)
            role = self._role(card_id)
            coord = self._offense_coord(card_id, role, lane, mode, elapsed_s)
            if coord is None:
                continue
            return Play(index, card_id, role, coord, lane, False, reason)

        if elixir_full:
            return self._leak(cards, lane, elapsed_s, spells_blocked)
        return None

    def _offense_coord(self, card_id: str, role: str, lane: str, mode: str, elapsed_s: float) -> tuple[int, int] | None:
        if mode in ("follow", "counter") and (role != ROLE_WIN or is_tank_led_win(card_id)):
            dy = _TANK_LED_BEHIND_DY if is_tank_led_win(card_id) else _FOLLOW_BEHIND_DY
            behind = self._behind_push(lane, dy) if (mode == "counter" or self._back_tank_drop) else None
            if behind is not None:
                return behind
        if not _known(card_id) or (role != ROLE_WIN and ck.has_role(card_id, ck.CY)):
            return OPENING_BACK_COORDS[lane]
        return self._offense_drop(card_id, role, lane, elapsed_s)

    def _leak(self, cards: list[Card], lane: str, elapsed_s: float, spells_blocked: bool) -> Play | None:
        """About to leak at 10: building, then utility / own-side spell, a spell only as a last resort."""
        hand_ids = [c[1] for c in cards]
        for role in (ROLE_BUILDING, ROLE_UTILITY, ROLE_OWN_SIDE_SPELL, ROLE_SPELL):
            options = [c for c in cards if c[2] == role]
            target = lane
            if role == ROLE_SPELL:
                if spells_blocked:
                    continue
                options = [c for c in options if not self._last_swarm_answer(c[1], hand_ids)]
                if self.enemy_tower_hp(lane) == 0:
                    target = _other(lane)  # never throw a spell at a destroyed tower
                    if self.enemy_tower_hp(target) == 0:
                        continue
            if not options:
                continue
            index, card_id, picked = self._pick(options)
            coord = self._offense_drop(card_id, picked, target, elapsed_s)
            if coord is not None:
                return Play(index, card_id, picked, coord, target, False, "spending elixir before it leaks")
        return None

    def _affordable_on_offense(self, card: Card, elixir: int | None, elixir_full: bool, mode: str = "normal") -> bool:
        """Elixir discipline: offense keeps a defense reserve unless backed by a reason."""
        if elixir is None or elixir_full:
            return True
        if card[2] in (ROLE_SPELL, ROLE_BUILDING, ROLE_UTILITY, ROLE_OWN_SIDE_SPELL):
            return True  # gated by their own rules (finishing / spawner / leak)
        cost = card_cost(card[1])
        if mode in ("normal", "pressure") and not self._push_reserve_ok(cost, elixir):
            return False  # a low tower of ours: new pushes keep LOW_TOWER_RESERVE back
        if self._role(card[1]) == ROLE_WIN:
            return (mode == "counter" and elixir - cost >= 1) or elixir - cost >= self.reserve()
        if mode == "follow":
            return elixir - cost >= max(1, self.reserve() - 1)  # committing to a live push is worth it
        if mode == "counter":
            return elixir - cost >= 1 and self._counter_push_spent + cost <= COUNTER_PUSH_SPEND_CAP
        return elixir >= NEAR_LEAK_BY_PHASE[self._phase]

    def _back_tank(self, cards: list[Card], lane: str, elixir: int | None, elixir_full: bool) -> Play | None:
        tanks = [c for c in cards if self._is_back_tank(c[1])]
        if not tanks or elixir is None:
            return None
        if self.opponent.style() == ARCHETYPE_PRESSURE and not elixir_full:
            return None  # a slow push at the back gets punished by bridge spam in the other lane
        threshold = BACK_TANK_MIN_BY_PHASE[self._phase]
        if any(is_tank_led_win(c) for c in self._named_hand.values()):
            threshold = min(threshold, TANK_LED_BACK_TANK_MIN)  # the Balloon follows the tank
        if not elixir_full and elixir < max(threshold, *(card_cost(c[1]) for c in tanks)):
            return None
        tanks = [c for c in tanks if self._push_reserve_ok(card_cost(c[1]), elixir, elixir_full)]
        if not tanks:
            return None
        index, card_id, _ = self._pick(tanks)
        return Play(
            index,
            card_id,
            self._role(card_id),
            BACK_TANK_COORDS[lane],
            lane,
            False,
            f"starting a push at the back in the {lane} lane",
        )

    def _spawner(
        self, cards: list[Card], lane: str, elixir: int | None, pushes: Mapping[str, EnemyPush]
    ) -> Play | None:
        """Empty board, plenty of elixir and no win card to start a push: a spawner builds value."""
        if elixir is None or elixir < CYCLE_MIN_ELIXIR or pushes:
            return None
        if any((self._role(c[1]) == ROLE_WIN and not is_tank_led_win(c[1])) or self._is_back_tank(c[1]) for c in cards):
            return None  # a held tank-led win condition (waiting for its tank) doesn't block value
        spawners = [c for c in cards if _is(c[1], SPAWNER_BUILDINGS) and self._push_reserve_ok(card_cost(c[1]), elixir)]
        if not spawners:
            return None
        index, card_id, picked = self._pick(spawners)
        return Play(
            index, card_id, picked, DEFENSE_BUILDING_PULL_COORDS[lane], lane, False, "placing a spawner for value"
        )

    def _cycle(self, cards: list[Card], lane: str, elixir: int) -> Play | None:
        """No win condition in hand: spend a cheap card at the back to bring one around."""
        if elixir < CYCLE_MIN_BY_PHASE[self._phase] or any(self._role(c[1]) == ROLE_WIN for c in cards):
            return None
        hand = [c[1] for c in cards]
        waits = [(n, w) for w in self.deck.win_conditions() if (n := self.deck.plays_until(w, hand))]
        if not waits:
            return None
        cheap = [
            c
            for c in cards
            if _known(c[1])
            and c[2] in (ROLE_TROOP, ROLE_SUPPORT)
            and card_cost(c[1]) <= _CYCLE_MAX_COST
            and self._push_reserve_ok(card_cost(c[1]), elixir)
        ]
        if not cheap:
            return None
        plays_away, win = min(waits)
        index, card_id, picked = self._cheapest(cheap)
        return Play(
            index, card_id, picked, OPENING_BACK_COORDS[lane], lane, False,
            f"cycling toward {win} ({plays_away} play{'s' if plays_away > 1 else ''} away)",
        )  # fmt: skip

    def _is_opening(self, elapsed_s: float) -> bool:
        return not self._played_any and elapsed_s < OPENING_WINDOW_S

    def _offense_drop(self, card_id: str, role: str, lane: str, elapsed_s: float) -> tuple[int, int] | None:
        if role == ROLE_BUILDING:
            return DEFENSE_BUILDING_PULL_COORDS[lane]
        if role == ROLE_OWN_SIDE_SPELL:
            # Drop at the bridge on our side so its troops walk into the lane.
            return (DEFENSE_BUILDING_PULL_COORDS[lane][0] - (60 if lane == "left" else -60), OWN_HALF_DROP_Y_RANGE[0])
        coord = calculate_play_coords(get_card_group(card_id), lane, elapsed_s)
        return (int(coord[0]), int(coord[1])) if coord is not None else None

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

    def record(self, play: Play, deployed: bool = True) -> None:
        """Remember a play we made (call after the clicks went through).

        ``deployed=False`` (tapped, but the card never went down) only logs the attempt for
        the deck tracker: no cooldown, rotation or recent-slot change.
        """
        self.deck.played(play.card_id, play.index, deployed)
        if not deployed:
            return
        self._played_any = True
        self.recent = [*self.recent[-2:], play.index]
        now = self._clock()
        if play.reason.startswith("stunning"):
            self._last_stun = now
        if play.defensive:
            threat_y = play.threat_y if play.threat_y is not None else play.coord[1]
            self._last_defense[play.lane] = (now, threat_y)
            push = self._pushes.get(play.lane)
            self._commits[play.lane] = _DefenseCommit(
                now, threat_y, push.count if push else 1, self.own_tower_hp(play.lane), set()
            )
            self._counter_push_lane = play.lane
            self._counter_push_until = now + COUNTER_PUSH_WINDOW_S
            self._counter_push_spent = 0
            return
        if self._counter_window_open() and play.role != ROLE_WIN:
            self._counter_push_spent += card_cost(play.card_id)
        if play.reason.startswith(_PAIR_REASON) and not is_tank_led_win(play.card_id):
            win_id = next((c for c in self.deck.deck if c in TANK_LED_WINS), "balloon")
            behind = (play.coord[0], play.coord[1] + _TANK_LED_BEHIND_DY)
            self._pending_win = (win_id, play.lane, behind, now + PAIR_FOLLOW_S)
        tank_led = is_tank_led_win(play.card_id)
        if tank_led:
            self._pending_win = None
            self._tank_led_until = now + TANK_LED_PUSH_S
        if play.coord == BACK_TANK_COORDS[play.lane] and (play.role == ROLE_WIN or self._is_back_tank(play.card_id)):
            self._last_win = (play.lane, now, BACK_TANK_FOLLOW_UP_S)
            self._back_tank_drop = (play.lane, now)
        elif play.role == ROLE_WIN:
            self._last_win = (play.lane, now, SUPPORT_FOLLOW_UP_S)
            behind_our_tank = self._back_tank_drop is not None and self._back_tank_drop[0] == play.lane
            if not (tank_led and behind_our_tank):
                self._back_tank_drop = None  # keep following the tank the win condition went in behind
