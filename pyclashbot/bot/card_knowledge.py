"""What each of our hand-detectable cards *is*, beyond a cost and a placement group.

A small hand-curated table the battle brain reads to reason about a card: whether it
tanks, kills tanks, cycles, spawns, flies, hits air, and where it goes on defense and
offense. Deliberately strategic only — no HP / DPS numbers (the brain never does damage
math; it works in pixels and elixir).

Roles (a card can have several):

* ``WIN``  — primary win condition (Hog, Golem, Balloon...).
* ``WIN2`` — secondary win condition: a card that can carry a push when the deck has no
  ``WIN`` card (Giant Skeleton).
* ``TANK`` — soaks damage in front of a push.
* ``MT``   — mini tank / melee bruiser.
* ``TK``   — high single-target damage: answers tanks.
* ``CY``   — cheap cycle card.
* ``SUP``  — ranged support that sits behind the action.
* ``BLD``  — building; ``SPN`` — spawner building.
* ``SPs`` / ``SPm`` — small / medium-or-large damage spell.

Defensive placement (``def_place``): ``ON`` the threat (cheap distractions), ``FRONT`` of it
(a tank it walks into), ``BEHIND`` it (melee, tower helps), ``BACK`` (ranged support),
``PULL`` (center pull spot for buildings), ``LEAD`` (spell landing a little ahead).
Offensive placement (``off_place``): ``BRIDGE``, ``BEHIND_TANK``, ``BACK``.

This module is a leaf: it imports nothing from ``pyclashbot.bot``.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---- roles -------------------------------------------------------------------------
WIN = "WIN"
WIN2 = "WIN2"
TANK = "TANK"
MT = "MT"
TK = "TK"
CY = "CY"
SUP = "SUP"
BLD = "BLD"
SPN = "SPN"
SPS = "SPs"
SPM = "SPm"
ROLES = frozenset({WIN, WIN2, TANK, MT, TK, CY, SUP, BLD, SPN, SPS, SPM})

# ---- kinds -------------------------------------------------------------------------
TROOP = "troop"
BUILDING = "building"
SPELL = "spell"
CHAMPION = "champion"

# ---- placement styles --------------------------------------------------------------
ON = "ON"
FRONT = "FRONT"
BEHIND = "BEHIND"
BACK = "BACK"
PULL = "PULL"
LEAD = "LEAD"
DEF_PLACES = frozenset({ON, FRONT, BEHIND, BACK, PULL, LEAD})
BRIDGE = "BRIDGE"
BEHIND_TANK = "BEHIND_TANK"
OFF_PLACES = frozenset({BRIDGE, BEHIND_TANK, BACK})


@dataclass(frozen=True)
class CardInfo:
    id: str
    cost: int
    kind: str
    roles: frozenset[str]
    fly: bool = False
    targets_air: bool = False
    ground_only_spell: bool = False
    def_place: str = BEHIND
    off_place: str = BRIDGE

    def has(self, *roles: str) -> bool:
        return any(r in self.roles for r in roles)


# Detected ids that name the same card as another id (fingerprint variants, typos).
_ALIASES = {
    "fire_cracker": "firecracker",
    "spirit_empress_air": "spirit_empress",
    "spirit_empress_ground": "spirit_empress",
    "furnace_evo_charging": "furnace",
    "lumberjack_evo_ready": "lumberjack",
}


def base_card_id(card_id: str) -> str:
    """The base card an id stands for: drops evo_/hero_ prefixes and folds detection aliases."""
    for prefix in ("evo_", "hero_"):
        if card_id.startswith(prefix):
            card_id = card_id[len(prefix) :]
            break
    return _ALIASES.get(card_id, card_id)


_KINDS = {"T": TROOP, "B": BUILDING, "S": SPELL, "C": CHAMPION}

# id | cost | kind (T/B/S/C) | roles | flags (F fly, A targets air, G ground-only spell) | def | off
# The user's deck comes first, then every other id in CARD_GROUPS / card_color_data / ELIXIR_COST.
_TABLE = """
giant_skeleton    6 T TANK,WIN2   -   FRONT  BACK
lumberjack        4 T MT,TK       -   BEHIND BEHIND_TANK
skeletons         1 T CY          -   ON     BEHIND_TANK
ice_spirit        1 T CY          A   ON     BEHIND_TANK
furnace           4 B BLD,SPN     A   PULL   BACK
arrows            3 S SPs         A   LEAD   BRIDGE
snowball          2 S SPs         A   LEAD   BRIDGE
archer_queen      5 C SUP         A   BACK   BEHIND_TANK
archers           3 T SUP         A   BACK   BEHIND_TANK
baby_dragon       4 T SUP         FA  BACK   BEHIND_TANK
balloon           5 T WIN         F   BEHIND BRIDGE
bandit            3 T MT          -   BEHIND BRIDGE
barb_barrel       2 S SPs         G   LEAD   BRIDGE
barb_hut          6 B BLD,SPN     -   PULL   BACK
barbarians        5 T TK          -   BEHIND BEHIND_TANK
bats              2 T CY          FA  ON     BEHIND_TANK
battle_healer     4 T MT          -   BEHIND BEHIND_TANK
battle_ram        4 T WIN         -   BEHIND BRIDGE
berserker         2 T CY          -   BEHIND BEHIND_TANK
bomb_tower        4 B BLD         -   PULL   BACK
bomber            2 T SUP         -   BACK   BEHIND_TANK
boss_bandit       6 C MT          -   BEHIND BRIDGE
bowler            5 T SUP         -   BEHIND BEHIND_TANK
cannon            3 B BLD,TK      -   PULL   BACK
cannon_cart       5 T MT,TK       -   BEHIND BRIDGE
clone             3 S -           A   LEAD   BRIDGE
dark_prince       4 T MT          -   BEHIND BEHIND_TANK
dart_goblin       3 T SUP         A   BACK   BEHIND_TANK
earthquake        3 S SPs         G   LEAD   BRIDGE
electro_dragon    5 T SUP         FA  BACK   BEHIND_TANK
electro_giant     7 T WIN,TANK    -   BEHIND BACK
electro_spirit    1 T CY          A   ON     BEHIND_TANK
electro_wizard    4 T SUP         A   BACK   BEHIND_TANK
elite_barbarians  6 T TK          -   BEHIND BRIDGE
elixir_collector  6 B BLD         -   PULL   BACK
elixir_golem      3 T WIN,TANK    -   BEHIND BACK
executioner       5 T SUP         A   BACK   BEHIND_TANK
fire_spirit       1 T CY          A   ON     BEHIND_TANK
fireball          4 S SPm         A   LEAD   BRIDGE
firecracker       3 T SUP         A   BACK   BEHIND_TANK
fisherman         3 T MT          -   BEHIND BEHIND_TANK
flying_machine    4 T SUP         FA  BACK   BEHIND_TANK
freeze            4 S SPm         A   LEAD   BRIDGE
giant             5 T WIN,TANK    -   BEHIND BACK
gob_curse         2 S SPs         A   LEAD   BRIDGE
goblin_barrel     3 S WIN         -   LEAD   BRIDGE
goblin_cage       4 B BLD,TK      -   PULL   BACK
goblin_demolisher 4 T SUP         -   BACK   BEHIND_TANK
goblin_drill      4 B WIN         -   BEHIND BRIDGE
goblin_gang       3 T CY          A   BEHIND BEHIND_TANK
goblin_giant      6 T WIN,TANK    A   BEHIND BACK
goblin_hut        5 B BLD,SPN     A   PULL   BACK
goblin_machine    5 T MT          -   BEHIND BRIDGE
goblins           2 T CY          -   ON     BEHIND_TANK
goblinstein       5 C TANK,MT     -   BEHIND BRIDGE
golden_knight     4 C MT          -   BEHIND BRIDGE
golem             8 T WIN,TANK    -   BEHIND BACK
graveyard         5 S WIN         -   LEAD   BRIDGE
guards            3 T MT          -   BEHIND BEHIND_TANK
heal_spirit       1 T CY          A   ON     BEHIND_TANK
hog               4 T WIN         -   BEHIND BRIDGE
hunter            4 T SUP,TK      A   BACK   BEHIND_TANK
ice_golem         2 T TANK,CY     -   FRONT  BRIDGE
ice_wizard        3 T SUP         A   BACK   BEHIND_TANK
inferno_dragon    4 T TK          FA  BACK   BEHIND_TANK
inferno_tower     5 B BLD,TK      A   PULL   BACK
knight            3 T MT          -   BEHIND BEHIND_TANK
lava_hound        7 T WIN,TANK    F   BEHIND BACK
lightning         6 S SPm         A   LEAD   BRIDGE
little_prince     3 C SUP         A   BACK   BEHIND_TANK
log               2 S SPs         G   LEAD   BRIDGE
magic_archer      4 T SUP         A   BACK   BEHIND_TANK
mega_knight       7 T TANK,MT     -   BEHIND BRIDGE
mega_minion       3 T SUP         FA  BACK   BEHIND_TANK
mighty_miner      4 C MT,TK       -   BEHIND BRIDGE
miner             3 T WIN         -   BEHIND BRIDGE
mini_pekka        4 T MT,TK       -   BEHIND BEHIND_TANK
minion_horde      5 T SUP         FA  BACK   BEHIND_TANK
minions           3 T SUP         FA  BACK   BEHIND_TANK
mirror            1 S -           A   LEAD   BRIDGE
monk              5 C MT          -   BEHIND BRIDGE
mortar            4 B WIN,BLD     -   PULL   BRIDGE
mother_witch      4 T SUP         A   BACK   BEHIND_TANK
musketeer         4 T SUP         A   BACK   BEHIND_TANK
night_witch       4 T SUP         -   BACK   BEHIND_TANK
pekka             7 T TANK,TK     -   BEHIND BRIDGE
phoenix           4 T SUP         FA  BACK   BEHIND_TANK
poison            4 S SPm         A   LEAD   BRIDGE
prince            5 T MT,TK       -   BEHIND BRIDGE
princess          3 T SUP         A   BACK   BEHIND_TANK
rage              2 S -           A   LEAD   BRIDGE
ram_rider         5 T WIN         -   BEHIND BRIDGE
rascals           5 T SUP         A   BEHIND BEHIND_TANK
rocket            6 S SPm         A   LEAD   BRIDGE
royal_delivery    3 S SPs         G   LEAD   BRIDGE
royal_ghost       3 T MT          -   BEHIND BRIDGE
royal_giant       6 T WIN,TANK    -   BEHIND BRIDGE
royal_hogs        5 T WIN         -   BEHIND BRIDGE
royal_recruits    7 T MT          -   BEHIND BRIDGE
rune_giant        4 T WIN,TANK    -   BEHIND BACK
skeleton_army     3 T TK          -   ON     BEHIND_TANK
skeleton_barrel   3 T WIN         F   BEHIND BRIDGE
skeleton_dragons  4 T SUP         FA  BACK   BEHIND_TANK
skeleton_king     4 C MT          -   BEHIND BRIDGE
sparky            6 T SUP         -   BACK   BEHIND_TANK
spear_goblins     2 T CY          A   BACK   BEHIND_TANK
spirit_empress    6 C SUP         A   BACK   BEHIND_TANK
suspicious_bush   2 T WIN         -   BEHIND BRIDGE
tesla             4 B BLD,TK      A   PULL   BACK
three_musketeers  9 T SUP         A   BACK   BEHIND_TANK
tombstone         3 B BLD,SPN     -   PULL   BACK
tornado           3 S SPs         A   LEAD   BRIDGE
valkyrie          4 T MT          -   BEHIND BEHIND_TANK
vines             3 S SPs         A   LEAD   BRIDGE
void              3 S SPm         A   LEAD   BRIDGE
wall_breakers     2 T WIN         -   BEHIND BRIDGE
witch             5 T SUP         A   BACK   BEHIND_TANK
wizard            5 T SUP         A   BACK   BEHIND_TANK
xbow              6 B WIN,BLD     -   PULL   BRIDGE
zap               2 S SPs         A   LEAD   BRIDGE
zappies           4 T SUP         A   BACK   BEHIND_TANK
"""


def _parse(table: str) -> dict[str, CardInfo]:
    cards: dict[str, CardInfo] = {}
    for line in table.strip().splitlines():
        card_id, cost, kind, roles, flags, def_place, off_place = line.split()
        role_set = frozenset() if roles == "-" else frozenset(roles.split(","))
        assert role_set <= ROLES, f"{card_id}: unknown role in {roles}"
        assert def_place in DEF_PLACES and off_place in OFF_PLACES, card_id
        assert card_id not in cards, f"duplicate card {card_id}"
        cards[card_id] = CardInfo(
            id=card_id,
            cost=int(cost),
            kind=_KINDS[kind],
            roles=role_set,
            fly="F" in flags,
            targets_air="A" in flags,
            ground_only_spell="G" in flags,
            def_place=def_place,
            off_place=off_place,
        )
    return cards


CARDS: dict[str, CardInfo] = _parse(_TABLE)


def card_info(card_id: str | None) -> CardInfo | None:
    """CardInfo for a detected id (evo_/hero_/alias ids resolve to the base card), or None."""
    if not card_id:
        return None
    return CARDS.get(base_card_id(card_id))


def has_role(card_id: str | None, *roles: str) -> bool:
    info = card_info(card_id)
    return info is not None and info.has(*roles)
