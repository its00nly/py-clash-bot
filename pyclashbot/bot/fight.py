"""random module for randomizing fight plays"""

import collections
import inspect
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy

from pyclashbot.bot import card_detection, state_detect
from pyclashbot.bot.battle_brain import BattleBrain, card_cost
from pyclashbot.bot.card_detection import (
    check_which_cards_are_available,
    create_default_bridge_iar,
    get_play_coords_for_card,
    identify_hand_cards,
    switch_side,
)
from pyclashbot.bot.coords import (
    CHAMPION_ABILITY_DISMISS_COORD,
    CLOSE_BATTLE_LOG_BUTTON,
    EMOTE_BUTTON_COORD,
    EMOTE_ICON_COORDS,
    HAND_CARDS_COORDS,
    PLAYABLE_PLAY_REGION_LTRB,
    QUICKMATCH_POPUP_BUTTON_COORD,
    START_FIGHT_BUTTON_COORD,
)
from pyclashbot.bot.find import EnemyUnit, find_enemy_units, find_friendly_units, read_tower_health
from pyclashbot.bot.nav import (
    get_to_activity_log,
    get_to_main_after_fight,
    select_mode,
    wait_for_battle_start,
    wait_for_clash_main_menu,
)
from pyclashbot.bot.recorder import (
    finish_fight_recording,
    is_recording,
    log_play,
    log_units,
    start_fight_recording,
    stop_fight_capture,
)
from pyclashbot.bot.state_detect import (
    check_if_battle_has_ended,
    check_if_battle_mode_is_selected,
    check_if_on_clash_main_menu,
    check_pixels_for_win_in_battle_log,
    in_battle_from_frame,
    read_elixir,
)
from pyclashbot.bot.unit_classifier import classify_units
from pyclashbot.utils.logger import Logger
from pyclashbot.utils.platform import get_log_dir
from pyclashbot.utils.versioning import __version__

# Smart battles wait only for the cheapest useful card; the brain decides the rest.
SMART_MIN_ELIXIR = 2
ELIXIR_WAIT_TIMEOUT = 40  # too high but someone got errors with that so idk

# Hero/Champion ability: fire once the button has been lit for ABILITY_READY_S across at least
# ABILITY_MIN_FRAMES consecutive frames and an enemy is on our half (smart mode), or after
# ABILITY_FALLBACK_S regardless. One unlit frame doesn't reset the timer; two in a row do.
ABILITY_READY_S = 1.0
ABILITY_MIN_FRAMES = 2
ABILITY_FALLBACK_S = 4.0
ABILITY_COOLDOWN_S = 5.0
ABILITY_MISSES_TO_RESET = 2

# Battle end: only after this long with no in-battle frame, then confirmed by check_if_battle_has_ended.
BATTLE_END_DEBOUNCE_S = 3.0
# Give up confirming and end the fight anyway after this long without an in-battle frame.
BATTLE_LOST_GIVE_UP_S = 15.0
LOST_FRAME_SLEEP_S = 0.25

# After a card play: decide again only on a frame captured this long after the taps returned.
PLAY_SETTLE_S = 0.3
# Smart loop: drop back to the elixir-wait preamble at least this often while holding.
PREAMBLE_BACKSTOP_S = 5.0
# A threat this deep is answered with whatever elixir we have.
DEEP_THREAT_Y = 375
# Mean absolute pixel change over a hand slot that counts as "a different card is there now".
SLOT_CHANGE_THRESHOLD = 18.0
# A card that failed to deploy stays raised in its slot; compare up to this many pixels higher.
SLOT_RAISE_MAX_DY = 12
# After a play that did not deploy, leave that slot alone this long before trying it again.
NOT_DEPLOYED_BACKOFF_S = 1.0
# Repeated hold / no-cards status lines and decision-log lines: at most this often unless they change.
REPEAT_STATUS_S = 1.0
DECISION_LOG_MIN_GAP_S = 1.0
# Decision logs kept on disk (oldest pruned when a new fight starts).
DECISION_LOG_KEEP = 30

# Per-decision JSONL logs go here (None = <log dir>/decisions); tests point it at a tmp dir.
DECISION_LOG_DIR: str | None = None


def _maybe_start_fight_recording(
    emulator, logger, recording_flag: bool, fight_mode_chosen: str, custom_path: str | None = None
) -> None:
    """Begin opt-in training-data capture for 1v1-type fights only (Trophy Road / Classic 1v1, never 2v2)."""
    if recording_flag and fight_mode_chosen in ["Classic 1v1", "Trophy Road"]:
        pack_mode = "1v1_trophy" if fight_mode_chosen == "Trophy Road" else "1v1_classic"
        start_fight_recording(emulator, pack_mode, __version__, logger=logger, custom_path=custom_path)


_hand_crop_setting: tuple[bool, str | None] | None = None


def _configure_hand_crops(recording_flag: bool, custom_path: str | None = None) -> None:
    """Unreadable hand-card crops are only kept when the user records fights (into their recordings folder)."""
    global _hand_crop_setting
    setting = (bool(recording_flag), custom_path)
    if setting == _hand_crop_setting:
        return
    try:
        from pyclashbot.bot import hand_crop_sink  # lazy: keeps recorder deps out of import

        root = hand_crop_sink.default_root(custom_path) if recording_flag else None
        hand_crop_sink.configure(root, enabled=bool(recording_flag))
        _hand_crop_setting = setting
    except Exception:  # crop capture must never break a battle
        pass


def do_fight_state(
    emulator,
    logger: Logger,
    random_fight_mode,
    fight_mode_chosen,
    called_from_launching=False,
    recording_flag: bool = False,
    custom_path: str | None = None,
    smart_battle: bool = False,
) -> bool:
    """Handle the entirety of a battle state (start fight, do fight, end fight)."""

    logger.change_status("Waiting for battle to start")

    # Wait for battle start
    if wait_for_battle_start(emulator, logger) is False:
        logger.change_status("Timed out waiting for battle to start")
        return False

    logger.change_status("Starting fight loop")
    logger.log(f'This is the fight mode: "{fight_mode_chosen}"')

    # Recording is started/stopped inside the fight loops themselves so each pack
    # brackets exactly the in-battle play loop (no pre-battle wait or post-fight nav).
    # Run regular fight loop if random mode not toggled
    if (
        not random_fight_mode
        and _fight_loop(emulator, logger, recording_flag, fight_mode_chosen, custom_path, smart_battle) is False
    ):
        logger.change_status("Fight loop failed")
        return False

    # Run random fight loop if random mode toggled
    if (
        random_fight_mode
        and _random_fight_loop(emulator, logger, recording_flag, fight_mode_chosen, custom_path) is False
    ):
        logger.change_status("Fight loop failed")
        return False

    # Only log the fight if not called from the start
    if not called_from_launching:
        if fight_mode_chosen in ["Classic 1v1", "Trophy Road"]:
            logger.add_1v1_fight()
        elif fight_mode_chosen == "Classic 2v2":
            logger.increment_2v2_fights()

        if fight_mode_chosen == "Trophy Road":
            logger.increment_trophy_road_fights()
        elif fight_mode_chosen == "Classic 1v1":
            logger.increment_classic_1v1_fights()
        elif fight_mode_chosen == "Classic 2v2":
            logger.increment_classic_2v2_fights()

    time.sleep(10)
    return True


def ensure_battle_mode_selected(emulator, logger, mode) -> bool:
    """Confirm ``mode`` is the selected battle mode, re-selecting it once if not."""
    if check_if_battle_mode_is_selected(emulator, mode):
        return True
    logger.change_status(f"{mode} is not the selected battle mode, re-selecting it")
    if select_mode(emulator, mode, logger) and check_if_battle_mode_is_selected(emulator, mode):
        return True
    logger.change_status(f"Could not confirm {mode} is selected — not starting the fight")
    return False


def start_fight(emulator, logger, mode) -> bool:
    """Start a fight with the specified mode.

    Args:
        emulator: The emulator controller
        logger: Logger instance
        mode: Fight mode - must be one of "Classic 1v1", "Classic 2v2", or "Trophy Road"

    Returns:
        bool: True if fight started successfully, False otherwise
    """
    # Validate mode parameter
    logger.log(f'Input mode type: "{type(mode)}"')
    logger.log(f"Input mode value: {mode}")
    valid_modes = ["Classic 1v1", "Classic 2v2", "Trophy Road"]
    logger.log(f"Valid modes: {valid_modes}")
    if mode not in valid_modes:
        logger.log(f"The valid modes for start_fight() are: {valid_modes}")
        logger.log(f"But start_fight() got an invalid mode: '{mode}'")
        return False

    logger.change_status(f"Starting a {mode} fight")

    # Check if on clash main menu
    logger.log("Checking if on main menu before starting fight...")
    if not check_if_on_clash_main_menu(emulator):
        logger.change_status("Not on main menu — cannot start fight")
        return False

    # Seasonal event modes (e.g. Road to Ronin, #744) can take over the battle
    # button after launch. Never press Start unless our mode is the selected one.
    if not ensure_battle_mode_selected(emulator, logger, mode):
        return False

    emulator.click(START_FIGHT_BUTTON_COORD[0], START_FIGHT_BUTTON_COORD[1])
    logger.log(f"Clicked Start button at {START_FIGHT_BUTTON_COORD}")

    # 2v2 needs a second popup after Start
    if mode == "Classic 2v2":
        logger.change_status("Classic 2v2 — clicking Quick Match popup...")
        time.sleep(3)
        emulator.click(QUICKMATCH_POPUP_BUTTON_COORD[0], QUICKMATCH_POPUP_BUTTON_COORD[1])
        logger.log(f"Clicked Quickmatch button at {QUICKMATCH_POPUP_BUTTON_COORD}")

    return True


def send_emote(emulator, logger: Logger):
    """Method to do an emote in a fight"""
    logger.change_status("Sending emote")

    # click emote button
    emulator.click(EMOTE_BUTTON_COORD[0], EMOTE_BUTTON_COORD[1])
    time.sleep(0.33)

    emote_coord = random.choice(EMOTE_ICON_COORDS)
    emulator.click(emote_coord[0], emote_coord[1])


# ===== One frame per tick ===============================================


@dataclass
class TickFrame:
    """Everything the fight loop reads from one screenshot."""

    iar: Any
    captured_at: float  # time.monotonic() when the capture started
    in_battle: bool
    elixir: int | None
    available: list[int]
    enemies: list[EnemyUnit] = field(default_factory=list)
    friends: list[EnemyUnit] = field(default_factory=list)
    threat: EnemyUnit | None = None


def _hero_ability_ready(iar) -> bool:
    """card_detection's frame predicate when it exists, else the local copy in state_detect."""
    shared = getattr(card_detection, "hero_ability_ready", None)
    if callable(shared):
        return bool(shared(iar))
    return state_detect.hero_ability_ready(iar)


def _available_slots(iar) -> list[int]:
    """Affordable hand slots in ``iar`` without taking another screenshot."""
    if iar is None:
        return []
    shared = getattr(card_detection, "cards_available", None)
    if callable(shared):
        return list(shared(iar))
    slots = []
    for i, (x_coords, y_coords) in enumerate(card_detection.card_coords):
        pixels = iar[numpy.ix_(y_coords, x_coords)].astype(numpy.int32)
        purple = numpy.all(numpy.abs(pixels - card_detection.purple_color) <= 30, axis=-1)
        if int(numpy.sum(purple)) >= 26:
            slots.append(i)
    return slots


def capture_frame(emulator) -> TickFrame:
    """Take exactly one screenshot and read battle state, elixir and hand availability from it."""
    captured_at = time.monotonic()
    iar = emulator.screenshot()
    card_detection.battle_iar = iar
    in_battle = in_battle_from_frame(iar)
    elixir = read_elixir(iar) if iar is not None else None
    available = _available_slots(iar) if in_battle else []
    return TickFrame(iar, captured_at, in_battle, elixir, available)


def _update_play_side() -> None:
    """Refresh card_detection.play_side from the current battle_iar (no screenshot)."""
    if card_detection.battle_iar is None or card_detection.bridge_iar is None:
        return
    try:
        _, card_detection.play_side = switch_side()
    except Exception:  # side detection is a nicety; never let it break a fight
        pass


def _tap_card(emulator, hand_xy, drop_xy) -> None:
    """Both taps of a card play, back to back (one ADB call when the emulator supports it)."""
    hand = (int(hand_xy[0]), int(hand_xy[1]))
    drop = (int(drop_xy[0]), int(drop_xy[1]))
    tap_sequence = getattr(emulator, "tap_sequence", None)
    if callable(tap_sequence):
        tap_sequence([hand, drop])
    else:
        emulator.click(*hand)
        emulator.click(*drop)


def _accepts(fn, name: str) -> bool:
    """True when callable ``fn`` takes a parameter called ``name``."""
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _brain_choose(
    brain,
    hand: dict[int, str],
    threats: list[EnemyUnit],
    elapsed: float,
    elixir: int | None,
    friends: list[EnemyUnit],
    clock: float | None,
    full_hand: dict[int, str] | None,
    affordable: set[int] | None = None,
):
    """brain.choose with the extras its signature supports (clock, full hand, affordable set)."""
    kwargs: dict[str, Any] = {"elixir": elixir, "friends": friends}
    if clock is not None and _accepts(brain.choose, "clock"):
        kwargs["clock"] = clock
    if full_hand is not None and _accepts(brain.choose, "full_hand"):
        kwargs["full_hand"] = full_hand
    if full_hand is not None and _accepts(brain.choose, "affordable"):
        kwargs["affordable"] = set(hand) if affordable is None else affordable
    return brain.choose(hand, threats, elapsed, **kwargs)


def _brain_record(brain, play, deployed: bool) -> None:
    """brain.record(play, deployed=...) when supported; otherwise only real deploys are recorded."""
    if _accepts(brain.record, "deployed"):
        brain.record(play, deployed=deployed)
    elif deployed:
        brain.record(play)


def _identify_hand(emulator, iar, available: list[int]) -> tuple[dict[int, str], dict[int, str] | None]:
    """(playable hand, full 4-slot hand or None). Uses identify_hand_cards_scored when present."""
    scored = getattr(card_detection, "identify_hand_cards_scored", None)
    if callable(scored):
        try:
            full = {int(entry[0]): str(entry[1]) for entry in scored(iar, slots=(0, 1, 2, 3))}
            return {i: full[i] for i in available if i in full}, full
        except Exception:
            pass  # fall back to the per-slot matcher
    return {index: identify_hand_cards(emulator, index) for index in available}, None


def _slot_fingerprint(iar, slot: int, dy: int = 0):
    """Coarse 6x6 colour grid of a hand slot (``dy`` px lower), for 'did a different card arrive' checks."""
    if iar is None:
        return None
    x, y = card_detection.toplefts[slot]
    y = max(0, y + dy)
    crop = iar[y : y + card_detection.TOTAL_HEIGHT, x : x + card_detection.TOTAL_WIDTH]
    if crop.size == 0 or crop.shape[0] != card_detection.TOTAL_HEIGHT:
        return None
    rows = numpy.array_split(crop.astype(numpy.float32), 6, axis=0)
    return numpy.array([[cell.mean(axis=(0, 1)) for cell in numpy.array_split(row, 6, axis=1)] for row in rows])


def _slot_changed(iar, slot: int, before) -> bool:
    """True when ``slot`` no longer shows the card in ``before`` (a merely raised card is the same card)."""
    if before is None:
        return False
    diffs = []
    for dy in range(-SLOT_RAISE_MAX_DY, 1):
        after = _slot_fingerprint(iar, slot, dy)
        if after is not None and after.shape == before.shape:
            diffs.append(float(numpy.mean(numpy.abs(after - before))))
    return bool(diffs) and min(diffs) >= SLOT_CHANGE_THRESHOLD


def _is_unknown(card_id: str | None) -> bool:
    return card_id is None or card_id.upper().startswith("UNKNOWN")


class AbilityTrigger:
    """Fires the Hero/Champion ability. Lives for the whole fight so 'ready since' survives ticks."""

    def __init__(self) -> None:
        self.ready_since: float | None = None
        self.hits = 0  # consecutive lit frames
        self.misses = 0  # consecutive unlit frames
        self.last_fired = -1e9
        self.fired = 0

    def tick(self, emulator, logger, iar, now: float, enemy_on_our_half: bool | None) -> bool:
        """Feed one frame; tap the ability if due. ``enemy_on_our_half`` None = don't gate on enemies.

        Call between card plays only (never between the two taps of a play).
        """
        if not _hero_ability_ready(iar):
            self.hits = 0
            self.misses += 1
            if self.misses >= ABILITY_MISSES_TO_RESET:
                self.ready_since = None
            return False
        self.misses = 0
        self.hits += 1
        if self.ready_since is None:
            self.ready_since = now
            logger.change_status("Hero/Champion ability ready")
        if now - self.last_fired < ABILITY_COOLDOWN_S:
            return False
        held = now - self.ready_since
        if self.hits < ABILITY_MIN_FRAMES or held < ABILITY_READY_S:
            return False
        if enemy_on_our_half is False and held < ABILITY_FALLBACK_S:
            return False
        emulator.click(*CHAMPION_ABILITY_DISMISS_COORD)
        logger.change_status("Triggered Hero/Champion ability")
        self.last_fired = now
        self.ready_since = None
        self.hits = 0
        self.fired += 1
        return True


@dataclass
class PendingPlay:
    play: Any
    slot: int
    card_id: str
    elixir_before: int | None
    fingerprint: Any
    tap_done_at: float
    raw_card_id: str | None = None  # the hand read before the brain resolved an UNKNOWN slot


class DecisionLog:
    """Per-decision JSONL lines for replaying fights through BattleBrain offline. Never raises."""

    def __init__(self, directory: str | None = None) -> None:
        self.path: str | None = None
        try:
            directory = directory or DECISION_LOG_DIR or os.path.join(get_log_dir("py-clash-bot"), "decisions")
            os.makedirs(directory, exist_ok=True)
            self._prune(directory)
            self.path = os.path.join(directory, time.strftime("%Y-%m-%d_%H-%M-%S") + ".jsonl")
        except OSError:
            self.path = None

    @staticmethod
    def _prune(directory: str) -> None:
        """Keep only the newest DECISION_LOG_KEEP - 1 logs, making room for this fight's."""
        logs = sorted(name for name in os.listdir(directory) if name.endswith(".jsonl"))
        for name in logs[: max(0, len(logs) - (DECISION_LOG_KEEP - 1))]:
            try:
                os.remove(os.path.join(directory, name))
            except OSError:
                pass

    def write(self, entry: dict) -> None:
        if self.path is None:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            pass


def _units_json(units) -> list[dict]:
    return [{"x": u.x, "y": u.y, "size": u.size, "kind": u.kind} for u in units]


class FightContext:
    """State shared by every tick of one fight: debounce, ability, pending play, logs."""

    def __init__(self, brain=None, decision_log: DecisionLog | None = None) -> None:
        self.brain = brain
        self.ability = AbilityTrigger()
        self.started_at = time.monotonic()
        self.last_in_battle_at = self.started_at
        self.first_in_battle_at: float | None = None
        self.last_end_check = -1e9
        self.last_lost_log_second = -1
        self.pending: PendingPlay | None = None
        self.decision_log = decision_log
        self.frames = 0
        self.plays = 0
        self.deployed = 0
        self.not_deployed = 0
        self.blocked_until: dict[int, float] = {}  # slot -> time it may be played again
        self.last_not_deployed_log = -1e9
        self.last_status: str | None = None
        self.last_status_at = -1e9
        self.last_decision_key: Any = None
        self.last_decision_at = -1e9

    @property
    def smart(self) -> bool:
        return self.brain is not None

    def match_clock(self, now: float) -> float:
        """Seconds since the first in-battle frame (the match-clock anchor)."""
        return now - self.first_in_battle_at if self.first_in_battle_at is not None else 0.0

    def next_frame(self, emulator) -> TickFrame:
        """Wait out the post-play settle time if needed, then take this tick's one screenshot."""
        if self.pending is not None:
            remaining = self.pending.tap_done_at + PLAY_SETTLE_S - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
        self.frames += 1
        return capture_frame(emulator)

    def observe(self, emulator, logger, frame: TickFrame) -> Literal["ok", "lost", "ended"]:
        """Per-frame bookkeeping: battle-end debounce, enemies/threat, play verification, ability."""
        now = frame.captured_at
        if not frame.in_battle:
            return self._lost(emulator, logger, now)
        self.last_in_battle_at = now
        if self.first_in_battle_at is None:
            self.first_in_battle_at = now
        if self.brain is not None and frame.iar is not None:
            frame.enemies = find_enemy_units(frame.iar)
            frame.friends = find_friendly_units(frame.iar)
            if frame.enemies:
                frame.threat = self.brain.should_defend_now(frame.enemies, frame.friends)
            if frame.threat is not None and hasattr(self.brain, "note_threat"):
                self.brain.note_threat(frame.threat, time.time())  # brain clocks are wall time
        if self.pending is not None and now >= self.pending.tap_done_at + PLAY_SETTLE_S:
            self._verify(logger, frame)
        enemy = (bool(frame.enemies) or frame.threat is not None) if self.smart else None
        self.ability.tick(emulator, logger, frame.iar, now, enemy)
        return "ok"

    def slot_blocked(self, slot: int, now: float) -> bool:
        """True while ``slot`` is backing off after a play that did not deploy."""
        return now < self.blocked_until.get(slot, -1e9)

    def repeat_status(self, logger, status: str, now: float) -> None:
        """change_status for a line that repeats every tick: on change or once per REPEAT_STATUS_S."""
        if status == self.last_status and now - self.last_status_at < REPEAT_STATUS_S:
            return
        self.last_status = status
        self.last_status_at = now
        logger.change_status(status)

    def ready_to_decide(self, frame: TickFrame) -> bool:
        """No decision on a frame captured before the last play settled."""
        return self.pending is None and frame.in_battle

    def _lost(self, emulator, logger, now: float) -> Literal["lost", "ended"]:
        lost_for = now - self.last_in_battle_at
        if lost_for >= BATTLE_END_DEBOUNCE_S and now - self.last_end_check >= BATTLE_END_DEBOUNCE_S:
            self.last_end_check = now
            if check_if_battle_has_ended(emulator):
                logger.change_status("Battle ended (confirmed)")
                return "ended"
        if lost_for >= BATTLE_LOST_GIVE_UP_S:
            logger.change_status("Lost battle detection for too long — assuming battle ended")
            return "ended"
        second = int(lost_for)
        if second != self.last_lost_log_second:
            self.last_lost_log_second = second
            logger.change_status(f"Lost battle detection for {second}s — waiting it out")
        time.sleep(LOST_FRAME_SLEEP_S)
        return "lost"

    def _verify(self, logger, frame: TickFrame) -> None:
        pending = self.pending
        self.pending = None
        if pending is None:
            return
        dropped = False
        if pending.elixir_before is not None and frame.elixir is not None:
            dropped = pending.elixir_before - frame.elixir >= max(1, card_cost(pending.card_id) - 1)
        changed = _slot_changed(frame.iar, pending.slot, pending.fingerprint)
        if changed:
            # A card that failed to deploy stays in its slot (raised): same identity = not deployed.
            # The raw hand read counts too, so an UNKNOWN the brain resolved still matches itself.
            same = {pending.raw_card_id} if pending.raw_card_id is not None else set()
            if not _is_unknown(pending.card_id):
                same.add(pending.card_id)
            if same and identify_hand_cards(None, pending.slot) in same:
                changed = False
        deployed = dropped or changed
        if deployed:
            self.deployed += 1
        else:
            self.not_deployed += 1
            now = frame.captured_at
            self.blocked_until[pending.slot] = now + NOT_DEPLOYED_BACKOFF_S
            if now - self.last_not_deployed_log >= REPEAT_STATUS_S:
                self.last_not_deployed_log = now
                logger.log(f"Play not deployed: {pending.card_id} (slot {pending.slot})")
        if self.brain is not None:
            _brain_record(self.brain, pending.play, deployed)

    def log_decision(self, frame: TickFrame, hand, full_hand, threats, play, tick_start: float) -> None:
        """One JSONL line per play or change of hold reason / situation, else at most once a second."""
        if self.decision_log is None:
            return
        now = time.monotonic()
        brain = self.brain
        key = (getattr(brain, "situation", None), getattr(brain, "_hold_reason", None))
        if play is None and key == self.last_decision_key and now - self.last_decision_at < DECISION_LOG_MIN_GAP_S:
            return
        self.last_decision_key = key
        self.last_decision_at = now
        towers = None
        if brain is not None and hasattr(brain, "enemy_tower_hp") and hasattr(brain, "own_tower_hp"):
            towers = {
                "enemy": {lane: brain.enemy_tower_hp(lane) for lane in ("left", "right")},
                "own": {lane: brain.own_tower_hp(lane) for lane in ("left", "right")},
            }
        slots = full_hand if full_hand is not None else hand
        chosen = None
        if play is not None:
            chosen = {
                "card": play.card_id,
                "slot": play.index,
                "coord": list(play.coord),
                "lane": play.lane,
                "defensive": play.defensive,
                "reason": play.reason,
            }
        self.decision_log.write(
            {
                "t": round(time.time(), 3),
                "elapsed": round(self.match_clock(frame.captured_at), 2),
                "elixir": frame.elixir,
                "hand": {str(i): {"card": c, "affordable": i in hand} for i, c in sorted(slots.items())},
                "threats": _units_json(threats),
                "friends": _units_json(frame.friends),
                "threat": _units_json([frame.threat])[0] if frame.threat is not None else None,
                "towers": towers,
                "play": chosen,
                "situation": getattr(brain, "situation", None),
                "frame_age": round(now - frame.captured_at, 3),
                "tick_s": round(now - tick_start, 3),
            }
        )

    def log_end(self, outcome: str) -> None:
        if self.decision_log is None:
            return
        self.decision_log.write(
            {
                "event": "end",
                "outcome": outcome,
                "t": round(time.time(), 3),
                "duration": round(time.monotonic() - self.started_at, 2),
                "frames": self.frames,
                "plays": self.plays,
                "deployed": self.deployed,
                "not_deployed": self.not_deployed,
                "ability_fired": self.ability.fired,
            }
        )


RANDOM_PLAY_ELIXIR_MIN = 3
RANDOM_PLAY_ELIXIR_MAX = 9


def play_random_available_card(
    emulator, logger, recording_flag: bool, elapsed_s: float, frame: TickFrame | None = None
) -> bool:
    """Play one card that is actually available, at a random friendly-half coord.

    Mirrors the rl-bot RandomPlayer so recorded plays are never polluted with cards
    that didn't deploy: read the available hand slots (affordable, non-empty) and, if
    any, play a random one; if none are available, do nothing. Returns True if a card
    was played. The caller gates this behind a random elixir wait. ``frame`` reuses the
    wait's last screenshot instead of taking a new one.
    """
    if frame is not None and frame.in_battle:
        card_indices = list(frame.available)
    else:
        card_indices = check_which_cards_are_available(emulator, check_side=True)
    if not card_indices:
        return False  # nothing playable -> no play, nothing recorded (caller re-rolls)

    card_index = random.choice(card_indices)
    left, top, right, bottom = PLAYABLE_PLAY_REGION_LTRB
    play_coord = (random.randint(left, right), random.randint(top, bottom))

    emulator.click(HAND_CARDS_COORDS[card_index][0], HAND_CARDS_COORDS[card_index][1])
    time.sleep(0.1)
    emulator.click(play_coord[0], play_coord[1])
    time.sleep(0.1)

    if recording_flag:
        # Card identity isn't computed for random plays; home re-derives it from pixels.
        log_play(card_index, play_coord[0], play_coord[1], elapsed_s)
    logger.add_card_played()
    return True


def wait_for_elixir(
    emulator,
    logger,
    elixir_wait_amount,
    WAIT_THRESHOLD=5000,  # noqa: N803
    PLAY_THRESHOLD=10000,  # noqa: N803
    recording_flag: bool = False,
    brain: BattleBrain | None = None,
    ctx: FightContext | None = None,
) -> Literal["restart", "no battle"] | bool:
    """Wait for ``elixir_wait_amount`` elixir during a battle, one screenshot per iteration."""
    result, _ = _wait_for_elixir(
        emulator, logger, elixir_wait_amount, WAIT_THRESHOLD, PLAY_THRESHOLD, recording_flag, brain, ctx
    )
    return result


def _wait_for_elixir(
    emulator,
    logger,
    elixir_wait_amount: int,
    wait_threshold: float,
    play_threshold: float,
    recording_flag: bool,
    brain,
    ctx: FightContext | None,
) -> tuple[Literal["restart", "no battle"] | bool, TickFrame | None]:
    """wait_for_elixir that also returns the frame it stopped on, so the caller can reuse it."""
    if ctx is None:
        ctx = FightContext(brain)
    brain = ctx.brain
    start_time = time.time()
    last_logged_second = -1
    last_threat_log_second = -1

    while True:
        frame = ctx.next_frame(emulator)
        status = ctx.observe(emulator, logger, frame)
        if status == "ended":
            logger.change_status(status="Battle ended — stopping elixir wait")
            return "no battle", frame
        wait_time = time.time() - start_time
        if status == "lost":
            continue

        if frame.elixir is not None and frame.elixir >= elixir_wait_amount:
            logger.change_status(f"Took {str(wait_time)[:4]}s for {elixir_wait_amount} elixir.")
            return True, frame

        elapsed_second = int(wait_time)
        if elapsed_second != last_logged_second:
            logger.change_status(f"Waiting for {elixir_wait_amount} elixir for {elapsed_second}s...")
            last_logged_second = elapsed_second

        card_inhand = len(frame.available)
        if brain is not None and card_inhand:
            iar = frame.iar
            if recording_flag and frame.enemies:
                log_units(iar, classify_units(iar, frame.enemies))
            threat = frame.threat
            if threat is not None:
                budget = brain.defense_budget(threat) if hasattr(brain, "defense_budget") else 0
                if (frame.elixir or 0) >= budget or threat.y >= DEEP_THREAT_Y:
                    logger.change_status(f"Enemy push in the {threat.lane} lane — defending now")
                    return True, frame
                if elapsed_second != last_threat_log_second:
                    logger.change_status(f"Enemy push in the {threat.lane} lane — saving up {budget} elixir")
                    last_threat_log_second = elapsed_second

        if card_detection.bridge_iar is not None and frame.iar is not None:
            action_offset, _ = switch_side()
            if action_offset > play_threshold and card_inhand > 0:
                logger.change_status("Battle too active — playing now")
                return True, frame
            if action_offset > wait_threshold and card_inhand == 4:
                logger.change_status("All cards are available!")
                return True, frame

        if wait_time > ELIXIR_WAIT_TIMEOUT:
            logger.change_status(status="Waited too long for elixir")
            return "restart", frame


def end_fight_state(
    emulator,
    logger: Logger,
    recording_flag,
    disable_win_tracker_toggle=True,
):
    """Method to handle the time after a fight and before the next state"""
    # count the crown score on this end-battle screen

    # get to clash main after this fight
    logger.log("Returning to main menu after fight")
    if get_to_main_after_fight(emulator, logger) is False:
        logger.log("Failed to return to main menu after fight")
        finish_fight_recording(None)
        return False

    logger.log("Returned to main menu after fight")
    time.sleep(3)

    # Determine the outcome. Force the win check when a recording is active so the
    # pack gets a real win/loss; otherwise honor the user's win-tracker toggle.
    if is_recording() or not disable_win_tracker_toggle:
        win_check_return = check_if_previous_game_was_win(emulator, logger)

        if win_check_return == "restart":
            logger.log("Failed while checking if previous game was a win")
            finish_fight_recording(None)
            return False

        outcome = "win" if win_check_return else "loss"

        # Only touch the user's win/loss stats when their tracker is enabled.
        if not disable_win_tracker_toggle:
            if win_check_return:
                logger.add_win()
            else:
                logger.add_loss()

        finish_fight_recording(outcome)
    else:
        logger.log("Not checking win/loss because check is disabled")
        finish_fight_recording(None)

    return True


def check_if_previous_game_was_win(
    emulator,
    logger: Logger,
) -> bool | Literal["restart"]:
    """Method to handle the checking if the previous game was a win or loss"""
    logger.change_status(status="Checking last game result")

    # Use wait_for_clash_main_menu to ensure we are on the main menu.
    if not wait_for_clash_main_menu(emulator, logger, deadspace_click=True):
        logger.change_status(status="Not on main menu — cannot check last game result")
        return "restart"

    # get to clash main options menu
    if get_to_activity_log(emulator, logger, printmode=False) == "restart":
        logger.change_status(status="Failed to open battle log")

        return "restart"

    logger.change_status(status="Checking battle log for win...")
    is_a_win = check_pixels_for_win_in_battle_log(emulator)
    result = "win" if is_a_win else "loss"
    logger.change_status(status=f"Last game result: {result}")

    # close battle log
    logger.change_status(status="Returning to main menu")
    emulator.click(CLOSE_BATTLE_LOG_BUTTON[0], CLOSE_BATTLE_LOG_BUTTON[1])
    if wait_for_clash_main_menu(emulator, logger) is False:
        logger.change_status(status="Timed out returning to main menu after battle log")
        return "restart"
    time.sleep(2)

    return is_a_win


# main fight loops

# Initialize a deque with a maximum length of 3 to store the last three chosen cards
last_three_cards = collections.deque(maxlen=3)


def select_card_index(card_indices, last_three_cards):
    if not card_indices:
        raise ValueError("card_indices cannot be empty")

    # First preference: Cards not in the last_three_cards queue
    preferred_cards = [index for index in card_indices if index not in last_three_cards]

    # Second preference: Cards not among the last two added to the queue
    if not preferred_cards and len(last_three_cards) == 3:
        preferred_cards = [index for index in card_indices if index not in list(last_three_cards)[-2:]]

    # Third preference: Any card except the most recently added one
    if not preferred_cards and last_three_cards:
        preferred_cards = [index for index in card_indices if index != last_three_cards[-1]]

    # Fallback: If all else fails, consider all cards
    if not preferred_cards:
        preferred_cards = card_indices

    return random.choice(preferred_cards)


def play_a_card(emulator, logger, recording_flag: bool, battle_strategy: "BattleStrategy") -> bool:
    print("\n")

    # check which cards are available
    logger.change_status("Looking at which cards are available")
    available_card_check_start_time = time.time()
    card_indices = check_which_cards_are_available(emulator, check_side=True)

    if not card_indices:
        logger.change_status("No cards ready yet...")
        return False

    available_card_check_time_taken = str(
        time.time() - available_card_check_start_time,
    )[:3]

    logger.change_status(
        f"These cards are available: {card_indices} ({available_card_check_time_taken}s)",
    )

    card_index = select_card_index(card_indices, last_three_cards)
    if card_index not in last_three_cards:
        last_three_cards.append(card_index)
    logger.change_status(f"Choosing this card index: {card_index}")

    # get a coord based on the selected side
    play_coord_calculation_start_time = time.time()
    card_id, play_coord = get_play_coords_for_card(emulator, logger, card_index, battle_strategy.get_elapsed_time())
    play_coord_calculation_time_taken = str(
        time.time() - play_coord_calculation_start_time,
    )[:3]

    logger.change_status(
        f"Calculated play for: {card_id} at {play_coord} ({play_coord_calculation_time_taken}s)",
    )

    # click the card index
    click_and_play_card_start_time = time.time()
    if None in [HAND_CARDS_COORDS, card_index]:
        logger.change_status("Non-fatal error: card index is None")
        return False

    emulator.click(HAND_CARDS_COORDS[card_index][0], HAND_CARDS_COORDS[card_index][1])

    # click the play coord
    if play_coord is None:
        logger.change_status("Non-fatal error: play coordinates are None")
        return False

    emulator.click(play_coord[0], play_coord[1])
    click_and_play_card_time_taken = str(time.time() - click_and_play_card_start_time)[:3]
    if recording_flag:
        log_play(card_index, play_coord[0], play_coord[1], battle_strategy.get_elapsed_time())

    logger.change_status(f"Made the play {click_and_play_card_time_taken}s")
    logger.add_card_played()

    if random.randint(0, 9) == 1:
        send_emote(emulator, logger)
    return True


def play_smart_card(
    emulator,
    logger,
    recording_flag: bool,
    battle_strategy: "BattleStrategy",
    brain: BattleBrain,
    frame: TickFrame | None = None,
    ctx: FightContext | None = None,
) -> bool | Literal["hold"]:
    """Read the hand and the board from one frame, then let the battle brain pick the card and spot.

    Returns "hold" when the brain chose to wait (saving elixir) rather than failing to play.
    With ``ctx`` the play is verified on a later frame before the brain records it; without
    it (standalone use) the play is recorded right away, as before.
    """
    tick_start = time.monotonic()
    if frame is None:
        frame = capture_frame(emulator)
        frame.available = _available_slots(frame.iar)
        if frame.iar is not None:
            frame.enemies = find_enemy_units(frame.iar)
            frame.friends = find_friendly_units(frame.iar)
    _update_play_side()
    card_indices = frame.available
    if not card_indices:
        if ctx is not None:
            ctx.repeat_status(logger, "No cards ready yet...", frame.captured_at)
        else:
            logger.change_status("No cards ready yet...")
        return False

    iar = frame.iar
    read_hand, full_hand = _identify_hand(emulator, iar, card_indices)
    hand = read_hand
    if ctx is not None:
        # A slot backing off after a failed deploy stays affordable for the deck tracker but isn't played.
        hand = {i: c for i, c in read_hand.items() if not ctx.slot_blocked(i, frame.captured_at)}
    threats = classify_units(iar, frame.enemies)
    friends = frame.friends
    brain.observe_towers(read_tower_health(iar))
    elapsed = battle_strategy.get_elapsed_time()
    if recording_flag:
        log_units(iar, threats, elapsed)
    clock = ctx.match_clock(frame.captured_at) if ctx is not None else None
    play = _brain_choose(brain, hand, threats, elapsed, frame.elixir, friends, clock, full_hand, set(read_hand))
    if ctx is not None:
        ctx.log_decision(frame, read_hand, full_hand, threats, play, tick_start)
    if play is None:
        if ctx is not None:
            ctx.repeat_status(logger, f"Holding — {brain.situation}", frame.captured_at)
        else:
            logger.change_status(f"Holding — {brain.situation}")
        return "hold"

    logger.change_status(f"Playing {play.card_id} — {play.reason}")
    logger.log(f"Board: {brain.situation}")
    fingerprint = _slot_fingerprint(iar, play.index)
    _tap_card(emulator, HAND_CARDS_COORDS[play.index], play.coord)
    tap_done_at = time.monotonic()
    if recording_flag:
        log_play(play.index, play.coord[0], play.coord[1], elapsed)
    logger.add_card_played()
    if ctx is None:
        brain.record(play)
    else:
        ctx.plays += 1
        ctx.pending = PendingPlay(
            play, play.index, play.card_id, frame.elixir, fingerprint, tap_done_at, read_hand.get(play.index)
        )
    return True


_PHASE_STRATEGIES = {
    "early": [0, 0, 0, 0, 0.3, 0.3, 0.4],  # 0-7s: Conservative, wait for more elixir
    "single": [0.05, 0.05, 0.1, 0.15, 0.15, 0.3, 0.2],  # 7-90s: Balanced distribution
    "double": [0.05, 0.05, 0.1, 0.15, 0.25, 0.3, 0.1],  # 90-200s: Favor 7-8 elixir
    "triple": [0.05, 0.05, 0.1, 0.1, 0.3, 0.4, 0],  # 200s+: Heavy favor 7-8, never 9
}

_PHASE_THRESHOLDS_1V1 = {
    "early": (6000, 9000),
    "single": (6000, 9000),
    "double": (7000, 10000),
    "triple": (8000, 11000),
}

_PHASE_THRESHOLDS_2V2 = {
    "early": (7000, 13000),
    "single": (7000, 14000),
    "double": (8000, 15000),
    "triple": (9000, 16000),
}


class BattleStrategy:
    """Manages battle timing and elixir selection strategy.

    Encapsulates the sophisticated elixir selection logic that changes
    based on battle phase, eliminating the need for global variables.
    """

    def __init__(self, fight_mode: str = "Classic 1v1"):
        self.start_time = None
        self.elixir_amounts = [3, 4, 5, 6, 7, 8, 9]

        is_2v2 = fight_mode == "Classic 2v2"
        self.phase_strategies = _PHASE_STRATEGIES
        self.phase_thresholds = _PHASE_THRESHOLDS_2V2 if is_2v2 else _PHASE_THRESHOLDS_1V1

    def start_battle(self):
        """Call when battle begins to start timing."""
        self.start_time = time.time()

    def get_elapsed_time(self):
        """Get seconds elapsed since battle start."""
        return time.time() - self.start_time if self.start_time else 0

    def get_battle_phase(self):
        """Determine current battle phase based on elapsed time."""
        elapsed = self.get_elapsed_time()
        if elapsed < 7:
            return "early"
        elif elapsed < 90:
            return "single"
        elif elapsed < 200:
            return "double"
        else:
            return "triple"

    def select_elixir_amount(self):
        """Select elixir amount to wait for based on current battle phase."""
        phase = self.get_battle_phase()
        weights = self.phase_strategies[phase]
        return random.choices(self.elixir_amounts, weights=weights, k=1)[0]

    def get_thresholds(self):
        """Get (WAIT_THRESHOLD, PLAY_THRESHOLD) for current battle phase."""
        phase = self.get_battle_phase()
        return self.phase_thresholds[phase]


def _fight_loop(
    emulator,
    logger: Logger,
    recording_flag: bool,
    fight_mode: str = "Classic 1v1",
    custom_path: str | None = None,
    smart_battle: bool = False,
) -> bool:
    """Method for handling dynamically timed fight"""
    create_default_bridge_iar(emulator)
    _maybe_start_fight_recording(emulator, logger, recording_flag, fight_mode, custom_path)
    _configure_hand_crops(recording_flag, custom_path)
    last_three_cards.clear()
    prev_cards_played = logger.get_cards_played()

    # Initialize battle strategy and start timing
    battle_strategy = BattleStrategy(fight_mode)
    battle_strategy.start_battle()
    brain = BattleBrain() if smart_battle else None
    if brain is not None:
        logger.log(f"Smart battle AI on — opening in the {brain.attack_lane} lane")
    ctx = FightContext(brain, DecisionLog() if brain is not None else None)

    while True:
        # Preamble: wait for elixir (one frame per iteration; ability + battle-end handled per frame).
        # The smart brain reads elixir itself and decides when to hold, so it only waits for a cheap card.
        elixir_amount = SMART_MIN_ELIXIR if brain is not None else battle_strategy.select_elixir_amount()
        wait_threshold, play_threshold = battle_strategy.get_thresholds()
        wait_output, frame = _wait_for_elixir(
            emulator, logger, elixir_amount, wait_threshold, play_threshold, recording_flag, brain, ctx
        )

        if wait_output == "restart":
            logger.change_status("Failed while waiting for elixir")
            ctx.log_end("restart")
            return False

        if wait_output == "no battle":
            logger.change_status("Not in battle anymore!")
            break

        if brain is None:
            play_start_time = time.time()
            if play_a_card(emulator, logger, recording_flag, battle_strategy) is False:
                logger.change_status("Failed to play a card, retrying...")
            else:
                logger.change_status(f"Made a play in {str(time.time() - play_start_time)[:4]}s")
            continue

        if _smart_ticks(emulator, logger, recording_flag, battle_strategy, brain, ctx, frame) == "ended":
            break

    ctx.log_end("ended")
    # Fight over: freeze capture so the pack excludes post-fight nav (manifest/outcome written later).
    stop_fight_capture()
    logger.change_status("Fight complete")
    time.sleep(2.13)
    cards_played = logger.get_cards_played()
    logger.change_status(f"Played ~{cards_played - prev_cards_played} cards this fight")

    return True


def _smart_ticks(
    emulator,
    logger,
    recording_flag: bool,
    battle_strategy: "BattleStrategy",
    brain,
    ctx: FightContext,
    frame: TickFrame | None,
) -> Literal["ended", "preamble"]:
    """The smart tick loop: one screenshot per iteration, decide on that frame, hold = next tick.

    Starts on ``frame`` (the preamble's last screenshot) without taking a new one. Returns
    "preamble" after a play or every PREAMBLE_BACKSTOP_S of holding, "ended" when the battle ends.
    """
    entered = time.monotonic()
    while True:
        if frame is None:
            frame = ctx.next_frame(emulator)
            status = ctx.observe(emulator, logger, frame)
            if status == "ended":
                return "ended"
        if ctx.ready_to_decide(frame):
            played = play_smart_card(emulator, logger, recording_flag, battle_strategy, brain, frame, ctx)
            if played is True:
                return "preamble"
        frame = None
        if time.monotonic() - entered >= PREAMBLE_BACKSTOP_S:
            return "preamble"


def _random_fight_loop(
    emulator,
    logger,
    recording_flag: bool = False,
    fight_mode_chosen: str = "Trophy Road",
    custom_path: str | None = None,
) -> bool:
    """Method for handling dynamically timed fight with random plays"""
    logger.change_status(status="Starting battle with random plays")
    _maybe_start_fight_recording(emulator, logger, recording_flag, fight_mode_chosen, custom_path)
    create_default_bridge_iar(emulator)
    fight_timeout = 5 * 60  # 5 minutes
    start_time = time.time()
    ctx = FightContext()

    # while in battle:
    while True:
        if time.time() - start_time > fight_timeout:
            logger.change_status("Random fight loop timed out after 5 minutes")
            return False

        # Clean random-play flow (mirrors rl-bot RandomPlayer): wait for a random
        # elixir gate, then play only if a card is actually available.
        target = random.randint(RANDOM_PLAY_ELIXIR_MIN, RANDOM_PLAY_ELIXIR_MAX)
        elixir_result, frame = _wait_for_elixir(emulator, logger, target, 5000, 10000, recording_flag, None, ctx)
        if elixir_result == "no battle":
            break
        if elixir_result == "restart":
            return False
        play_random_available_card(emulator, logger, recording_flag, time.time() - start_time, frame)

    # Fight over: freeze capture so the pack excludes post-fight nav (manifest/outcome written later).
    stop_fight_capture()
    logger.change_status("Random-plays fight complete")
    return True


if __name__ == "__main__":
    pass
