"""Live "what the bot sees" debug view for smart battles.

``render_debug_frame(iar, info)`` draws one tick's detections on a 2x copy of the battle frame:
enemy / friendly units, the threat lead, the unit scan regions, each hand slot with its card id
and match scores, elixir, tower HP, the Hero ability marker and the chosen play (or the hold
reason), plus a top text strip with tick timing and match phase.

``DebugFrameSink`` publishes those frames from the worker process without ever blocking the
fight loop: ``submit`` copies the frame into a size-1 queue (an older pending frame is dropped)
and a daemon thread renders, JPEG-encodes and writes ``<app data>/debug/latest.jpg`` +
``latest.json`` atomically (temp file + ``os.replace``), throttled to ``max_fps``. The last
``history`` frames are also kept as ``debug/history/NNN.jpg`` (+ ``NNN.json``), overwritten
cyclically, so a bad moment can be reviewed after the fact. The GUI's debug window
(``interface/debug_window.py``) polls ``latest.jpg`` / ``latest.json``.

``info`` is plain JSON (see :func:`build_debug_info`), so the same dict is written to disk.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import cv2
import numpy as np

from pyclashbot.bot import card_detection
from pyclashbot.bot.coords import (
    CHAMPION_ABILITY_DISMISS_COORD,
    ENEMY_TOWER_HP_BARS,
    ENEMY_UNIT_CROP_OFFSETS_LTRB,
    ENEMY_UNIT_SCAN_REGION_LTRB,
    FRIENDLY_UNIT_SCAN_REGION_LTRB,
    HAND_CARDS_COORDS,
    OWN_TOWER_HP_BARS,
)
from pyclashbot.bot.state_detect import ELIXIR_COORDS
from pyclashbot.utils.platform import get_app_data_dir

_log = logging.getLogger(__name__)

SCALE = 2
DEFAULT_MAX_FPS = 5.0
DEFAULT_HISTORY = 200
JPEG_QUALITY = 80
LATEST_JPG = "latest.jpg"
LATEST_JSON = "latest.json"
HISTORY_DIR = "history"

# Overrides the debug folder (None = <app data dir>/debug); tests point it at a tmp dir.
DEBUG_DIR: str | None = None

# BGR colours
_RED = (40, 40, 235)
_DARK_RED = (30, 30, 150)
_BLUE = (235, 150, 40)
_DARK_BLUE = (150, 90, 20)
_YELLOW = (0, 220, 255)
_GREEN = (60, 210, 60)
_GREY = (150, 150, 150)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_MAGENTA = (230, 60, 230)
_ORANGE = (0, 150, 255)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TEXT_SCALE = 0.45
_SMALL_SCALE = 0.38
_LINE_H = 18
_STRIP_LINES = 3


def default_debug_dir() -> str:
    """Where the worker writes debug frames and the GUI reads them."""
    return DEBUG_DIR or os.path.join(get_app_data_dir("py-clash-bot"), "debug")


# ===== info dict ============================================================


def _unit_json(unit: Any) -> dict[str, Any] | None:
    if unit is None:
        return None
    if isinstance(unit, Mapping):
        return {k: unit.get(k) for k in ("x", "y", "size", "kind")}
    return {
        "x": int(unit.x),
        "y": int(unit.y),
        "size": int(getattr(unit, "size", 0)),
        "kind": getattr(unit, "kind", None),
    }


def _hp(value: Any) -> float | None:
    return None if value is None else round(float(value), 3)


def build_debug_info(
    *,
    elixir: int | None = None,
    available: Iterable[int] = (),
    scores: Iterable[tuple[int, str, int, int]] = (),
    hand: Mapping[int, str] | None = None,
    enemies: Iterable[Any] = (),
    friends: Iterable[Any] = (),
    threat: Any = None,
    towers_raw: Any = None,
    brain: Any = None,
    ability_ready: bool | None = None,
    play: Any = None,
    tick_ms: float | None = None,
    frame_age_ms: float | None = None,
    phase: str | None = None,
    elapsed: float | None = None,
    in_battle: bool = True,
) -> dict[str, Any]:
    """Flatten one tick's values (already computed by the fight loop) into JSON for render + disk."""
    available = [int(s) for s in available]
    hand_rows: dict[int, dict[str, Any]] = {}
    for slot, card, best, second in scores:
        hand_rows[int(slot)] = {"slot": int(slot), "card": str(card), "best": int(best), "second": int(second)}
    for slot, card in (hand or {}).items():
        hand_rows.setdefault(int(slot), {"slot": int(slot), "card": str(card), "best": None, "second": None})
    for row in hand_rows.values():
        row["affordable"] = row["slot"] in available

    towers: dict[str, Any] = {"enemy": {}, "own": {}, "enemy_smoothed": {}, "own_smoothed": {}}
    if towers_raw is not None:
        towers["enemy"] = {lane: _hp(v) for lane, v in (getattr(towers_raw, "enemy", None) or {}).items()}
        towers["own"] = {lane: _hp(v) for lane, v in (getattr(towers_raw, "own", None) or {}).items()}
    if brain is not None and hasattr(brain, "enemy_tower_hp") and hasattr(brain, "own_tower_hp"):
        towers["enemy_smoothed"] = {lane: _hp(brain.enemy_tower_hp(lane)) for lane in ("left", "right")}
        towers["own_smoothed"] = {lane: _hp(brain.own_tower_hp(lane)) for lane in ("left", "right")}

    chosen = None
    if play is not None:
        chosen = {
            "card": str(play.card_id),
            "slot": int(play.index),
            "coord": [int(play.coord[0]), int(play.coord[1])],
            "lane": play.lane,
            "defensive": bool(play.defensive),
            "reason": str(play.reason),
        }
    return {
        "t": round(time.time(), 3),
        "in_battle": bool(in_battle),
        "tick_ms": None if tick_ms is None else round(float(tick_ms), 1),
        "frame_age_ms": None if frame_age_ms is None else round(float(frame_age_ms), 1),
        "phase": phase,
        "elapsed": None if elapsed is None else round(float(elapsed), 1),
        "elixir": None if elixir is None else int(elixir),
        "available": available,
        "hand": [hand_rows[s] for s in sorted(hand_rows)],
        "enemies": [_unit_json(u) for u in enemies],
        "friends": [_unit_json(u) for u in friends],
        "threat": _unit_json(threat),
        "towers": towers,
        "ability_ready": None if ability_ready is None else bool(ability_ready),
        "play": chosen,
        "situation": str(getattr(brain, "situation", "") or ""),
        "hold_reason": str(getattr(brain, "_hold_reason", "") or ""),
    }


# ===== rendering ============================================================


def _s(value: float) -> int:
    return round(value * SCALE)


def _pt(x: float, y: float) -> tuple[int, int]:
    return _s(x), _s(y)


_ASCII = str.maketrans({chr(0xB7): "|", chr(0x2014): "-", chr(0x2013): "-", chr(0x2026): "...", chr(0x2192): "->"})


def _ascii(text: str) -> str:
    """cv2's Hershey fonts are ASCII-only; map the brain's separators and drop the rest."""
    return text.translate(_ASCII).encode("ascii", "replace").decode("ascii")


def _text(img: np.ndarray, text: str, org: tuple[int, int], color, scale: float = _TEXT_SCALE) -> None:
    """Outlined text, clamped inside the image."""
    text = _ascii(text)
    (w, h), _ = cv2.getTextSize(text, _FONT, scale, 1)
    x = min(max(2, org[0]), max(2, img.shape[1] - w - 2))
    y = min(max(h + 2, org[1]), img.shape[0] - 3)
    cv2.putText(img, text, (x, y), _FONT, scale, _BLACK, 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), _FONT, scale, color, 1, cv2.LINE_AA)


def _fit(text: str, width: int, scale: float = _TEXT_SCALE) -> str:
    """Trim ``text`` with an ellipsis so it fits ``width`` pixels."""
    text = _ascii(text)
    if cv2.getTextSize(text, _FONT, scale, 1)[0][0] <= width:
        return text
    while text and cv2.getTextSize(text + "...", _FONT, scale, 1)[0][0] > width:
        text = text[:-1]
    return text + "..."


def _rect(img: np.ndarray, ltrb, color, thickness: int = 1) -> None:
    left, top, right, bottom = ltrb
    cv2.rectangle(img, _pt(left, top), _pt(right, bottom), color, thickness)


def _pct(value: Any) -> str:
    return "?" if value is None else f"{round(float(value) * 100)}%"


def _draw_regions(img: np.ndarray) -> None:
    _rect(img, ENEMY_UNIT_SCAN_REGION_LTRB, _DARK_RED, 1)
    _text(img, "enemy scan", _pt(ENEMY_UNIT_SCAN_REGION_LTRB[0] + 2, ENEMY_UNIT_SCAN_REGION_LTRB[1] + 8), _RED, 0.33)
    _rect(img, FRIENDLY_UNIT_SCAN_REGION_LTRB, _DARK_BLUE, 1)
    _text(
        img,
        "friendly scan",
        _pt(FRIENDLY_UNIT_SCAN_REGION_LTRB[0] + 2, FRIENDLY_UNIT_SCAN_REGION_LTRB[1] + 8),
        _BLUE,
        0.33,
    )


def _draw_units(img: np.ndarray, info: Mapping[str, Any]) -> None:
    dl, dt, dr, db = ENEMY_UNIT_CROP_OFFSETS_LTRB
    for unit in info.get("friends") or []:
        x, y, size = unit["x"], unit["y"], unit.get("size") or 0
        cv2.circle(img, _pt(x, y), _s(7), _BLUE, 2)
        _text(img, f"s{size}", _pt(x + 8, y + 3), _BLUE, _SMALL_SCALE)
    for unit in info.get("enemies") or []:
        x, y, size = unit["x"], unit["y"], unit.get("size") or 0
        _rect(img, (x + dl, y + dt, x + dr, y + db), _RED, 1)
        cv2.circle(img, _pt(x, y), _s(3), _RED, -1)
        label = f"{unit['kind']} s{size}" if unit.get("kind") else f"s{size}"
        _text(img, label, _pt(x + dl, y + dt - 2), _RED, _SMALL_SCALE)
    threat = info.get("threat")
    if threat:
        x, y = threat["x"], threat["y"]
        _rect(img, (x + dl - 2, y + dt - 2, x + dr + 2, y + db + 2), _YELLOW, 4)
        _text(img, "THREAT", _pt(x + dl, y + db + 10), _YELLOW)


def _draw_hand(img: np.ndarray, info: Mapping[str, Any]) -> None:
    w, h = card_detection.TOTAL_WIDTH, card_detection.TOTAL_HEIGHT
    for row in info.get("hand") or []:
        slot = row["slot"]
        if not 0 <= slot < len(card_detection.toplefts):
            continue
        x, y = card_detection.toplefts[slot]
        card = str(row.get("card") or "UNKNOWN")
        if card.upper().startswith("UNKNOWN"):
            color = _RED
        elif row.get("affordable"):
            color = _GREEN
        else:
            color = _GREY
        _rect(img, (x, y, x + w, y + h), color, 3)
        _text(img, _fit(card, _s(w) + 10, _SMALL_SCALE), _pt(x, y - 3), color, _SMALL_SCALE)
        if row.get("best") is not None:
            _text(img, f"{row['best']}/{row['second']}", _pt(x + 2, y + h - 3), color, _SMALL_SCALE)


def _draw_hud(img: np.ndarray, info: Mapping[str, Any]) -> None:
    elixir = info.get("elixir")
    last_y, last_x = ELIXIR_COORDS[-1]
    _text(img, f"E {'?' if elixir is None else elixir}", _pt(last_x + 10, last_y + 4), _MAGENTA, 0.55)

    towers = info.get("towers") or {}
    for side, bars in (("enemy", ENEMY_TOWER_HP_BARS), ("own", OWN_TOWER_HP_BARS)):
        raw = towers.get(side) or {}
        smooth = towers.get(f"{side}_smoothed") or {}
        color = _RED if side == "enemy" else _BLUE
        for lane, (x_left, x_right, y) in bars.items():
            cv2.line(img, _pt(x_left, y + 2), _pt(x_right, y + 2), color, 1)
            label = _pct(raw.get(lane))
            if smooth.get(lane) is not None:
                label += f" ({_pct(smooth.get(lane))})"
            _text(img, label, _pt(x_left, y + 14), color, _SMALL_SCALE)

    if info.get("ability_ready"):
        ax, ay = CHAMPION_ABILITY_DISMISS_COORD
        cv2.circle(img, _pt(ax, ay), _s(14), _YELLOW, 3)
        _text(img, "ABILITY READY", _pt(ax - 45, ay - 18), _YELLOW)


def _draw_play(img: np.ndarray, info: Mapping[str, Any]) -> None:
    play = info.get("play")
    if not play:
        return
    x, y = play["coord"]
    color = _ORANGE if play.get("defensive") else _GREEN
    slot = play.get("slot")
    if isinstance(slot, int) and 0 <= slot < len(HAND_CARDS_COORDS):
        cv2.line(img, _pt(*HAND_CARDS_COORDS[slot]), _pt(x, y), color, 1, cv2.LINE_AA)
    cx, cy = _pt(x, y)
    arm = _s(12)
    cv2.line(img, (cx - arm, cy), (cx + arm, cy), color, 3)
    cv2.line(img, (cx, cy - arm), (cx, cy + arm), color, 3)
    cv2.circle(img, (cx, cy), _s(8), color, 2)
    _text(img, _fit(f"{play['card']}: {play.get('reason', '')}", img.shape[1] - 8), (cx + arm + 4, cy - 6), color)


def _draw_strip(img: np.ndarray, info: Mapping[str, Any]) -> None:
    height = _LINE_H * _STRIP_LINES + 8
    band = img[:height]
    band[:] = (band.astype(np.float32) * 0.25).astype(np.uint8)  # darken, keep a hint of the arena
    width = img.shape[1] - 12

    def num(key: str, fmt: str) -> str:
        value = info.get(key)
        return "?" if value is None else fmt.format(value)

    line1 = (
        f"tick {num('tick_ms', '{:.0f}')}ms | age {num('frame_age_ms', '{:.0f}')}ms | "
        f"{info.get('phase') or '?'} {num('elapsed', '{:.1f}')}s | elixir {num('elixir', '{}')}"
    )
    play = info.get("play")
    situation = info.get("situation") or ""
    if play:
        kind = "DEFEND" if play.get("defensive") else "PLAY"
        line2 = f"{kind} {play['card']} slot {play.get('slot')} @ {tuple(play['coord'])} {play.get('lane') or ''}"
        line3 = situation
        color2 = _ORANGE if play.get("defensive") else _GREEN
    else:
        line2 = f"HOLD: {situation or 'no cards ready'}"
        line3 = info.get("hold_reason") or ""
        color2 = _YELLOW
    _text(img, _fit(line1, width), (6, _LINE_H), _WHITE)
    _text(img, _fit(line2, width), (6, _LINE_H * 2), color2)
    if line3:
        _text(img, _fit(line3, width), (6, _LINE_H * 3), _GREY)


def render_debug_frame(iar: np.ndarray, info: Mapping[str, Any]) -> np.ndarray:
    """Annotated 2x copy of a BGR battle frame (``iar`` is never modified). Pure: no I/O."""
    img = cv2.resize(iar, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    _draw_regions(img)
    _draw_units(img, info)
    _draw_hand(img, info)
    _draw_hud(img, info)
    _draw_play(img, info)
    _draw_strip(img, info)
    return img


# ===== publishing ===========================================================


def _write_atomic(path: str, data: bytes) -> None:
    """Write ``data`` to a temp file next to ``path`` then rename over it (readers never see half a file).

    On Windows the rename fails while a reader holds the target open; retry briefly, then skip.
    """
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    for attempt in range(3):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 2:
                break
            time.sleep(0.01)
    try:
        os.remove(tmp)
    except OSError:
        pass


class DebugFrameSink:
    """Throttled, non-blocking writer of annotated frames. ``submit`` never raises or blocks."""

    def __init__(
        self,
        root: str | None = None,
        *,
        max_fps: float = DEFAULT_MAX_FPS,
        history: int = DEFAULT_HISTORY,
        clock: Callable[[], float] = time.monotonic,
        frame_queue: queue.Queue | None = None,
        start_thread: bool = True,
    ) -> None:
        self.root = root or default_debug_dir()
        self.min_interval = 1.0 / max_fps if max_fps > 0 else 0.0
        self.history = max(1, int(history))
        self.clock = clock
        self.queue: queue.Queue = frame_queue if frame_queue is not None else queue.Queue(maxsize=1)
        self.last_submit = -1e9
        self.seq = 0  # frames written
        self.submitted = 0
        self.dropped = 0
        self.errors = 0
        self._error_logged = False
        self._thread: threading.Thread | None = None
        if start_thread:
            self._thread = threading.Thread(target=self._run, name="debug-frame-sink", daemon=True)
            self._thread.start()

    def _log_error_once(self, exc: BaseException) -> None:
        self.errors += 1
        if not self._error_logged:
            self._error_logged = True
            _log.warning("Debug view: could not publish a frame (%s); further errors are silent", exc)

    def submit(self, iar: np.ndarray | None, info: Mapping[str, Any]) -> bool:
        """Queue one frame for the writer thread. False when throttled, frameless or on error."""
        try:
            if iar is None:
                return False
            now = self.clock()
            if now - self.last_submit < self.min_interval:
                return False
            self.last_submit = now
            item = (np.array(iar, copy=True), dict(info))
            try:
                self.queue.put_nowait(item)
            except queue.Full:
                try:
                    self.queue.get_nowait()  # drop the stale pending frame
                    self.dropped += 1
                except queue.Empty:
                    pass
                try:
                    self.queue.put_nowait(item)
                except queue.Full:
                    self.dropped += 1
                    return False
            self.submitted += 1
            return True
        except Exception as exc:  # the fight loop must never see a debug-view error
            self._log_error_once(exc)
            return False

    def process_pending(self) -> int:
        """Write every queued frame on the calling thread (tests / no-thread mode). Returns frames written."""
        written = 0
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                return written
            if item is not None and self._write(*item):
                written += 1

    def _run(self) -> None:
        while True:
            item = self.queue.get()
            if item is None:
                return
            self._write(*item)

    def _write(self, iar: np.ndarray, info: dict[str, Any]) -> bool:
        try:
            img = render_debug_frame(iar, info)
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                raise ValueError("JPEG encode failed")
            jpg = buf.tobytes()
            index = self.seq % self.history
            meta = dict(info, seq=self.seq, history_index=index)
            text = json.dumps(meta, default=str).encode("utf-8")
            history_dir = os.path.join(self.root, HISTORY_DIR)
            os.makedirs(history_dir, exist_ok=True)
            _write_atomic(os.path.join(history_dir, f"{index:03d}.jpg"), jpg)
            _write_atomic(os.path.join(history_dir, f"{index:03d}.json"), text)
            _write_atomic(os.path.join(self.root, LATEST_JPG), jpg)
            _write_atomic(os.path.join(self.root, LATEST_JSON), text)
            self.seq += 1
            return True
        except Exception as exc:
            self._log_error_once(exc)
            return False

    def close(self) -> None:
        """Stop the writer thread (pending frame, if any, is still written first)."""
        if self._thread is None:
            return
        try:
            self.queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)
        self._thread = None


_sink: DebugFrameSink | None = None
_sink_lock = threading.Lock()


def get_sink() -> DebugFrameSink:
    """The worker process's shared sink (one writer thread, reused across fights)."""
    global _sink
    with _sink_lock:
        root = default_debug_dir()
        if _sink is None or _sink.root != root:
            if _sink is not None:
                _sink.close()
            _sink = DebugFrameSink(root)
        return _sink
