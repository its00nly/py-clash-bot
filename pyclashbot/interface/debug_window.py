"""Debug view window: polls the worker's ``debug/latest.jpg`` + ``latest.json`` (see ``bot/debug_view.py``).

The bot runs in another process, so the window only reads files. Missing, half-written or
undecodable files are skipped and the last good frame stays on screen.
"""

from __future__ import annotations

import io
import json
import os
import time
import tkinter as tk
from typing import Any

from PIL import Image, ImageTk

from pyclashbot.bot.debug_view import LATEST_JPG, LATEST_JSON, default_debug_dir

POLL_MS = 200
# Frames older than this are flagged as stale (bot between battles, stopped, or debug view off).
STALE_S = 3.0
_SCREEN_MARGIN_PX = 160


def read_jpg(directory: str) -> bytes | None:
    """Bytes of ``latest.jpg``, or None when missing / unreadable."""
    try:
        with open(os.path.join(directory, LATEST_JPG), "rb") as f:
            return f.read() or None
    except OSError:
        return None


def read_info(directory: str) -> dict[str, Any] | None:
    """``latest.json`` as a dict, or None when missing, half-written or not a JSON object."""
    try:
        with open(os.path.join(directory, LATEST_JSON), encoding="utf-8") as f:
            loaded = json.load(f)
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def read_latest(directory: str) -> tuple[bytes | None, dict[str, Any] | None]:
    """(JPEG bytes, info dict) from the debug folder; either is None when missing or unreadable."""
    return read_jpg(directory), read_info(directory)


def _pct(value: Any) -> str:
    return "?" if value is None else f"{round(float(value) * 100)}%"


def format_info(info: dict[str, Any] | None, now: float | None = None) -> str:
    """Key fields of one tick's info as plain text for the side panel."""
    if not info:
        return "Waiting for frames...\n\nStart a smart battle with Debug view on."
    now = time.time() if now is None else now
    lines: list[str] = []
    age = now - float(info.get("t") or now)
    lines.append(f"Updated {age:.1f}s ago" + ("  (stale)" if age > STALE_S else ""))
    lines.append(f"Phase: {info.get('phase') or '?'}  {info.get('elapsed', '?')}s")
    lines.append(f"Tick: {info.get('tick_ms', '?')} ms   frame age: {info.get('frame_age_ms', '?')} ms")
    lines.append(f"Elixir: {info.get('elixir', '?')}")
    lines.append(f"Ability ready: {'yes' if info.get('ability_ready') else 'no'}")
    lines.append("")
    play = info.get("play")
    if play:
        kind = "Defend" if play.get("defensive") else "Play"
        lines.append(f"{kind}: {play.get('card')} (slot {play.get('slot')}) at {tuple(play.get('coord') or ())}")
        lines.append(f"  lane {play.get('lane')} — {play.get('reason')}")
    else:
        lines.append("Hold")
        if info.get("hold_reason"):
            lines.append(f"  {info['hold_reason']}")
    if info.get("situation"):
        lines.append(f"Board: {info['situation']}")
    lines.append("")
    lines.append("Hand (best/second):")
    for row in info.get("hand") or []:
        scores = f"{row.get('best')}/{row.get('second')}" if row.get("best") is not None else "-"
        mark = "ready" if row.get("affordable") else "     "
        lines.append(f"  {row.get('slot')}: {row.get('card'):<18} {scores:>11} {mark}")
    lines.append("")
    enemies = info.get("enemies") or []
    friends = info.get("friends") or []
    lines.append(f"Enemies: {len(enemies)}   friends: {len(friends)}")
    threat = info.get("threat")
    if threat:
        kind = f" {threat['kind']}" if threat.get("kind") else ""
        lines.append(f"Threat:{kind} at ({threat.get('x')}, {threat.get('y')}) size {threat.get('size')}")
    for unit in enemies[:8]:
        kind = unit.get("kind") or "?"
        lines.append(f"  enemy {kind} ({unit.get('x')}, {unit.get('y')}) s{unit.get('size')}")
    towers = info.get("towers") or {}
    for side in ("enemy", "own"):
        raw = towers.get(side) or {}
        if raw:
            lines.append(f"{side.title()} towers: L {_pct(raw.get('left'))}  R {_pct(raw.get('right'))}")
    return "\n".join(lines)


class DebugViewWindow:
    """Toplevel that shows the latest annotated frame and its key fields, refreshed every POLL_MS."""

    def __init__(self, master: tk.Misc, directory: str | None = None, poll_ms: int = POLL_MS) -> None:
        self.directory = directory or default_debug_dir()
        self.poll_ms = poll_ms
        self.top = tk.Toplevel(master)
        self.top.title("py-clash-bot — debug view")
        self.top.protocol("WM_DELETE_WINDOW", self.close)
        self._photo: ImageTk.PhotoImage | None = None
        self._last_stamp: tuple[int, int] | None = None
        self._after_id: str | None = None
        self._max_h = max(400, self.top.winfo_screenheight() - _SCREEN_MARGIN_PX)

        self.image_label = tk.Label(self.top, text="Waiting for frames...", bg="#111111", fg="#dddddd")
        self.image_label.grid(row=0, column=0, sticky="nsew")
        self.text = tk.Text(self.top, width=46, wrap="none", font=("Consolas", 9), bg="#1b1b1b", fg="#e0e0e0")
        self.text.grid(row=0, column=1, sticky="ns")
        self.footer = tk.Label(self.top, text=self.directory, anchor="w", fg="#888888")
        self.footer.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.top.columnconfigure(0, weight=1)
        self.top.rowconfigure(0, weight=1)
        self._poll()

    def alive(self) -> bool:
        try:
            return bool(self.top.winfo_exists())
        except tk.TclError:
            return False

    def lift(self) -> None:
        self.top.deiconify()
        self.top.lift()

    def close(self) -> None:
        if self._after_id is not None:
            try:
                self.top.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None
        try:
            self.top.destroy()
        except tk.TclError:
            pass

    def _jpg_stamp(self) -> tuple[int, int] | None:
        try:
            st = os.stat(os.path.join(self.directory, LATEST_JPG))
        except OSError:
            return None
        return st.st_mtime_ns, st.st_size

    def _show_image(self, jpg: bytes) -> bool:
        try:
            image = Image.open(io.BytesIO(jpg))
            image.load()  # a partial file fails here and the previous frame stays up
        except Exception:
            return False
        if image.height > self._max_h:
            ratio = self._max_h / image.height
            image = image.resize((max(1, int(image.width * ratio)), self._max_h), Image.Resampling.BILINEAR)
        self._photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self._photo, text="")
        return True

    def _poll(self) -> None:
        self._after_id = None
        if not self.alive():
            return
        try:
            stamp = self._jpg_stamp()
            if stamp is not None and stamp != self._last_stamp:
                jpg = read_jpg(self.directory)
                if jpg and self._show_image(jpg):
                    self._last_stamp = stamp  # a half-written frame is retried on the next poll
            self._set_text(format_info(read_info(self.directory)))
        except Exception:
            pass  # never let a bad file kill the poll loop
        self._after_id = self.top.after(self.poll_ms, self._poll)

    def _set_text(self, text: str) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("end", text)
        self.text.configure(state="disabled")
