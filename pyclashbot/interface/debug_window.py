"""Debug view: polls the worker's ``debug/latest.jpg`` + ``latest.json`` (see ``bot/debug_view.py``).

The bot runs in another process, so the GUI only reads files. Missing, half-written or
undecodable files are skipped and the last good frame stays on screen. ``DebugPanel`` is the
reusable polling image + info frame: the Battle AI page embeds one and the ``DebugViewWindow``
popout hosts another. ``read_last_battle`` summarises the newest decision log's end event.
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
from pyclashbot.utils.platform import get_log_dir

POLL_MS = 200
# Frames older than this are flagged as stale (bot between battles, stopped, or debug view off).
STALE_S = 3.0
_SCREEN_MARGIN_PX = 160
# A hidden panel (page not shown, window minimised) re-checks at this slower rate without reading files.
HIDDEN_POLL_MS = 1000
# Decision logs: how many of the newest files to scan for an end event, and how much of each tail.
_LAST_BATTLE_FILES = 3
_TAIL_BYTES = 32 * 1024
_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600
_SECONDS_PER_DAY = 86400


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


# ---- last battle summary (decision logs) ------------------------------------------


def default_decisions_dir() -> str:
    """Where ``bot/fight.py`` ``DecisionLog`` writes one JSONL file per fight."""
    return os.path.join(get_log_dir("py-clash-bot"), "decisions")


def _last_end_event(path: str) -> dict[str, Any] | None:
    """The last ``{"event": "end", ...}`` line in the tail of one decision log, or None."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - _TAIL_BYTES))
            raw = f.read()
    except OSError:
        return None
    for line in reversed(raw.decode("utf-8", errors="replace").splitlines()):
        if '"end"' not in line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue  # the first line of a tail read is usually cut mid-way
        if isinstance(entry, dict) and entry.get("event") == "end":
            return entry
    return None


def read_last_battle(directory: str | None = None) -> dict[str, Any] | None:
    """Newest fight's end event plus ``file`` and ``newer_unfinished`` keys; None when there is none.

    Logs are named by start time, so name order is age order. When the newest log has no end event
    yet (a fight in progress, or one that crashed) the previous finished fight is returned with
    ``newer_unfinished`` set. Missing folders and malformed lines are skipped, never raised.
    """
    directory = directory or default_decisions_dir()
    try:
        names = sorted((name for name in os.listdir(directory) if name.endswith(".jsonl")), reverse=True)
    except OSError:
        return None
    for index, name in enumerate(names[:_LAST_BATTLE_FILES]):
        end = _last_end_event(os.path.join(directory, name))
        if end is not None:
            return {**end, "file": name, "newer_unfinished": index > 0}
    return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ago(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < _SECONDS_PER_MINUTE:
        return "just now"
    if seconds < _SECONDS_PER_HOUR:
        return f"{int(seconds // _SECONDS_PER_MINUTE)} min ago"
    if seconds < _SECONDS_PER_DAY:
        return f"{int(seconds // _SECONDS_PER_HOUR)} h ago"
    return f"{int(seconds // _SECONDS_PER_DAY)} d ago"


def _duration(value: Any) -> str:
    try:
        total = round(float(value))
    except (TypeError, ValueError):
        return "?"
    minutes, seconds = divmod(max(0, total), _SECONDS_PER_MINUTE)
    return f"{minutes}:{seconds:02d}"


_OUTCOME_LABELS = {"ended": "Finished", "restart": "Restarted (recovery)"}
# Row labels of the Last battle card, in display order (format_last_battle returns these).
LAST_BATTLE_LABELS = ("Outcome", "When", "Duration", "Plays", "Deployed", "Not deployed", "Ability fired")


def format_last_battle(summary: dict[str, Any] | None, now: float | None = None) -> list[tuple[str, str]]:
    """(label, value) rows for the Last battle card; empty when there is no summary."""
    if not summary:
        return []
    now = time.time() if now is None else now
    outcome = str(summary.get("outcome") or "?")
    plays = _as_int(summary.get("plays"))
    deployed = _as_int(summary.get("deployed"))
    deployed_text = "?" if deployed is None else str(deployed)
    if deployed is not None and plays:
        deployed_text = f"{deployed} ({round(deployed / plays * 100)}%)"
    stamp = summary.get("t")
    try:
        when = _ago(now - float(stamp)) if stamp is not None else "?"
    except (TypeError, ValueError):
        when = "?"

    def num(key: str) -> str:
        value = _as_int(summary.get(key))
        return "?" if value is None else str(value)

    values = (
        _OUTCOME_LABELS.get(outcome, outcome.title()),
        when,
        _duration(summary.get("duration")),
        num("plays"),
        deployed_text,
        num("not_deployed"),
        num("ability_fired"),
    )
    return list(zip(LAST_BATTLE_LABELS, values, strict=True))


# ---- Tk widgets ---------------------------------------------------------------------

_PANEL_BG = "#111111"
_PANEL_FG = "#dddddd"
_TEXT_BG = "#1b1b1b"
_TEXT_FG = "#e0e0e0"
_FOOTER_FG = "#888888"
_PLACEHOLDER_SIZE_CHARS = (34, 16)  # placeholder label (width, height) in characters before the first frame


class DebugPanel(tk.Frame):
    """Latest annotated frame + its key fields, re-read every ``poll_ms`` while the panel is visible."""

    def __init__(
        self,
        master: tk.Misc,
        directory: str | None = None,
        poll_ms: int = POLL_MS,
        max_height: int | None = None,
        text_width: int = 46,
        show_footer: bool = True,
    ) -> None:
        super().__init__(master, bg=_PANEL_BG)
        self.directory = directory or default_debug_dir()
        self.poll_ms = poll_ms
        self._max_h = max_height or max(400, self.winfo_screenheight() - _SCREEN_MARGIN_PX)
        self._photo: ImageTk.PhotoImage | None = None
        self._last_stamp: tuple[int, int] | None = None
        self._after_id: str | None = None
        self._last_text: str | None = None

        width, height = _PLACEHOLDER_SIZE_CHARS
        self.image_label = tk.Label(
            self, text="Waiting for frames...", bg=_PANEL_BG, fg=_PANEL_FG, width=width, height=height
        )
        self.image_label.grid(row=0, column=0, sticky="nsew")
        self.text = tk.Text(
            self,
            width=text_width,
            wrap="none",
            font=("Consolas", 9),
            bg=_TEXT_BG,
            fg=_TEXT_FG,
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            padx=8,
            pady=6,
        )
        self.text.grid(row=0, column=1, sticky="nsew")
        self.footer: tk.Label | None = None
        if show_footer:
            self.footer = tk.Label(self, text=self.directory, anchor="w", bg=_PANEL_BG, fg=_FOOTER_FG)
            self.footer.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._set_text(format_info(None))
        self._after_id = self.after_idle(self._poll)

    def alive(self) -> bool:
        try:
            return bool(self.winfo_exists())
        except tk.TclError:
            return False

    def stop(self) -> None:
        """Cancel the poll loop (the widget itself stays)."""
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def destroy(self) -> None:
        self.stop()
        super().destroy()

    def poll_once(self) -> None:
        """Re-read the debug folder now. Never raises: a bad file keeps the previous frame."""
        try:
            stamp = self._jpg_stamp()
            if stamp is not None and stamp != self._last_stamp:
                jpg = read_jpg(self.directory)
                if jpg and self._show_image(jpg):
                    self._last_stamp = stamp  # a half-written frame is retried on the next poll
            self._set_text(format_info(read_info(self.directory)))
        except Exception:
            pass  # never let a bad file kill the poll loop

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
        # With an image, width/height are pixels; 0 lets the label take the image's own size.
        self.image_label.configure(image=self._photo, text="", width=0, height=0)
        return True

    def _poll(self) -> None:
        self._after_id = None
        if not self.alive():
            return
        try:
            visible = bool(self.winfo_ismapped())
        except tk.TclError:
            return
        delay = HIDDEN_POLL_MS
        if visible:
            self.poll_once()
            delay = self.poll_ms
        self._after_id = self.after(delay, self._poll)

    def _set_text(self, text: str) -> None:
        if text == self._last_text:
            return
        self._last_text = text
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("end", text)
        self.text.configure(state="disabled")


class DebugViewWindow:
    """Popout Toplevel hosting a full-size ``DebugPanel``."""

    def __init__(self, master: tk.Misc, directory: str | None = None, poll_ms: int = POLL_MS) -> None:
        self.top = tk.Toplevel(master)
        self.top.title("py-clash-bot — debug view")
        self.top.protocol("WM_DELETE_WINDOW", self.close)
        self.panel = DebugPanel(self.top, directory=directory, poll_ms=poll_ms)
        self.panel.pack(fill="both", expand=True)
        self.directory = self.panel.directory
        self.poll_ms = poll_ms
        self.image_label = self.panel.image_label
        self.text = self.panel.text
        self.footer = self.panel.footer

    def alive(self) -> bool:
        try:
            return bool(self.top.winfo_exists())
        except tk.TclError:
            return False

    def lift(self) -> None:
        self.top.deiconify()
        self.top.lift()

    def close(self) -> None:
        self.panel.stop()
        try:
            self.top.destroy()
        except tk.TclError:
            pass
