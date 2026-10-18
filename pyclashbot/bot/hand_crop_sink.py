"""Capture hand-card crops the matcher could not identify, so they can be labelled into fingerprints.

``card_detection.identify_hand_cards_scored`` calls :func:`save` for every UNKNOWN slot. The
caller thread only copies the 54x66 crop and enqueues it; a daemon thread writes the PNG plus
one JSON line (slot, time, 4x16 corner histograms, top-3 candidates) to
``get_recordings_dir()/hand_crops/``. Disk use is bounded: at most ``per_cluster`` crops per
rounded-fingerprint cluster, a global byte cap, and the recorder's free-disk fraction.

Labelling workflow: write ``labels.json`` next to the crops, mapping crop file name -> card id
(e.g. ``{"crop_....png": "furnace_evo_charging"}``), then run::

    python -m pyclashbot.bot.hand_crop_sink build path/to/labels.json

which prints ``card_color_data`` entries (891 pixels per corner) to paste into card_detection.py.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

CROP_W, CROP_H = 54, 66
INDEX_NAME = "index.jsonl"
DEFAULT_PER_CLUSTER = 5
DEFAULT_MAX_BYTES = 50 * 1024 * 1024
# The fight loop identifies the hand every tick (~2-3/s); keep at most one crop per slot per interval.
DEFAULT_MIN_INTERVAL_S = 3.0
# Histogram bins are rounded to this many pixels to group crops of the same art.
CLUSTER_BIN = 150
_FALLBACK_MIN_FREE_DISK_FRACTION = 0.10


def default_root(custom_path: str | None = None) -> Path:
    from pyclashbot.utils.platform import get_recordings_dir  # lazy, avoids import cycles

    return Path(get_recordings_dir(custom_path=custom_path)) / "hand_crops"


def _min_free_fraction() -> float:
    try:
        from pyclashbot.bot.recorder import MIN_FREE_DISK_FRACTION  # lazy (cv2/recorder deps)
    except Exception:  # the sink must work without the recorder
        return _FALLBACK_MIN_FREE_DISK_FRACTION
    return MIN_FREE_DISK_FRACTION


def cluster_key(histograms: Any) -> tuple[int, ...]:
    """Coarse identity of a crop: its 4x16 corner histograms rounded to CLUSTER_BIN pixels."""
    arr = np.asarray(histograms, dtype=np.int64).reshape(-1)
    return tuple(int(v) for v in np.rint(arr / CLUSTER_BIN))


class HandCropSink:
    """Bounded, non-blocking writer for UNKNOWN hand crops."""

    def __init__(
        self,
        root: str | os.PathLike[str] | None = None,
        per_cluster: int = DEFAULT_PER_CLUSTER,
        max_bytes: int = DEFAULT_MAX_BYTES,
        min_free_fraction: float | None = None,
        work_queue: queue.Queue | None = None,
        start: bool = True,
        min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
    ) -> None:
        self._root = Path(root) if root is not None else None
        self.per_cluster = per_cluster
        self.max_bytes = max_bytes
        self.min_free_fraction = min_free_fraction
        self.queue: queue.Queue = work_queue if work_queue is not None else queue.Queue(maxsize=64)
        self._counts: dict[tuple[int, ...], int] = {}
        self._queued: dict[tuple[int, ...], int] = {}
        self.min_interval_s = min_interval_s
        self._last_slot_save: dict[int, float] = {}
        self._seq = 0
        self._bytes = 0
        self._loaded = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        if start:
            self.start()

    @property
    def root(self) -> Path:
        if self._root is None:
            self._root = default_root()
        return self._root

    # ---- caller thread: no I/O -------------------------------------------------
    def save(self, iar, slot: int, topleft, histograms, top3) -> bool:
        """Queue one crop. Returns False when it is dropped (throttled, cap reached or queue full)."""
        now = time.monotonic()
        last = self._last_slot_save.get(int(slot))
        if last is not None and now - last < self.min_interval_s:
            return False
        with self._lock:
            if self._bytes >= self.max_bytes:
                return False
            key = cluster_key(histograms)
            if self._counts.get(key, 0) + self._queued.get(key, 0) >= self.per_cluster:
                return False
            self._queued[key] = self._queued.get(key, 0) + 1
            self._last_slot_save[int(slot)] = now
        x, y = int(topleft[0]), int(topleft[1])
        item = {
            "crop": np.array(iar[y : y + CROP_H, x : x + CROP_W], copy=True),
            "slot": int(slot),
            "time": time.time(),
            "topleft": [x, y],
            "histograms": np.asarray(histograms, dtype=np.int64).reshape(4, -1).tolist(),
            "top3": [[str(name), int(dist)] for name, dist in top3],
        }
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            self._dequeued(key)
            return False
        return True

    def _dequeued(self, key: tuple[int, ...]) -> None:
        with self._lock:
            left = self._queued.get(key, 0) - 1
            if left > 0:
                self._queued[key] = left
            else:
                self._queued.pop(key, None)

    # ---- worker thread ---------------------------------------------------------
    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name="hand-crop-sink", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                self.process(item)
            except Exception:  # capture is best effort, never kill the worker
                pass
            finally:
                self.queue.task_done()

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until queued crops are written (tests / shutdown)."""
        deadline = time.monotonic() + timeout
        while self.queue.unfinished_tasks:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.01)
        return True

    def close(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self.queue.put(None)
            self._thread.join(timeout=5)

    def _load_existing(self) -> None:
        """Seed caps from crops written by earlier sessions."""
        if self._loaded:
            return
        self._loaded = True
        root = self.root
        if not root.is_dir():
            return
        total = 0
        for entry in os.scandir(root):
            if entry.is_file():
                total += entry.stat().st_size
        counts: dict[tuple[int, ...], int] = {}
        for rec in read_index(root):
            key = cluster_key(rec["histograms"])
            counts[key] = counts.get(key, 0) + 1
        with self._lock:
            self._bytes = max(self._bytes, total)
            for k, v in counts.items():
                self._counts[k] = max(self._counts.get(k, 0), v)

    def _enough_disk(self) -> bool:
        fraction = self.min_free_fraction if self.min_free_fraction is not None else _min_free_fraction()
        try:
            usage = shutil.disk_usage(self.root)
        except OSError:
            return False
        return usage.free / usage.total >= fraction

    def process(self, item: dict) -> bool:
        """Write one queued crop (worker thread). Returns True when it was written."""
        key = cluster_key(item["histograms"])
        try:
            return self._write(item, key)
        finally:
            self._dequeued(key)

    def _write(self, item: dict, key: tuple[int, ...]) -> bool:
        import cv2  # keep module import light

        root = self.root
        root.mkdir(parents=True, exist_ok=True)
        self._load_existing()
        ok, png = cv2.imencode(".png", item["crop"])
        if not ok:
            return False
        data = png.tobytes()
        with self._lock:
            if self._counts.get(key, 0) >= self.per_cluster or self._bytes + len(data) > self.max_bytes:
                return False
        if not self._enough_disk():
            return False
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(item["time"]))
        self._seq += 1
        name = f"crop_{stamp}_{int(item['time'] * 1000) % 1000:03d}_{self._seq:04d}_s{item['slot']}.png"
        while (root / name).exists():
            self._seq += 1
            name = f"crop_{stamp}_{int(item['time'] * 1000) % 1000:03d}_{self._seq:04d}_s{item['slot']}.png"
        (root / name).write_bytes(data)
        record = {k: v for k, v in item.items() if k != "crop"}
        record["file"] = name
        line = json.dumps(record) + "\n"
        with open(root / INDEX_NAME, "a", encoding="utf-8") as fh:
            fh.write(line)
        with self._lock:
            self._counts[key] = self._counts.get(key, 0) + 1
            self._bytes += len(data) + len(line.encode("utf-8"))
        return True


def read_index(root: str | os.PathLike[str]) -> list[dict]:
    path = Path(root) / INDEX_NAME
    if not path.is_file():
        return []
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# ---- module-level default sink ---------------------------------------------------
_default: HandCropSink | None = None
_default_lock = threading.Lock()
_enabled = True


def configure(root: str | os.PathLike[str] | None = None, enabled: bool = True, **kwargs: Any) -> None:
    """Replace the default sink (e.g. with the user's custom recordings folder) or disable capture."""
    global _default, _enabled
    with _default_lock:
        if _default is not None:
            _default.close()
        _default = HandCropSink(root, **kwargs) if enabled else None
        _enabled = enabled


def get_sink() -> HandCropSink | None:
    global _default
    with _default_lock:
        # Under pytest the implicit default would write into the user's real recordings
        # folder; tests that want capture call configure(root=tmp_path) explicitly.
        if _default is None and _enabled and "PYTEST_CURRENT_TEST" not in os.environ:
            _default = HandCropSink()
        return _default


def save(iar, slot: int, topleft, histograms, top3) -> bool:
    sink = get_sink()
    return sink.save(iar, slot, topleft, histograms, top3) if sink is not None else False


# ---- build: labelled crops -> card_color_data entries -----------------------------
def build_entries(labels_path: str | os.PathLike[str]) -> dict[str, list[dict[str, int]]]:
    """Fingerprints for labelled crops. labels.json maps crop file name -> card id.

    Histograms come from the crop's index line when present, else they are recomputed from
    the PNG. Several crops with one label are averaged (rounded, so each corner still sums
    to 891).
    """
    from pyclashbot.bot import card_detection as cd  # heavy module, only for the CLI

    labels_path = Path(labels_path)
    root = labels_path.parent
    labels: dict[str, str] = json.loads(labels_path.read_text(encoding="utf-8"))
    by_file = {rec.get("file"): rec for rec in read_index(root)}
    grouped: dict[str, list[np.ndarray]] = {}
    for file_name, card in labels.items():
        rec = by_file.get(file_name) or by_file.get(Path(file_name).name)
        if rec is not None:
            hist = np.asarray(rec["histograms"], dtype=np.int64).reshape(4, -1)
        else:
            import cv2

            crop = cv2.imread(str(root / file_name))
            if crop is None:
                raise FileNotFoundError(root / file_name)
            hist = _crop_histograms(cd, crop)
        grouped.setdefault(card, []).append(hist)
    out: dict[str, list[dict[str, int]]] = {}
    for card, hists in grouped.items():
        mean = _round_preserving_sum(np.mean(hists, axis=0))
        out[card] = [{k: int(v) for k, v in zip(cd.COLORS_KEYS, row, strict=True)} for row in mean]
    return out


def _crop_histograms(cd, crop: np.ndarray) -> np.ndarray:
    labels = cd._label_pixels(crop[:CROP_H, :CROP_W])
    hw, hh = CROP_W // 2, CROP_H // 2
    return np.stack(
        [
            np.bincount(labels[r : r + hh, c : c + hw].ravel(), minlength=len(cd.COLORS_KEYS))
            for r, c in ((0, 0), (0, hw), (hh, 0), (hh, hw))
        ]
    )


def _round_preserving_sum(arr: np.ndarray) -> np.ndarray:
    """Round each row to integers keeping its (integer) sum (largest-remainder)."""
    out = np.floor(arr).astype(np.int64)
    for i, row in enumerate(arr):
        short = round(float(row.sum())) - int(out[i].sum())
        if short > 0:
            out[i, np.argsort(-(row - out[i]), kind="stable")[:short]] += 1
    return out


def format_entries(entries: dict[str, list[dict[str, int]]]) -> str:
    lines = []
    for card, corners in entries.items():
        lines.append(f'    "{card}": [')
        for corner in corners:
            lines.append("        {")
            lines.extend(f'            "{k}": {v},' for k, v in corner.items())
            lines.append("        },")
        lines.append("    ],")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] != "build":
        print("usage: python -m pyclashbot.bot.hand_crop_sink build <labels.json>", file=sys.stderr)
        return 2
    print(format_entries(build_entries(argv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
