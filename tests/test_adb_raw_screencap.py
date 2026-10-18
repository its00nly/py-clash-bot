"""Offline coverage for the raw ``screencap`` path and FrameStream (no hardware).

Raw buffers are synthesized from the RGBA regression fixture, so the decoded BGR
must be pixel-identical to what the ``screencap -p`` + ``cv2.imdecode`` path
yields for the same frame.
"""

from __future__ import annotations

import struct
import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from pyclashbot.emulators import adb_base
from pyclashbot.emulators.adb_base import AdbBasedController, FrameStream, _decode_raw_screencap
from pyclashbot.emulators.base import EmulatorNotReadyError

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "card_hand_bgr_regression.png"


@pytest.fixture(scope="module")
def bgra() -> np.ndarray:
    img = cv2.imread(str(FIXTURE), cv2.IMREAD_UNCHANGED)
    assert img is not None and img.shape == (633, 419, 4)
    return img


@pytest.fixture(scope="module")
def expected_bgr() -> np.ndarray:
    # Exactly what the PNG path does with screencap -p output.
    data = np.frombuffer(FIXTURE.read_bytes(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _raw(bgra: np.ndarray, fmt: int, header_len: int = 16) -> bytes:
    h, w = bgra.shape[:2]
    pixels = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA) if fmt in (1, 2) else bgra
    header = struct.pack("<III", w, h, fmt)
    if header_len == 16:
        header += struct.pack("<I", 0)
    return header + np.ascontiguousarray(pixels).tobytes()


@pytest.mark.parametrize("header_len", [12, 16])
@pytest.mark.parametrize("fmt", [1, 2, 5])
def test_decode_matches_png_path(bgra, expected_bgr, fmt, header_len):
    out = _decode_raw_screencap(_raw(bgra, fmt, header_len))
    assert out is not None
    assert out.shape == (633, 419, 3)
    assert np.array_equal(out, expected_bgr)


def test_decode_uses_header_size_for_other_resolution(bgra, expected_bgr):
    crop = np.ascontiguousarray(bgra[:400, :300])
    out = _decode_raw_screencap(_raw(crop, 1))
    assert out is not None
    assert np.array_equal(out, expected_bgr[:400, :300])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b[:-1],  # truncated
        lambda b: b + b"\0" * 8,  # header length 24
        lambda b: b"",
        lambda b: b"\x01\x02\x03",
        lambda b: b[:8] + struct.pack("<I", 4) + b[12:],  # unknown format (RGB_565)
        lambda b: struct.pack("<II", 0, 0) + b[8:],  # header size disagrees with the buffer
    ],
)
def test_decode_rejects_bad_buffers(bgra, mutate):
    assert _decode_raw_screencap(mutate(_raw(bgra, 1))) is None


class _ScriptedAdb(AdbBasedController):
    """Controller whose adb() returns canned bytes per command."""

    def __init__(self, responses: dict[str, object]) -> None:
        self.device_serial = "127.0.0.1:5555"
        self.adb_path = "adb"
        self.adb_server_port = None
        self.adb_env = None
        self.responses = responses
        self.calls: list[str] = []

    def adb(self, command: str, binary_output: bool = False):
        self.calls.append(command)
        resp = self.responses[command]
        if isinstance(resp, BaseException):
            raise resp
        return subprocess.CompletedProcess([command], 0, resp, b"")


def test_screenshot_prefers_raw(bgra, expected_bgr):
    ctrl = _ScriptedAdb({"exec-out screencap": _raw(bgra, 1)})
    assert np.array_equal(ctrl.screenshot(), expected_bgr)
    assert ctrl.calls == ["exec-out screencap"]


@pytest.mark.parametrize("raw_resp", [b"garbage", b"", RuntimeError("boom")])
def test_screenshot_falls_back_and_stops_retrying_raw(expected_bgr, raw_resp):
    ctrl = _ScriptedAdb({"exec-out screencap": raw_resp, "exec-out screencap -p": FIXTURE.read_bytes()})
    assert np.array_equal(ctrl.screenshot(), expected_bgr)
    assert np.array_equal(ctrl.screenshot(), expected_bgr)
    assert ctrl.calls == ["exec-out screencap", "exec-out screencap -p", "exec-out screencap -p"]
    assert ctrl._raw_screencap_failed is True
    # Per-controller: a fresh controller still tries raw.
    assert AdbBasedController._raw_screencap_failed is False


def test_screenshot_raw_timeout_propagates():
    ctrl = _ScriptedAdb({"exec-out screencap": EmulatorNotReadyError("timed out")})
    with pytest.raises(EmulatorNotReadyError):
        ctrl.screenshot()


def test_raw_disabled_by_flag_and_env(monkeypatch, expected_bgr):
    ctrl = _ScriptedAdb({"exec-out screencap -p": FIXTURE.read_bytes()})
    ctrl.raw_screencap_enabled = False
    assert np.array_equal(ctrl.screenshot(), expected_bgr)
    assert ctrl.calls == ["exec-out screencap -p"]

    ctrl2 = _ScriptedAdb({"exec-out screencap -p": FIXTURE.read_bytes()})
    monkeypatch.setenv(adb_base.RAW_SCREENCAP_ENV, "0")
    ctrl2.screenshot()
    assert ctrl2.calls == ["exec-out screencap -p"]


# --- FrameStream -------------------------------------------------------------


class _ChunkPipe:
    """Fake byte pipe: yields preset chunks (gated so tests control timing), then EOF."""

    def __init__(self) -> None:
        self._chunks: list[bytes] = []
        self._cond = threading.Condition()
        self._eof = False

    def feed(self, data: bytes) -> None:
        with self._cond:
            self._chunks.append(data)
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._eof = True
            self._cond.notify_all()

    def read(self, _n: int) -> bytes:
        with self._cond:
            while not self._chunks and not self._eof:
                self._cond.wait(2)
            return self._chunks.pop(0) if self._chunks else b""


class _Clock:
    def __init__(self) -> None:
        self.t = 100.0
        self.lock = threading.Lock()

    def __call__(self) -> float:
        with self.lock:
            self.t += 0.001
            return self.t


def _frame_bytes(bgra: np.ndarray, value: int) -> bytes:
    frame = bgra.copy()
    frame[0, 0, :3] = value  # mark the frame so we can tell them apart
    return _raw(frame, 5, 16)


def _wait_frames(stream: FrameStream, n: int) -> None:
    for _ in range(400):
        if stream.frames_read >= n:
            return
        threading.Event().wait(0.005)
    raise AssertionError(f"only {stream.frames_read} frames read")


def test_frame_stream_slices_across_chunk_boundaries(bgra):
    pipe = _ChunkPipe()
    ctrl = _ScriptedAdb({})
    stream = FrameStream(ctrl, width=419, height=633, header_len=16, pipe=pipe, clock=_Clock()).start()
    f1, f2 = _frame_bytes(bgra, 11), _frame_bytes(bgra, 22)
    blob = f1 + f2
    # Split at awkward offsets: inside the first header, mid-pixels, across the frame boundary.
    cuts = [5, 7, 1000, len(f1) - 3, len(f1) + 9, len(blob)]
    prev = 0
    for c in cuts:
        pipe.feed(blob[prev:c])
        prev = c
    _wait_frames(stream, 2)
    img, ts = stream.get_frame(timeout=0.5)
    assert img.shape == (633, 419, 3)
    assert tuple(img[0, 0]) == (22, 22, 22)
    assert np.array_equal(img[1:], cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)[1:])
    assert ts > 100.0
    pipe.close()
    stream.stop()
    assert ctrl.calls == []


def test_frame_stream_never_returns_older_frame(bgra):
    pipe = _ChunkPipe()
    clock = _Clock()
    ctrl = _ScriptedAdb({"exec-out screencap": _raw(bgra, 1)})
    stream = FrameStream(ctrl, width=419, height=633, header_len=16, pipe=pipe, clock=clock).start()
    pipe.feed(_frame_bytes(bgra, 1))
    _wait_frames(stream, 1)
    _, ts1 = stream.get_frame(timeout=0.5)

    # Ask for something newer: a frame arriving later satisfies it.
    threading.Timer(0.05, pipe.feed, args=(_frame_bytes(bgra, 2),)).start()
    img2, ts2 = stream.get_frame(newer_than=ts1, timeout=2.0)
    assert ts2 > ts1
    assert tuple(img2[0, 0]) == (2, 2, 2)

    # Nothing newer arrives: falls back to screenshot(), stamped no older than asked.
    img3, ts3 = stream.get_frame(newer_than=ts2, timeout=0.05)
    assert ts3 >= ts2
    assert ctrl.calls == ["exec-out screencap"]
    assert img3.shape == (633, 419, 3)
    pipe.close()
    stream.stop()


def test_frame_stream_falls_back_when_pipe_closes(bgra, expected_bgr):
    pipe = _ChunkPipe()
    ctrl = _ScriptedAdb({"exec-out screencap": _raw(bgra, 1)})
    stream = FrameStream(ctrl, width=419, height=633, header_len=16, pipe=pipe, clock=_Clock()).start()
    pipe.feed(_frame_bytes(bgra, 9)[:500])  # partial frame, then EOF
    pipe.close()
    for _ in range(400):
        if not stream.running:
            break
        threading.Event().wait(0.005)
    assert not stream.running
    img, _ = stream.get_frame(timeout=5.0)  # must not block for the full timeout
    assert np.array_equal(img, expected_bgr)
    assert ctrl.calls == ["exec-out screencap"]
    stream.stop()


def test_frame_stream_probe_learns_geometry(bgra):
    ctrl = _ScriptedAdb({"exec-out screencap": _raw(bgra, 2, 12)})
    pipe = _ChunkPipe()
    stream = FrameStream(ctrl, pipe=pipe).start()
    assert (stream.width, stream.height, stream.header_len) == (419, 633, 12)
    assert stream.frame_size == 12 + 419 * 633 * 4
    pipe.close()
    stream.stop()


def test_frame_stream_off_by_default():
    ctrl = _ScriptedAdb({})
    assert ctrl.frame_stream_enabled is False
    assert ctrl.start_frame_stream() is None
    assert ctrl.calls == []
