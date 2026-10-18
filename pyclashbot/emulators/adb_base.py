import argparse
import logging
import os
import re
import shlex
import struct
import subprocess
import threading
import time
from abc import ABC
from collections.abc import Callable, Iterable
from subprocess import TimeoutExpired
from typing import Protocol

import cv2
import numpy as np

from pyclashbot.emulators.base import (
    BaseEmulatorController,
    EmulatorNotReadyError,
)
from pyclashbot.utils.subprocess import run as run_command

logger = logging.getLogger(__name__)

# Valid device serial patterns to prevent command injection
_DEVICE_SERIAL_PATTERNS = [
    re.compile(r"^[a-zA-Z0-9._-]+:\d{1,5}$"),  # host:port (localhost, IP, hostname)
    re.compile(r"^emulator-\d+$"),  # emulator-5554
    re.compile(r"^[A-Za-z0-9]{6,}$"),  # USB serial
]


def validate_device_serial(serial: str | None) -> bool:
    """Validate device serial format to prevent command injection.

    Returns True if serial is None (will use auto-discovery) or matches
    a known safe pattern.
    """
    if serial is None:
        return True
    serial = serial.strip()
    if not serial:
        return True  # Empty treated as None
    return any(pattern.match(serial) for pattern in _DEVICE_SERIAL_PATTERNS)


# Raw `screencap` (no -p) output: a little-endian uint32 header followed by
# w*h*4 bytes of pixels. Older Android writes 12 bytes (w, h, format); Android 12+
# appends a colour-space word (16 bytes). PNG encoding on the device is skipped.
_RAW_HEADER_LENGTHS = (12, 16)
_RAW_FORMAT_TO_CV2 = {
    1: cv2.COLOR_RGBA2BGR,  # PIXEL_FORMAT_RGBA_8888
    2: cv2.COLOR_RGBA2BGR,  # PIXEL_FORMAT_RGBX_8888
    5: cv2.COLOR_BGRA2BGR,  # PIXEL_FORMAT_BGRA_8888
}
DEFAULT_RAW_WIDTH = 419
DEFAULT_RAW_HEIGHT = 633
_MAX_RAW_DIM = 16384

# Env off switch for the raw screencap path ("0"/"false"/"off"/"no" disables it).
RAW_SCREENCAP_ENV = "PYCLASHBOT_RAW_SCREENCAP"


def _raw_header_length(data_len: int, w: int, h: int) -> int | None:
    hl = data_len - w * h * 4
    return hl if hl in _RAW_HEADER_LENGTHS else None


def _decode_raw_screencap(
    data: bytes | bytearray | memoryview | None,
    w: int = DEFAULT_RAW_WIDTH,
    h: int = DEFAULT_RAW_HEIGHT,
) -> np.ndarray | None:
    """Decode raw ``screencap`` output to a BGR image, or return None.

    ``w``/``h`` are the expected frame size. If the buffer does not fit them, the
    width/height stored in the header are used instead, so a device at another
    resolution still decodes (at its own size; the PNG path does not resize
    either). The result is pixel-identical to ``cv2.imdecode(png,
    cv2.IMREAD_COLOR)`` of ``screencap -p`` for the same frame: alpha is dropped,
    not composited.

    Returns None for a bad length, inconsistent header or unknown pixel format.
    """
    if not data or len(data) < 12:
        return None
    buf = memoryview(data).cast("B")
    hdr_w, hdr_h, fmt = struct.unpack_from("<III", buf, 0)

    hl = _raw_header_length(len(buf), w, h)
    if hl is None or (hdr_w, hdr_h) != (w, h):
        # Use the header's own size (device at a different resolution).
        if not (0 < hdr_w <= _MAX_RAW_DIM and 0 < hdr_h <= _MAX_RAW_DIM):
            return None
        hl = _raw_header_length(len(buf), hdr_w, hdr_h)
        if hl is None:
            return None
        w, h = hdr_w, hdr_h

    code = _RAW_FORMAT_TO_CV2.get(fmt)
    if code is None:
        return None
    pixels = np.frombuffer(buf, dtype=np.uint8, count=w * h * 4, offset=hl).reshape(h, w, 4)
    return cv2.cvtColor(pixels, code)


def _raw_screencap_env_enabled() -> bool:
    return os.environ.get(RAW_SCREENCAP_ENV, "1").strip().lower() not in {"0", "false", "off", "no"}


class AdbBasedController(BaseEmulatorController, ABC):
    """
    Abstract base class for emulator controllers that use ADB.

    This class provides concrete implementations for common ADB operations
    (click, swipe, screenshot, start_app, installation waiting, device listing)
    with configurable ADB path, server port, and environment.

    Subclasses must set before using ADB methods:
    - device_serial: str | None (the ADB device serial to target)

    Subclasses may override these properties for custom ADB configuration:
    - adb_path: str (default: "adb" for system ADB)
    - adb_server_port: int | None (default: None, no custom port)
    - adb_env: dict | None (default: None, inherit environment)
    """

    device_serial: str | None = None

    adb_path: str = "adb"
    adb_server_port: int | None = None
    adb_env: dict | None = None

    # Raw (non-PNG) screencap: on by default. Set False on a class/instance, or
    # PYCLASHBOT_RAW_SCREENCAP=0, to always use `screencap -p`.
    raw_screencap_enabled: bool = True
    # Set per controller after the first raw failure so it is not retried each frame.
    _raw_screencap_failed: bool = False
    # Persistent screencap stream for fights; off by default (see FrameStream).
    frame_stream_enabled: bool = False
    frame_stream: "FrameStream | None" = None
    # time.monotonic() of the last input sent (click / tap_sequence / swipe), for
    # freshness barriers: a frame captured before this cannot show that input.
    last_input_time: float = 0.0

    @classmethod
    def find_adb(cls) -> str | None:
        """Find ADB executable. Override in subclasses for bundled ADB."""
        return None  # Default: use system adb

    @classmethod
    def discover_devices(cls) -> list[str]:
        """List connected ADB device serials using this controller's ADB."""
        adb_path = cls.find_adb() or "adb"
        argv = [adb_path]
        if cls.adb_server_port:
            argv += ["-P", str(cls.adb_server_port)]
        argv.append("devices")
        try:
            result = run_command(argv, timeout=15)
        except OSError:
            return []  # ADB not available
        except TimeoutExpired as e:
            raise EmulatorNotReadyError("adb devices timed out after 15s") from e
        if result.returncode != 0:
            return []
        devices = []
        for line in result.stdout.strip().splitlines()[1:]:
            if "\tdevice" in line:
                devices.append(line.split("\t")[0])
        return devices

    @classmethod
    def is_device_connected(cls, device_serial: str | None) -> bool:
        """Check if a device serial is in the connected devices list."""
        if not device_serial:
            return False
        return device_serial in cls.discover_devices()

    def _is_server_command(self, command: str) -> bool:
        """Check if command targets ADB server rather than a specific device.

        Server commands should not have -s device_serial added.
        """
        c = command.strip()
        if re.match(r"-s\s+\S+", c):
            return True  # Already device-scoped
        first = c.split()[0] if c else ""
        return first in {
            "connect",
            "disconnect",
            "devices",
            "start-server",
            "kill-server",
            "version",
            "help",
            "keys",
        }

    def _adb_prefix(self, device_scoped: bool = True) -> list[str]:
        """argv up to and including ``-s serial``; validates the serial first."""
        if self.device_serial and not validate_device_serial(self.device_serial):
            raise ValueError(f"Invalid device serial format: {self.device_serial}")
        argv = [self.adb_path]
        if self.adb_server_port:
            argv += ["-P", str(self.adb_server_port)]
        if device_scoped and self.device_serial:
            argv += ["-s", self.device_serial]
        return argv

    def _build_adb_argv(self, command: str) -> list[str]:
        return self._adb_prefix(not self._is_server_command(command)) + shlex.split(command)

    def adb(self, command: str, binary_output: bool = False) -> subprocess.CompletedProcess:
        """
        Execute an ADB command with automatic device scoping.

        Builds the command with the configured ADB path, optional server port,
        and device serial (unless it's a server-level command).

        Args:
            command (str): The ADB command to execute (e.g., "shell input tap 1 2").
            binary_output (bool): Whether to capture stdout as bytes (True) or
                                  decode as text (False).

        Returns:
            subprocess.CompletedProcess: The result of the ADB command.

        Raises:
            ValueError: If device_serial contains invalid characters.
        """
        argv = self._build_adb_argv(command)

        logger.debug("Executing ADB: %s", argv)

        try:
            result = run_command(argv, timeout=30, text=not binary_output, env=self.adb_env)
        except TimeoutExpired as e:
            # A wedged adb is an unreachable emulator, not merely a failed command.
            raise EmulatorNotReadyError(f"adb command timed out after 30s: {command}") from e

        if binary_output:
            logger.debug("ADB result: rc=%d, stdout=%d bytes", result.returncode, len(result.stdout or b""))
        else:
            stdout_preview = (result.stdout or "")[:200]
            logger.debug("ADB result: rc=%d, stdout=%r", result.returncode, stdout_preview)

        if result.returncode != 0 and result.stderr:
            stderr_msg = (
                result.stderr.strip()
                if isinstance(result.stderr, str)
                else result.stderr.decode("utf-8", "ignore").strip()
            )
            if stderr_msg:
                logger.warning("ADB command failed (rc=%d): %s", result.returncode, stderr_msg)

        return result

    def is_app_installed(self, package: str) -> bool:
        return self._check_app_installed(package)

    def _check_app_installed(self, package_name: str) -> bool:
        """Check if an app is installed via ADB.

        Args:
            package_name (str): The package name to check (e.g., "com.supercell.clashroyale").

        Returns:
            bool: True if the app is installed, False otherwise.
        """
        logger.debug("Checking if app is installed: %s", package_name)
        result = self.adb("shell pm list packages")
        installed = result.stdout is not None and package_name in result.stdout
        if installed:
            logger.info("App %s is installed", package_name)
        else:
            logger.info("App %s is NOT installed", package_name)
        return installed

    def list_devices(self) -> list[tuple[str, str]]:
        """List all ADB devices with their status.

        Returns:
            list of (serial, status) tuples, e.g., [("localhost:6520", "device"), ("emulator-5554", "offline")]
        """
        result = self.adb("devices")
        devices = []
        if result.stdout:
            for line in result.stdout.strip().splitlines()[1:]:
                parts = line.split()
                if len(parts) >= 2:
                    devices.append((parts[0], parts[1]))

        logger.info("Found %d ADB device(s): %s", len(devices), devices)

        for serial, status in devices:
            if status == "offline":
                logger.warning("Device %s is offline - may need reconnection", serial)
            elif status == "unauthorized":
                logger.warning("Device %s is unauthorized - check USB debugging authorization on device", serial)

        return devices

    def list_online_devices(self) -> list[str]:
        """List serials of online (status='device') devices only."""
        return [serial for serial, status in self.list_devices() if status == "device"]

    def click(self, x_coord: int, y_coord: int, clicks: int = 1, interval: float = 0.0):
        """Click on the screen using ADB input tap."""
        for _ in range(max(1, clicks)):
            self.adb(f"shell input tap {x_coord} {y_coord}")
            self.last_input_time = time.monotonic()
            if clicks > 1:
                time.sleep(max(0.0, interval))

    def tap_sequence(self, points: Iterable[tuple[int, int]]) -> None:
        """Send several taps, in order, with ONE adb call.

        Runs ``shell "input tap x1 y1; input tap x2 y2; ..."`` so only one adb
        process and handshake is paid, instead of one per tap with click().
        """
        taps = [f"input tap {int(x)} {int(y)}" for x, y in points]
        if not taps:
            return
        joined = "; ".join(taps)
        self.adb(f'shell "{joined}"')
        self.last_input_time = time.monotonic()

    def swipe(self, x_coord1: int, y_coord1: int, x_coord2: int, y_coord2: int):
        """Swipe on the screen using ADB input swipe."""
        self.adb(f"shell input swipe {x_coord1} {y_coord1} {x_coord2} {y_coord2}")
        self.last_input_time = time.monotonic()

    def _use_raw_screencap(self) -> bool:
        return self.raw_screencap_enabled and not self._raw_screencap_failed and _raw_screencap_env_enabled()

    def _screenshot_raw(self) -> np.ndarray | None:
        """Try raw ``exec-out screencap``.

        Returns None, and disables raw for this controller, on any failure so the
        caller falls back to PNG. A timed-out adb (EmulatorNotReadyError) is not a
        format problem and propagates unchanged.
        """
        img = None
        try:
            result = self.adb("exec-out screencap", binary_output=True)
            if result.returncode == 0 and result.stdout:
                img = _decode_raw_screencap(result.stdout)
        except EmulatorNotReadyError:
            raise
        except Exception:
            logger.debug("Raw screencap raised; falling back to PNG", exc_info=True)
            img = None
        if img is None:
            logger.info("Raw screencap unavailable on %s; using screencap -p from now on", self.device_serial)
            self._raw_screencap_failed = True
        return img

    def _screenshot_png(self) -> np.ndarray:
        result = self.adb("exec-out screencap -p", binary_output=True)

        if result.returncode != 0 or not result.stdout:
            err = result.stderr if result.stderr else b"Unknown ADB error"
            error_msg = err.decode("utf-8", "ignore") if isinstance(err, bytes) else str(err)
            logger.error("Screenshot capture failed: %s", error_msg)
            raise RuntimeError(f"ADB screencap failed: {error_msg}")

        img_array = np.frombuffer(result.stdout, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            logger.error("Failed to decode screenshot data (%d bytes)", len(result.stdout))
            raise ValueError("Failed to decode screenshot. Image data may be corrupt or empty.")
        return img

    def screenshot(self) -> np.ndarray:
        """
        Capture a screenshot as a BGR image.

        Tries raw ``exec-out screencap`` first (no on-device PNG encode), then
        falls back to ``exec-out screencap -p``; both give identical pixels. Raw
        is skipped after its first failure on this controller, or when disabled
        via ``raw_screencap_enabled = False`` / ``PYCLASHBOT_RAW_SCREENCAP=0``.

        Returns:
            np.ndarray: The screenshot as a CV2 (BGR) image.

        Raises:
            RuntimeError: If the ADB screencap command fails.
            ValueError: If the captured image data cannot be decoded.
        """
        img = self._screenshot_raw() if self._use_raw_screencap() else None
        if img is None:
            img = self._screenshot_png()
        logger.debug("Screenshot captured: %dx%d", img.shape[1], img.shape[0])
        return img

    def start_frame_stream(self) -> "FrameStream | None":
        """Start the persistent capture stream if ``frame_stream_enabled``.

        Meant for fights only: start on fight start, stop_frame_stream() on fight
        end. Returns the running stream, or None when disabled or it failed.
        """
        if not self.frame_stream_enabled:
            return None
        if self.frame_stream is not None and self.frame_stream.running:
            return self.frame_stream
        stream = FrameStream(self)
        try:
            stream.start()
        except Exception:
            logger.warning("Frame stream failed to start; using screenshot()", exc_info=True)
            stream.stop()
            return None
        self.frame_stream = stream
        return stream

    def stop_frame_stream(self) -> None:
        if self.frame_stream is not None:
            self.frame_stream.stop()
            self.frame_stream = None

    def get_fresh_frame(self, newer_than: float = float("-inf"), timeout: float = 1.0) -> tuple[np.ndarray, float]:
        """Return ``(bgr, capture_time)`` on the time.monotonic clock.

        Uses the frame stream when running (blocking for a frame captured after
        ``newer_than``), else screenshot() stamped with the time it was requested.
        """
        if self.frame_stream is not None and self.frame_stream.running:
            return self.frame_stream.get_frame(newer_than, timeout=timeout)
        ts = time.monotonic()
        return self.screenshot(), ts

    def start_app(self, package_name: str):
        """Start an app on the emulator using ADB monkey command.

        Args:
            package_name (str): The package name to start.

        Returns:
            True if the app was launched.

        Raises:
            EmulatorNotReadyError: If the app is not installed or cannot be launched.
        """
        if not self._check_app_installed(package_name):
            raise EmulatorNotReadyError(f"{package_name} is not installed on the emulator")

        logger.info("Launching app: %s", package_name)
        result = self.adb(f"shell monkey -p {package_name} -c android.intent.category.LAUNCHER 1")
        if result.returncode != 0:
            raise EmulatorNotReadyError(f"Failed to launch {package_name}: monkey returned rc={result.returncode}")
        return True

    def wait_for_boot(self, timeout: float = 120, poll_interval: float = 2) -> None:
        """Block until Android reports ``sys.boot_completed=1``.

        adbd accepts connections (and ``pm list packages`` answers) well before
        Android finishes booting; launching an app in that window fails with
        "No activities found" and the connection can drop mid-boot (issue #741).

        Raises:
            EmulatorNotReadyError: If boot does not complete within ``timeout``.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                result = self.adb("shell getprop sys.boot_completed")
            except EmulatorNotReadyError:
                result = None
            if result is not None and result.returncode == 0 and (result.stdout or "").strip() == "1":
                return
            time.sleep(poll_interval)
        raise EmulatorNotReadyError(f"Android did not finish booting within {timeout:.0f}s")

    def start_app_when_ready(self, package_name: str, attempts: int = 5, retry_delay: float = 3) -> bool:
        """Launch an app, retrying while the emulator is still settling after boot.

        A missing app is reported immediately; only launch failures are retried.
        """
        if not self._check_app_installed(package_name):
            raise EmulatorNotReadyError(f"{package_name} is not installed on the emulator")
        for attempt in range(1, attempts + 1):
            try:
                return self.start_app(package_name)
            except EmulatorNotReadyError:
                if attempt == attempts:
                    raise
                logger.info("Launch attempt %d/%d for %s failed, retrying", attempt, attempts, package_name)
                time.sleep(retry_delay)
        return False


class ReadablePipe(Protocol):
    def read(self, n: int, /) -> bytes: ...


class FrameStream:
    """Persistent raw-screencap stream over one long-lived adb process.

    Runs ``exec-out sh -c 'while true; do screencap; done'``. A reader thread
    slices the byte stream into fixed-size frames (header + w*h*4) and keeps only
    the latest one with its capture timestamp (clock() when its last byte
    arrived). get_frame(newer_than) blocks for a fresh frame and falls back to
    controller.screenshot() on timeout, a bad frame, or once the pipe has closed.

    Tests can pass ``pipe`` (anything with ``read(n) -> bytes``) together with
    ``width``/``height``/``header_len`` to skip the probe and the Popen.
    """

    STREAM_COMMAND = "sh -c 'while true; do screencap; done'"

    def __init__(
        self,
        controller: AdbBasedController,
        *,
        width: int | None = None,
        height: int | None = None,
        header_len: int | None = None,
        pipe: ReadablePipe | None = None,
        clock: Callable[[], float] = time.monotonic,
        read_size: int = 1 << 16,
    ) -> None:
        self.controller = controller
        self.width = width
        self.height = height
        self.header_len = header_len
        self._pipe: ReadablePipe | None = pipe
        self._proc: subprocess.Popen | None = None
        self._clock = clock
        self._read_size = read_size
        self._cond = threading.Condition()
        self._latest: bytes | None = None
        self._latest_ts: float = float("-inf")
        self._closed = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.frames_read = 0

    @property
    def frame_size(self) -> int:
        if self.width is None or self.height is None or self.header_len is None:
            raise RuntimeError("FrameStream frame size unknown; call start() first")
        return self.header_len + self.width * self.height * 4

    @property
    def running(self) -> bool:
        return self._thread is not None and not self._closed

    def _probe(self) -> None:
        """One raw screencap to learn width, height and header length."""
        result = self.controller.adb("exec-out screencap", binary_output=True)
        data = result.stdout or b""
        if result.returncode != 0 or len(data) < 12:
            raise RuntimeError("raw screencap probe failed")
        w, h, _fmt = struct.unpack_from("<III", data, 0)
        hl = _raw_header_length(len(data), w, h)
        if hl is None or _decode_raw_screencap(data, w, h) is None:
            raise RuntimeError("raw screencap probe returned an unsupported frame")
        self.width, self.height, self.header_len = w, h, hl

    def start(self) -> "FrameStream":
        if self.width is None or self.height is None or self.header_len is None:
            self._probe()
        if self._pipe is None:
            argv = [*self.controller._adb_prefix(), "exec-out", self.STREAM_COMMAND]
            logger.debug("Starting frame stream: %s", argv)
            # Long-lived streaming reader, not an output-capturing call: run() does not apply.
            self._proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                env=self.controller.adb_env,
            )
            self._pipe = self._proc.stdout
        self._thread = threading.Thread(target=self._reader, name="adb-frame-stream", daemon=True)
        self._thread.start()
        return self

    def _reader(self) -> None:
        size = self.frame_size
        buf = bytearray()
        pipe = self._pipe
        try:
            while pipe is not None and not self._stop.is_set():
                chunk = pipe.read(self._read_size)
                if not chunk:
                    break
                buf += chunk
                if len(buf) < size:
                    continue
                n = len(buf) // size
                last = bytes(buf[(n - 1) * size : n * size])
                del buf[: n * size]
                ts = self._clock()
                with self._cond:
                    self._latest = last
                    self._latest_ts = max(ts, self._latest_ts)
                    self.frames_read += n
                    self._cond.notify_all()
        except Exception:
            logger.debug("Frame stream reader stopped on error", exc_info=True)
        finally:
            with self._cond:
                self._closed = True
                self._cond.notify_all()

    def get_frame(self, newer_than: float = float("-inf"), timeout: float = 1.0) -> tuple[np.ndarray, float]:
        """Latest frame captured after ``newer_than``, as ``(bgr, capture_time)``.

        Never returns a frame older than ``newer_than``: on timeout, a closed pipe
        or an undecodable frame it falls back to controller.screenshot(), stamped
        max(request time, newer_than).
        """
        deadline = self._clock() + timeout
        with self._cond:
            while self._latest_ts <= newer_than and not self._closed:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
            data, ts = self._latest, self._latest_ts
        if data is not None and ts > newer_than:
            img = _decode_raw_screencap(data, self.width or DEFAULT_RAW_WIDTH, self.height or DEFAULT_RAW_HEIGHT)
            if img is not None:
                return img, ts
        ts = self._clock()
        return self.controller.screenshot(), max(ts, newer_than)

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=5)
            except Exception:
                logger.debug("Frame stream process cleanup failed", exc_info=True)
            self._proc = None
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)
        with self._cond:
            self._closed = True
            self._cond.notify_all()


class _BenchController(AdbBasedController):
    """Minimal controller for the --bench CLI (no emulator lifecycle)."""

    def __init__(self, serial: str | None, adb_path: str, port: int | None) -> None:
        self.device_serial = serial
        self.adb_path = adb_path
        self.adb_server_port = port
        self.adb_env = None


def _bench(ctrl: AdbBasedController, n: int) -> None:
    def timed(cmd: str) -> tuple[list[float], bytes]:
        times: list[float] = []
        last = b""
        for _ in range(n):
            t0 = time.perf_counter()
            last = ctrl.adb(cmd, binary_output=True).stdout or b""
            times.append(time.perf_counter() - t0)
        return times, last

    def summary(name: str, times: list[float], decode_s: float) -> None:
        arr = np.array(times) * 1000
        print(
            f"{name:>4}: mean {arr.mean():7.1f} ms  median {np.median(arr):7.1f} ms  "
            f"min {arr.min():7.1f} ms  (+decode {decode_s * 1000:.1f} ms)"
        )

    raw_t, raw_data = timed("exec-out screencap")
    t0 = time.perf_counter()
    raw_img = _decode_raw_screencap(raw_data)
    raw_dec = time.perf_counter() - t0
    png_t, png_data = timed("exec-out screencap -p")
    t0 = time.perf_counter()
    png_img = cv2.imdecode(np.frombuffer(png_data, np.uint8), cv2.IMREAD_COLOR) if png_data else None
    png_dec = time.perf_counter() - t0

    print(f"{n} captures each, device {ctrl.device_serial or '(default)'}")
    summary("raw", raw_t, raw_dec)
    summary("png", png_t, png_dec)
    if raw_img is None:
        print(f"raw decode FAILED ({len(raw_data)} bytes): screenshot() would fall back to -p")
    elif png_img is not None:
        same = raw_img.shape == png_img.shape and bool(np.array_equal(raw_img, png_img))
        print(f"shapes raw {raw_img.shape} png {png_img.shape}; identical={same} (only meaningful on a static screen)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pyclashbot.emulators.adb_base")
    parser.add_argument("--bench", type=int, metavar="N", help="time N raw vs N -p screencaps")
    parser.add_argument("--serial", default=None, help="adb device serial (default: the only device)")
    parser.add_argument("--adb", default="adb", help="adb executable path")
    parser.add_argument("--port", type=int, default=None, help="adb server port (-P), e.g. 5041 for BlueStacks")
    args = parser.parse_args(argv)
    if not args.bench:
        parser.print_help()
        return 2
    _bench(_BenchController(args.serial, args.adb, args.port), max(1, args.bench))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
