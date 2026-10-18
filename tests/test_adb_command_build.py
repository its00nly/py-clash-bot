"""Offline coverage for ADB command construction (argv, no shell).

Pins that ``adb()`` and ``discover_devices()`` build proper argv lists and route
through the timeout helper with ``shell`` never involved, without any hardware.
The helper is monkeypatched to capture the argv and return a canned result.
"""

import subprocess

import pytest

from pyclashbot.emulators import adb_base
from pyclashbot.emulators.adb_base import AdbBasedController
from pyclashbot.emulators.base import EmulatorNotReadyError


class _FakeAdb(AdbBasedController):
    """Concrete, cheap-to-build controller (base __init__ is bypassed)."""


def _make_controller(**attrs) -> _FakeAdb:
    ctrl = _FakeAdb.__new__(_FakeAdb)
    ctrl.device_serial = attrs.get("device_serial", "127.0.0.1:5555")
    ctrl.adb_path = attrs.get("adb_path", "adb")
    ctrl.adb_server_port = attrs.get("adb_server_port", None)
    ctrl.adb_env = attrs.get("adb_env", None)
    return ctrl


def _capture(monkeypatch, stdout: str = "ok"):
    seen: dict = {}

    def spy(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(adb_base, "run_command", spy)
    return seen


def test_adb_scopes_serial_via_argv(monkeypatch):
    seen = _capture(monkeypatch)
    _make_controller(adb_server_port=5041).adb("shell wm size")
    argv = seen["argv"]

    assert "-P" in argv and argv[argv.index("-P") + 1] == "5041"
    assert "-s" in argv and argv[argv.index("-s") + 1] == "127.0.0.1:5555"
    assert "wm" in argv  # the command itself made it in after shlex.split
    assert "shell" not in seen["kwargs"]  # never a shell=True kwarg


def test_adb_binary_requests_bytes(monkeypatch):
    seen = _capture(monkeypatch)
    _make_controller().adb("exec-out screencap -p", binary_output=True)

    assert seen["kwargs"]["text"] is False


def test_adb_raises_not_ready_on_timeout(monkeypatch):
    def spy(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 30)

    monkeypatch.setattr(adb_base, "run_command", spy)

    with pytest.raises(EmulatorNotReadyError, match="timed out"):
        _make_controller().adb("shell pm list packages")


def test_discover_devices_raises_on_timeout(monkeypatch):
    def spy(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 15)

    monkeypatch.setattr(adb_base, "run_command", spy)

    with pytest.raises(EmulatorNotReadyError, match="adb devices"):
        AdbBasedController.discover_devices()


def _capture_all(monkeypatch):
    calls: list[list[str]] = []

    def spy(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(adb_base, "run_command", spy)
    return calls


def test_tap_sequence_is_one_adb_call_with_taps_in_order(monkeypatch):
    calls = _capture_all(monkeypatch)
    ctrl = _make_controller()
    ctrl.tap_sequence([(10, 20), (300, 400)])

    assert len(calls) == 1
    argv = calls[0]
    assert argv[:3] == ["adb", "-s", "127.0.0.1:5555"]
    # One remote-shell string; adb hands it to the device shell, which runs both in order.
    assert argv[3:] == ["shell", "input tap 10 20; input tap 300 400"]


def test_tap_sequence_empty_sends_nothing(monkeypatch):
    calls = _capture_all(monkeypatch)
    ctrl = _make_controller()
    ctrl.tap_sequence([])
    assert calls == []
    assert ctrl.last_input_time == 0.0


def test_click_and_tap_sequence_update_last_input_time(monkeypatch):
    _capture_all(monkeypatch)
    ticks = iter([5.0, 7.5])
    monkeypatch.setattr(adb_base.time, "monotonic", lambda: next(ticks))
    ctrl = _make_controller()
    assert ctrl.last_input_time == 0.0

    ctrl.click(1, 2)
    assert ctrl.last_input_time == 5.0
    ctrl.tap_sequence([(3, 4), (5, 6)])
    assert ctrl.last_input_time == 7.5


def test_raw_screencap_command_argv(monkeypatch):
    seen = _capture(monkeypatch)
    _make_controller().adb("exec-out screencap", binary_output=True)
    assert seen["argv"][-2:] == ["exec-out", "screencap"]


def test_frame_stream_command_is_single_remote_string():
    from pyclashbot.emulators.adb_base import FrameStream

    # adb joins exec-out args with spaces for the device shell, so the loop must be
    # one argv element to keep its quoting intact.
    assert FrameStream.STREAM_COMMAND == "sh -c 'while true; do screencap; done'"
