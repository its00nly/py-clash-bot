"""Offline coverage for waiting on Android boot and retrying app launch (#741)."""

import subprocess

import pytest

from pyclashbot.emulators import adb_base
from pyclashbot.emulators.adb_base import AdbBasedController
from pyclashbot.emulators.base import EmulatorNotReadyError

PKG = "com.supercell.clashroyale"


class _ScriptedAdb(AdbBasedController):
    """Controller whose adb() replays scripted results per command prefix."""

    def __init__(self, script: dict[str, list]):
        self.script = script
        self.calls: list[str] = []

    def adb(self, command: str, binary_output: bool = False):
        self.calls.append(command)
        for prefix, results in self.script.items():
            if command.startswith(prefix):
                result = results.pop(0) if len(results) > 1 else results[0]
                if isinstance(result, Exception):
                    raise result
                return subprocess.CompletedProcess(command, result[0], result[1], "")
        raise AssertionError(f"unexpected adb command: {command}")


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(adb_base.time, "sleep", lambda _s: None)


def test_wait_for_boot_polls_until_completed():
    ctrl = _ScriptedAdb({"shell getprop": [(0, "\n"), EmulatorNotReadyError("closed"), (0, "1\n")]})
    ctrl.wait_for_boot(timeout=60)
    assert ctrl.calls.count("shell getprop sys.boot_completed") == 3


def test_wait_for_boot_times_out(monkeypatch):
    clock = iter(range(0, 1000, 10))
    monkeypatch.setattr(adb_base.time, "time", lambda: next(clock))
    ctrl = _ScriptedAdb({"shell getprop": [(0, "0\n")]})
    with pytest.raises(EmulatorNotReadyError, match="did not finish booting"):
        ctrl.wait_for_boot(timeout=30)


def test_start_app_when_ready_retries_monkey_failure():
    ctrl = _ScriptedAdb(
        {
            "shell pm list": [(0, f"package:{PKG}\n")],
            "shell monkey": [(-4, "** No activities found to run, monkey aborted.\n"), (0, "Events injected: 1\n")],
        }
    )
    assert ctrl.start_app_when_ready(PKG, attempts=3) is True
    assert sum(c.startswith("shell monkey") for c in ctrl.calls) == 2


def test_start_app_when_ready_gives_up_after_attempts():
    ctrl = _ScriptedAdb({"shell pm list": [(0, f"package:{PKG}\n")], "shell monkey": [(-4, "")]})
    with pytest.raises(EmulatorNotReadyError, match="monkey returned"):
        ctrl.start_app_when_ready(PKG, attempts=3)
    assert sum(c.startswith("shell monkey") for c in ctrl.calls) == 3


def test_start_app_when_ready_fails_fast_when_not_installed():
    ctrl = _ScriptedAdb({"shell pm list": [(0, "package:com.android.music\n")]})
    with pytest.raises(EmulatorNotReadyError, match="not installed"):
        ctrl.start_app_when_ready(PKG)
    assert not any(c.startswith("shell monkey") for c in ctrl.calls)
