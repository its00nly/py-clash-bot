"""Offline coverage for MuMu Player 12 discovery and MuMuManager parsing (#651)."""

import subprocess

import pytest

from pyclashbot.emulators import mumu
from pyclashbot.emulators.base import EmulatorNotReadyError
from pyclashbot.emulators.mumu import MuMuEmulatorController, find_mumu_tools, mumu_adb_port, parse_manager_info


def test_adb_port_per_instance():
    assert mumu_adb_port(0) == 16384
    assert mumu_adb_port(2) == 16448


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ('{"adb_port": 16384, "is_android_started": true}', {"adb_port": 16384, "is_android_started": True}),
        ('log line\n{"adb_host_ip": "127.0.0.1"}\ntrailer', {"adb_host_ip": "127.0.0.1"}),
        ("", {}),
        (None, {}),
        ("not json {", {}),
        ("[1, 2]", {}),
    ],
)
def test_parse_manager_info(stdout, expected):
    assert parse_manager_info(stdout) == expected


@pytest.mark.parametrize("sub", ["shell", "nx_main"])
def test_find_tools_in_either_layout(tmp_path, sub):
    folder = tmp_path / "MuMuPlayer-12.0" / sub
    folder.mkdir(parents=True)
    (folder / "MuMuManager.exe").write_text("")
    (folder / "adb.exe").write_text("")
    assert find_mumu_tools([str(tmp_path / "MuMuPlayer-12.0")]) == (
        str(folder / "MuMuManager.exe"),
        str(folder / "adb.exe"),
    )
    # A registry value pointing straight at the tool dir also resolves.
    assert find_mumu_tools([str(folder)]) is not None


def test_find_tools_missing(tmp_path):
    assert find_mumu_tools([str(tmp_path / "nope")]) is None


def _controller(monkeypatch, index=0) -> MuMuEmulatorController:
    monkeypatch.setattr(mumu, "find_mumu_tools", lambda *_a: ("MuMuManager.exe", "adb.exe"))
    return MuMuEmulatorController(logger=None, instance_index=index)


def test_init_without_install_raises(monkeypatch):
    monkeypatch.setattr(mumu, "find_mumu_tools", lambda *_a: None)
    with pytest.raises(FileNotFoundError):
        MuMuEmulatorController(logger=None)


def test_init_sets_default_serial(monkeypatch):
    ctrl = _controller(monkeypatch, index=1)
    assert ctrl.device_serial == "127.0.0.1:16416"
    assert ctrl.adb_path == "adb.exe"


def test_info_refreshes_serial(monkeypatch):
    ctrl = _controller(monkeypatch)
    calls = []

    def fake_run(argv, timeout):
        calls.append(argv)
        out = '{"adb_host_ip": "127.0.0.1", "adb_port": 16500, "is_android_started": true}'
        return subprocess.CompletedProcess(argv, 0, out, "")

    monkeypatch.setattr(mumu, "run_command", fake_run)
    info = ctrl._info()
    ctrl._refresh_serial(info)
    assert calls[0] == ["MuMuManager.exe", "info", "-v", "0"]
    assert ctrl.device_serial == "127.0.0.1:16500"


def test_manager_timeout_becomes_not_ready(monkeypatch):
    ctrl = _controller(monkeypatch)

    def slow(argv, timeout):
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.setattr(mumu, "run_command", slow)
    with pytest.raises(EmulatorNotReadyError):
        ctrl.start()
    ctrl.stop()  # stop swallows it
    assert ctrl._info() == {}
