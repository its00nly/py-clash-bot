"""MuMu Player 12 controller (Windows, experimental).

MuMu exposes a ``MuMuManager.exe`` CLI for lifecycle (``control -v N launch|shutdown``,
``info -v N`` -> JSON) and a standard ADB endpoint per instance (port
``16384 + 32 * N``). Everything screen-related goes over ADB via ``AdbBasedController``;
resolution is forced with ``wm size``/``wm density`` like the plain ADB controller,
so no MuMu setting keys are needed.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import suppress
from subprocess import TimeoutExpired

from pyclashbot.bot.coords import BOOT_MAIN_MENU_DEADSPACE_COORD, CLAN_VOYAGE_CLOSE_BUTTON_COORDS
from pyclashbot.bot.state_detect import check_if_on_clan_voyage, check_if_on_clash_main_menu
from pyclashbot.emulators.adb_base import AdbBasedController
from pyclashbot.emulators.base import CLASH_ROYALE_PACKAGE, EmulatorNotReadyError
from pyclashbot.utils.platform import Platform, is_windows
from pyclashbot.utils.subprocess import run as run_command

_BASE_ADB_PORT = 16384
_PORT_STRIDE = 32
_REQUIRED_SIZE = (419, 633)
_REQUIRED_DENSITY = 160
_UNINSTALL_KEYS = (
    r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\MuMuPlayer-12.0",
    r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\MuMuPlayer-12.0",
    r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\MuMuPlayer",
    r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\MuMuPlayer",
)
_DEFAULT_INSTALL_DIRS = (
    r"C:\Program Files\Netease\MuMuPlayer-12.0",
    r"C:\Program Files\Netease\MuMu Player 12",
    r"C:\Program Files\Netease\MuMuPlayer",
)
# Newer builds (v4+) moved the tools from shell\ to nx_main\.
_TOOL_SUBDIRS = ("shell", "nx_main")


def mumu_adb_port(instance_index: int) -> int:
    return _BASE_ADB_PORT + _PORT_STRIDE * instance_index


def parse_manager_info(stdout: str | None) -> dict:
    """Parse ``MuMuManager info`` output, tolerating log noise around the JSON."""
    if not stdout:
        return {}
    start, end = stdout.find("{"), stdout.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        data = json.loads(stdout[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _registry_install_dirs() -> list[str]:
    if not is_windows():
        return []
    import winreg

    found: list[str] = []
    for key_path in _UNINSTALL_KEYS:
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                for value_name in ("InstallLocation", "UninstallString", "DisplayIcon"):
                    with suppress(OSError):
                        raw = str(winreg.QueryValueEx(key, value_name)[0]).strip().strip('"')
                        if raw:
                            found.append(raw if os.path.isdir(raw) else os.path.dirname(raw))
        except OSError:
            continue
    return found


def find_mumu_tools(candidates: list[str] | None = None) -> tuple[str, str] | None:
    """Return (MuMuManager.exe, adb.exe) paths, or None if MuMu isn't installed."""
    roots = candidates if candidates is not None else [*_registry_install_dirs(), *_DEFAULT_INSTALL_DIRS]
    for root in roots:
        # Registry values may point at a tool subdir already; check it and its parent.
        for base in (root, os.path.dirname(root)):
            for sub in ("", *_TOOL_SUBDIRS):
                folder = os.path.join(base, sub) if sub else base
                manager = os.path.join(folder, "MuMuManager.exe")
                adb = os.path.join(folder, "adb.exe")
                if os.path.isfile(manager) and os.path.isfile(adb):
                    return manager, adb
    return None


class MuMuEmulatorController(AdbBasedController):
    supported_platforms = [Platform.WINDOWS]

    @classmethod
    def find_adb(cls) -> str | None:
        tools = find_mumu_tools()
        return tools[1] if tools else None

    def __init__(self, logger, instance_index: int = 0):
        self.logger = logger
        self.instance_index = max(0, int(instance_index))
        tools = find_mumu_tools()
        if tools is None:
            raise FileNotFoundError("MuMu Player 12 not found. Install it from mumuplayer.com.")
        self.manager_path, self.adb_path = tools
        self.device_serial = f"127.0.0.1:{mumu_adb_port(self.instance_index)}"

    # ---- MuMuManager CLI -------------------------------------------------

    def _manager(self, *args: str, timeout: float = 30):
        argv = [self.manager_path, *args]
        try:
            return run_command(argv, timeout=timeout)
        except TimeoutExpired as e:
            raise EmulatorNotReadyError(f"MuMuManager timed out: {' '.join(args)}") from e

    def _info(self) -> dict:
        try:
            result = self._manager("info", "-v", str(self.instance_index), timeout=15)
        except EmulatorNotReadyError:
            return {}
        return parse_manager_info(result.stdout)

    def _refresh_serial(self, info: dict) -> None:
        host, port = info.get("adb_host_ip"), info.get("adb_port")
        if host and port:
            self.device_serial = f"{host}:{port}"

    def _is_running(self) -> bool:
        info = self._info()
        return bool(info.get("is_process_started") or info.get("is_android_started"))

    # ---- lifecycle -------------------------------------------------------

    def start(self):
        self._manager("control", "-v", str(self.instance_index), "launch")

    def stop(self):
        with suppress(EmulatorNotReadyError):
            self._manager("control", "-v", str(self.instance_index), "shutdown")

    def _connect(self) -> bool:
        self.adb(f"connect {self.device_serial}")
        state = self.adb("get-state")
        return state.returncode == 0 and "device" in (state.stdout or "")

    def _ensure_screen_geometry(self) -> None:
        width, height = _REQUIRED_SIZE
        self.adb(f"shell wm size {width}x{height}")
        self.adb(f"shell wm density {_REQUIRED_DENSITY}")

    def restart(self) -> bool:
        start_ts = time.time()
        self.logger.change_status(f"Restarting MuMu Player (instance {self.instance_index})...")

        self.stop()
        deadline = time.time() + 60
        while self._is_running():
            if time.time() > deadline:
                raise EmulatorNotReadyError("MuMu instance did not shut down within 60s")
            time.sleep(1)

        self.logger.change_status("Launching MuMu Player...")
        self.start()

        deadline = time.time() + 180
        while True:
            info = self._info()
            if info.get("is_android_started"):
                self._refresh_serial(info)
                break
            if time.time() > deadline:
                raise EmulatorNotReadyError("MuMu instance did not start within 180s")
            time.sleep(2)

        self.logger.change_status(f"Connecting ADB to {self.device_serial}...")
        deadline = time.time() + 60
        while not self._connect():
            if time.time() > deadline:
                raise EmulatorNotReadyError(f"Could not connect ADB to MuMu at {self.device_serial}")
            time.sleep(1)

        self.logger.change_status("Waiting for Android to finish booting...")
        self.wait_for_boot(timeout=120)
        self._ensure_screen_geometry()

        self.logger.change_status("Launching Clash Royale...")
        self.start_app_when_ready(CLASH_ROYALE_PACKAGE)
        time.sleep(5)

        self.logger.change_status("Waiting for Clash Royale main menu...")
        deadline = time.time() + 240
        while time.time() < deadline:
            if check_if_on_clash_main_menu(self):
                self.logger.change_status("Clash Royale main menu detected")
                self.logger.log(f"MuMu restart completed in {time.time() - start_ts:.1f}s")
                return True
            if check_if_on_clan_voyage(self):
                self.logger.change_status("Closing clan voyage page")
                self.click(*CLAN_VOYAGE_CLOSE_BUTTON_COORDS)
                time.sleep(2)
                continue
            self.click(*BOOT_MAIN_MENU_DEADSPACE_COORD)
            time.sleep(1)

        raise EmulatorNotReadyError("Timed out waiting for the Clash Royale main menu on MuMu")
