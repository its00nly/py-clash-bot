"""GUI layout: settings round-trip, sidebar navigation, last-battle summary and the embedded debug panel.

The window is built for real but withdrawn; tests skip when no Tk display is available.
"""

from __future__ import annotations

import json
import os
import time
import tkinter as tk
from pathlib import Path

import pytest

from pyclashbot.emulators import EmulatorType, get_available_emulators
from pyclashbot.interface import debug_window
from pyclashbot.interface.config import GOOGLE_PLAY_SETTINGS, deck_hero_card_options
from pyclashbot.interface.enums import UIField

# Derived in __main__.make_job_dictionary from the three clan-action toggles, never a widget.
_DERIVED_FIELDS = {UIField.CLAN_CHAT_USER_TOGGLE}


@pytest.fixture(scope="module")
def ui(tmp_path_factory: pytest.TempPathFactory):
    appdata = tmp_path_factory.mktemp("appdata")
    with pytest.MonkeyPatch.context() as mp:
        # Keep the real user's settings / logs out of reach of the window under test.
        mp.setenv("APPDATA", str(appdata))
        mp.setenv("LOCALAPPDATA", str(appdata / "local"))
        mp.delenv("PYCLASHBOT_UI_START_PAGE", raising=False)
        from pyclashbot.interface.ui import PyClashBotUI

        try:
            window = PyClashBotUI()
        except tk.TclError as exc:  # no display
            pytest.skip(f"Tk unavailable: {exc}")
        window.withdraw()
        window.update()
        try:
            yield window
        finally:
            window.destroy()


# ---- settings round-trip ------------------------------------------------------------


def _changed_values(ui) -> dict[str, object]:
    """Every value flipped / moved off its default so a dropped key can't pass by accident."""
    values = ui.get_all_values()
    changed: dict[str, object] = {}
    for key, value in values.items():
        if isinstance(value, bool):
            changed[key] = not value
        elif isinstance(value, int):
            changed[key] = value + 1
        else:
            changed[key] = value

    available = list(get_available_emulators())
    current = ui.emulator_var.get()
    target = next((e for e in available if e != current), current)
    emulator_keys = {
        EmulatorType.MEMU: UIField.MEMU_EMULATOR_TOGGLE,
        EmulatorType.GOOGLE_PLAY: UIField.GOOGLE_PLAY_EMULATOR_TOGGLE,
        EmulatorType.BLUESTACKS: UIField.BLUESTACKS_EMULATOR_TOGGLE,
        EmulatorType.ADB: UIField.ADB_TOGGLE,
        EmulatorType.MUMU: UIField.MUMU_EMULATOR_TOGGLE,
    }
    for emulator, field in emulator_keys.items():
        changed[field.value] = emulator == target

    # Radio pairs/triples: exactly one renderer selected.
    changed[UIField.DIRECTX_TOGGLE.value] = not values[UIField.DIRECTX_TOGGLE.value]
    changed[UIField.OPENGL_TOGGLE.value] = not changed[UIField.DIRECTX_TOGGLE.value]
    bs = [UIField.BS_RENDERER_DX, UIField.BS_RENDERER_GL, UIField.BS_RENDERER_VK]
    current_bs = next((i for i, f in enumerate(bs) if values[f.value]), 0)
    for i, field in enumerate(bs):
        changed[field.value] = i == (current_bs + 1) % len(bs)

    for config in GOOGLE_PLAY_SETTINGS:
        options = [str(o) for o in config.values]
        now = str(values[config.key.value])
        changed[config.key.value] = next((o for o in options if o != now), now)

    heroes = deck_hero_card_options()
    hero_now = values[UIField.DECK_HERO_CARD.value]
    changed[UIField.DECK_HERO_CARD.value] = next((h for h in heroes if h != hero_now), hero_now)

    changed[UIField.ADB_SERIAL.value] = "127.0.0.1:5555"
    changed[UIField.GP_DEVICE_SERIAL.value] = "emulator-5556"
    changed[UIField.BS_DEVICE_SERIAL.value] = "127.0.0.1:5565"
    changed[UIField.BS_INSTANCE.value] = "BlueStacks App Player"
    changed[UIField.THEME_NAME.value] = "superhero"
    changed[UIField.RECORDING_FOLDER_PATH.value] = str(Path(os.environ.get("APPDATA", ".")) / "recs")
    changed[UIField.DISCORD_WEBHOOK_URL.value] = "https://example.invalid/webhook"
    changed[UIField.ACTIVE_START_HOUR.value] = 7
    changed[UIField.ACTIVE_END_HOUR.value] = 21
    changed[UIField.DECK_NUMBER_SELECTION.value] = 3
    changed[UIField.MAX_DECK_SELECTION.value] = 4
    changed[UIField.MAX_ACCOUNT_SELECTION.value] = 3
    return changed


def test_get_all_values_covers_every_ui_field(ui) -> None:
    keys = set(ui.get_all_values())
    expected = {field.value for field in UIField if field not in _DERIVED_FIELDS}
    assert expected - keys == set()


def test_set_get_round_trip_every_ui_field(ui) -> None:
    original = ui.get_all_values()
    changed = _changed_values(ui)
    try:
        ui.set_all_values(changed)
        ui.update()
        after = ui.get_all_values()
        mismatched = {k: (changed[k], after.get(k)) for k in changed if after.get(k) != changed[k]}
        assert mismatched == {}
        assert ui._style.theme_use() == "superhero"
    finally:
        ui.set_all_values(original)
        ui.update()


# ---- sidebar navigation -------------------------------------------------------------


def test_sidebar_switches_pages(ui) -> None:
    from pyclashbot.interface.ui import PAGE_DASHBOARD, PAGES

    names = [name for name, _label, _glyph in PAGES]
    assert ui.current_page == PAGE_DASHBOARD
    for name in [*names, PAGE_DASHBOARD]:
        ui.show_page(name)
        ui.update_idletasks()
        assert ui.current_page == name
        for other, page in ui._pages.items():
            assert (page.winfo_manager() == "grid") == (other == name), other
        for other, button in ui._nav_buttons.items():
            assert button.active == (other == name), other

    ui.show_page("no-such-page")
    assert ui.current_page == PAGE_DASHBOARD


def test_nav_button_click_selects_page(ui) -> None:
    from pyclashbot.interface.ui import PAGE_DASHBOARD, PAGE_JOBS

    ui._nav_buttons[PAGE_JOBS].event_generate("<Button-1>")
    ui.update_idletasks()
    assert ui.current_page == PAGE_JOBS
    ui.show_page(PAGE_DASHBOARD)


def test_header_status_pill_follows_state(ui) -> None:
    ui.set_button_state("running")
    assert ui.status_state == "Running"
    ui.set_paused(True)
    assert ui.status_state == "Paused"
    ui.set_button_state("idle")
    assert ui.status_state == "Idle"


# ---- last battle summary ------------------------------------------------------------


def _write_log(path: Path, events: list[dict | str]) -> None:
    lines = [e if isinstance(e, str) else json.dumps(e) for e in events]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


_END = {
    "event": "end",
    "outcome": "ended",
    "t": 1000.0,
    "duration": 182.4,
    "frames": 900,
    "plays": 30,
    "deployed": 29,
    "not_deployed": 1,
    "ability_fired": 2,
}


def test_read_last_battle_missing_dir_returns_none(tmp_path: Path) -> None:
    assert debug_window.read_last_battle(str(tmp_path / "nope")) is None
    assert debug_window.read_last_battle(str(tmp_path)) is None


def test_read_last_battle_parses_end_and_skips_garbage(tmp_path: Path) -> None:
    _write_log(
        tmp_path / "2026-10-17_10-00-00.jsonl",
        [{"event": "start", "t": 800.0}, "{not json", _END, ""],
    )
    summary = debug_window.read_last_battle(str(tmp_path))
    assert summary is not None
    assert summary["plays"] == 30
    assert summary["outcome"] == "ended"
    assert summary["newer_unfinished"] is False
    assert summary["file"] == "2026-10-17_10-00-00.jsonl"


def test_read_last_battle_falls_back_past_unfinished_newest(tmp_path: Path) -> None:
    _write_log(tmp_path / "2026-10-17_10-00-00.jsonl", [{"event": "start"}, _END])
    _write_log(tmp_path / "2026-10-17_11-00-00.jsonl", [{"event": "start"}, {"event": "play"}])
    summary = debug_window.read_last_battle(str(tmp_path))
    assert summary is not None
    assert summary["file"] == "2026-10-17_10-00-00.jsonl"
    assert summary["newer_unfinished"] is True


def test_format_last_battle_rows() -> None:
    summary = {**_END, "outcome": "restart", "file": "x.jsonl", "newer_unfinished": False}
    rows = dict(debug_window.format_last_battle(summary, now=1000.0 + 120))
    assert set(rows) == set(debug_window.LAST_BATTLE_LABELS)
    assert rows["Outcome"] == "Restarted (recovery)"
    assert rows["Duration"] == "3:02"
    assert rows["Deployed"] == "29 (97%)"
    assert rows["Not deployed"] == "1"
    assert "2 min" in rows["When"]


def test_format_last_battle_tolerates_missing_fields() -> None:
    rows = dict(debug_window.format_last_battle({"event": "end"}))
    assert set(rows) == set(debug_window.LAST_BATTLE_LABELS)


def test_ui_last_battle_card_reads_decisions_dir(ui, tmp_path: Path) -> None:
    from pyclashbot.interface.ui import PAGE_BATTLE_AI, PAGE_DASHBOARD

    _write_log(tmp_path / "2026-10-17_10-00-00.jsonl", [{**_END, "t": time.time() - 60}])
    old_dir = ui.decisions_dir
    ui.decisions_dir = str(tmp_path)
    try:
        ui.show_page(PAGE_BATTLE_AI)
        ui._refresh_last_battle(force=True)
        assert ui.last_battle_summary is not None
        assert ui.last_battle_summary["plays"] == 30
    finally:
        ui.decisions_dir = old_dir
        ui.show_page(PAGE_DASHBOARD)


# ---- embedded debug panel -----------------------------------------------------------


def test_debug_panel_handles_missing_and_bad_files(ui, tmp_path: Path) -> None:
    panel = debug_window.DebugPanel(ui, directory=str(tmp_path / "missing"), poll_ms=10_000)
    try:
        panel.poll_once()
        assert "Waiting for frames" in panel.text.get("1.0", "end")

        tmp_path.joinpath("latest.jpg").write_bytes(b"not a jpeg")
        tmp_path.joinpath("latest.json").write_text("{broken", encoding="utf-8")
        panel.directory = str(tmp_path)
        panel.poll_once()  # must not raise
        assert panel.alive()
    finally:
        panel.destroy()
    assert not panel.alive()
