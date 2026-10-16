"""Offline coverage for the pre-Start battle-mode guard (#744).

Event modes such as Road to Ronin can replace the selected battle mode after
launch; start_fight must not press Start unless the requested mode is selected.
"""

from dataclasses import dataclass, field

import pytest

from pyclashbot.bot import fight
from pyclashbot.bot.coords import START_FIGHT_BUTTON_COORD


class _Logger:
    def __init__(self):
        self.statuses: list[str] = []

    def change_status(self, status):
        self.statuses.append(status)

    def log(self, _message):
        pass


class _Emulator:
    def __init__(self):
        self.clicks: list[tuple[int, int]] = []

    def click(self, x, y):
        self.clicks.append((x, y))


@dataclass
class _State:
    selected: list[bool] = field(default_factory=list)
    select_calls: int = 0


@pytest.fixture
def env(monkeypatch) -> _State:
    state = _State()

    def is_selected(_emulator, _mode):
        return state.selected.pop(0) if state.selected else False

    def select(_emulator, _mode, _logger):
        state.select_calls += 1
        return True

    monkeypatch.setattr(fight, "check_if_on_clash_main_menu", lambda _e: True)
    monkeypatch.setattr(fight, "check_if_battle_mode_is_selected", is_selected)
    monkeypatch.setattr(fight, "select_mode", select)
    monkeypatch.setattr(fight.time, "sleep", lambda _s: None)
    return state


def test_starts_when_mode_already_selected(env):
    env.selected = [True]
    emulator = _Emulator()
    assert fight.start_fight(emulator, _Logger(), "Classic 1v1") is True
    assert emulator.clicks == [START_FIGHT_BUTTON_COORD]
    assert env.select_calls == 0


def test_reselects_when_event_mode_took_over(env):
    env.selected = [False, True]
    emulator = _Emulator()
    assert fight.start_fight(emulator, _Logger(), "Classic 1v1") is True
    assert env.select_calls == 1
    assert START_FIGHT_BUTTON_COORD in emulator.clicks


def test_refuses_to_start_wrong_mode(env):
    env.selected = [False, False]
    emulator = _Emulator()
    assert fight.start_fight(emulator, _Logger(), "Classic 1v1") is False
    assert START_FIGHT_BUTTON_COORD not in emulator.clicks
