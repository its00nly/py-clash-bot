"""BlueStacks instance selection: picker listing, user-picked instance, shared-port guard."""

from types import SimpleNamespace

import pytest

from pyclashbot.emulators import bluestacks as bs
from pyclashbot.emulators.base import EmulatorNotReadyError

CONF = """bst.installed_images="Pie64,Tiramisu64"
bst.instance.Pie64.adb_port="5555"
bst.instance.Pie64.display_name="BlueStacks App Player"
bst.instance.Pie64.status.adb_port="5555"
bst.instance.Tiramisu64.adb_port="5555"
bst.instance.Tiramisu64.display_name="pyclashbot-136"
bst.instance.Tiramisu64.status.adb_port="5555"
"""


@pytest.fixture
def conf_path(tmp_path, monkeypatch):
    path = tmp_path / "bluestacks.conf"
    path.write_text(CONF, encoding="utf-8")
    monkeypatch.setattr(
        bs.BlueStacksEmulatorController, "_find_config_paths", staticmethod(lambda: (str(path), "missing.json"))
    )
    return str(path)


def _bare(instance: str | None, conf_path: str, sdk_reply: str = "") -> bs.BlueStacksEmulatorController:
    c = object.__new__(bs.BlueStacksEmulatorController)
    c._user_instance = bool(instance)
    c.instance_name = instance or bs.DEFAULT_INSTANCE_NAME
    c.internal_name = None
    c.instance_port = None
    c.bs_conf_path = conf_path
    c.mim_meta_path = "missing.json"
    c.adb = lambda cmd: SimpleNamespace(returncode=0, stdout=sdk_reply)
    return c


def test_list_instances(conf_path):
    assert bs.BlueStacksEmulatorController.list_instances() == ["BlueStacks App Player", "pyclashbot-136"]


def test_user_picked_instance_resolves_without_rename(conf_path):
    c = _bare("BlueStacks App Player", conf_path)
    c._ensure_managed_instance()
    assert c.internal_name == "Pie64"
    assert c.instance_port == 5555
    assert c.instance_name == "BlueStacks App Player"


def test_unknown_user_instance_errors_with_choices(conf_path):
    c = _bare("Nope", conf_path)
    with pytest.raises(EmulatorNotReadyError, match="BlueStacks App Player"):
        c._ensure_managed_instance()


@pytest.mark.parametrize(
    ("internal", "reply", "ok"),
    [
        ("Tiramisu64", "33", True),
        ("Tiramisu64", "28", False),  # Pie instance answered on the shared port
        ("Pie64_2", "28", True),
        ("Pie64", "garbage", True),  # can't tell -> don't block
        ("Mystery64", "28", True),  # unknown image -> don't block
    ],
)
def test_connected_instance_matches(conf_path, internal, reply, ok):
    c = _bare(None, conf_path, sdk_reply=reply)
    c.internal_name = internal
    assert c._connected_instance_matches() is ok
