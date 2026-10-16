"""Offline coverage for the Discord session webhook (no network)."""

import pytest

from pyclashbot.interface.enums import StatField
from pyclashbot.utils import discord_webhook
from pyclashbot.utils.discord_webhook import build_session_embed, is_valid_webhook_url, post_webhook


@pytest.mark.parametrize(
    "url",
    [
        "https://discord.com/api/webhooks/123456789/abcDEF_-123",
        "https://discordapp.com/api/webhooks/1/x",
        "https://ptb.discord.com/api/webhooks/1/x",
        "  https://canary.discord.com/api/webhooks/1/x  ",
    ],
)
def test_valid_webhook_urls(url):
    assert is_valid_webhook_url(url)


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "http://discord.com/api/webhooks/1/x",
        "https://evil.example/api/webhooks/1/x",
        "https://discord.com.evil.example/api/webhooks/1/x",
        "https://discord.com/api/webhooks/abc/x",
        "https://discord.com/api/webhooks/1/x?wait=true",
    ],
)
def test_invalid_webhook_urls(url):
    assert not is_valid_webhook_url(url)


def test_start_embed_has_no_stats():
    payload = build_session_embed("start", {"wins": 9}, emulator="MEmu")
    embed = payload["embeds"][0]
    assert embed["title"] == "Bot started"
    assert [f["name"] for f in embed["fields"]] == ["Emulator"]
    assert payload["allowed_mentions"] == {"parse": []}


def test_end_embed_summarises_stats():
    stats = {
        StatField.CLASSIC_1V1_FIGHTS.value: 4,
        StatField.TROPHY_ROAD_1V1_FIGHTS.value: 2,
        "wins": 4,
        "losses": 2,
        "winrate": "66.7%",
        "time_since_start": "01:02:03",
    }
    embed = build_session_embed("end", stats, reason="played 6 battles")["embeds"][0]
    fields = {f["name"]: f["value"] for f in embed["fields"]}
    assert fields["Battles"] == "6"
    assert fields["Wins / Losses"] == "4 / 2"
    assert fields["Win rate"] == "66.7%"
    assert fields["Runtime"] == "01:02:03"
    assert embed["description"] == "played 6 battles"


def test_error_embed_is_red_and_tolerates_missing_stats():
    embed = build_session_embed("error", None, reason="x" * 5000)["embeds"][0]
    assert embed["color"] == discord_webhook._COLOR_ERROR
    assert len(embed["description"]) == 2000


def test_post_webhook_rejects_bad_url_without_network(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("must not hit the network")

    monkeypatch.setattr(discord_webhook.urllib.request, "urlopen", boom)
    assert post_webhook("https://example.com/hook", {}) == (False, "Not a Discord webhook URL")
