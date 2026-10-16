"""Post session summaries to a user-configured Discord webhook.

Outbound HTTPS only (stdlib ``urllib``); nothing here touches the user's Discord
account. Sends run on a daemon thread so the GUI never blocks on the network.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

from pyclashbot.utils.session_control import total_battles
from pyclashbot.utils.versioning import __version__

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

logger = logging.getLogger(__name__)

_WEBHOOK_URL_RE = re.compile(
    r"^https://(?:(?:ptb|canary)\.)?discord(?:app)?\.com/api/webhooks/\d+/[A-Za-z0-9_-]+$",
)
_TIMEOUT_S = 10
_COLOR_START = 0x3BA55D
_COLOR_END = 0x5865F2
_COLOR_ERROR = 0xED4245


def is_valid_webhook_url(url: str | None) -> bool:
    return bool(url) and _WEBHOOK_URL_RE.match(url.strip()) is not None


def _stat(stats: Mapping[str, Any], key: str, default: Any = 0) -> Any:
    value = stats.get(key, default)
    return default if value in (None, "") else value


def build_session_embed(
    event: str, stats: Mapping[str, Any] | None, reason: str | None = None, emulator: str | None = None
) -> dict[str, Any]:
    """Build the webhook JSON payload for a session ``event`` ("start", "end" or "error")."""
    stats = stats or {}
    title = {"start": "Bot started", "end": "Session finished", "error": "Bot stopped with an error"}.get(
        event, "py-clash-bot"
    )
    color = {"start": _COLOR_START, "error": _COLOR_ERROR}.get(event, _COLOR_END)

    fields: list[dict[str, Any]] = []
    if emulator:
        fields.append({"name": "Emulator", "value": str(emulator), "inline": True})
    if event != "start":
        wins, losses = int(_stat(stats, "wins")), int(_stat(stats, "losses"))
        fields += [
            {"name": "Battles", "value": str(total_battles(stats)), "inline": True},
            {"name": "Wins / Losses", "value": f"{wins} / {losses}", "inline": True},
            {"name": "Win rate", "value": str(_stat(stats, "winrate", "0%")), "inline": True},
            {"name": "Cards played", "value": str(_stat(stats, "cards_played")), "inline": True},
            {"name": "Upgrades", "value": str(_stat(stats, "upgrades")), "inline": True},
            {"name": "Donations", "value": str(_stat(stats, "cards_donated")), "inline": True},
            {"name": "Runtime", "value": str(_stat(stats, "time_since_start", "00:00:00")), "inline": True},
            {"name": "Recovery restarts", "value": str(_stat(stats, "restarts_after_failure")), "inline": True},
        ]

    embed: dict[str, Any] = {
        "title": title,
        "color": color,
        "fields": fields,
        "footer": {"text": f"py-clash-bot {__version__}"},
    }
    if reason:
        embed["description"] = reason[:2000]
    return {"username": "py-clash-bot", "embeds": [embed], "allowed_mentions": {"parse": []}}


def post_webhook(url: str, payload: Mapping[str, Any]) -> tuple[bool, str]:
    """POST ``payload`` to ``url`` synchronously. Returns (ok, message)."""
    if not is_valid_webhook_url(url):
        return False, "Not a Discord webhook URL"
    request = urllib.request.Request(
        url.strip(),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": f"py-clash-bot/{__version__}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
            return 200 <= response.status < 300, f"HTTP {response.status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return False, str(getattr(e, "reason", e))


def post_webhook_async(
    url: str, payload: Mapping[str, Any], on_done: Callable[[bool, str], None] | None = None
) -> threading.Thread:
    """Send on a daemon thread; ``on_done`` (if given) runs on that thread."""

    def _send() -> None:
        ok, message = post_webhook(url, payload)
        if not ok:
            logger.warning("Discord webhook failed: %s", message)
        if on_done is not None:
            on_done(ok, message)

    thread = threading.Thread(target=_send, name="discord-webhook", daemon=True)
    thread.start()
    return thread
