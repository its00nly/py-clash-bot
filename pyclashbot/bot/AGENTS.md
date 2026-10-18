# bot/ — automation state machine

`worker.py` runs the process lifecycle and an infinite loop calling `state_tree()` in `states.py`, which sequences jobs (upgrade → card_mastery → select_battle_mode → randomize/cycle deck → fight → end). Each state returns the next state name; a failed state returns `"restart"`, which restarts the emulator. The worker aborts after **5 consecutive restarts**.

## Adding or editing a state / navigation step

- Call a `check_if_on_*` detection (from `state_detect.py`) before acting; loop on `time.sleep()` with an explicit timeout (e.g. `wait_for_clash_main_menu()` waits up to 240s and clears reward popups mid-wait). Handle async popups before proceeding.
- Navigation is consolidated in `nav.py`: use `navigate_main_page(emulator, logger, start, end)` and the `NAV_CLICKS` table rather than ad-hoc click sequences. Deck pages use image matching on `DECK_TABS_REGION`.
- New coordinates go in `coords.py` as named constants — see the resolution warning in the root `AGENTS.md`. `NAV_CLICKS` itself references named constants from `coords.py`; don't inline raw tuples even inside the table.
- Pure detection helpers (`(emulator) → coords | None`) go in `find.py`. Pixel predicates (`check_if_on_*`, `pixel_indicates_*`) go in `state_detect.py`. `find.py` is leaf-only: it never imports from `pyclashbot.bot.*` except `coords` and `state_detect`.
- New feature jobs go in their own `<feature>_state.py` (see `clan_chat_state.py`, `account_switch.py`, `upgrade_state.py` for the shape). Wire into `states.py:state_tree()` last.

## Status messages

User-facing progress goes through `logger.change_status()` (updates the live status line and log). Use `logger.log()` for detail that should not replace the status line.

- **main menu** — home screen; not "clash main".
- **Clash Royale** — game/app name (especially emulator startup).
- **burger menu** — main-menu options overlay.
- Battle modes: **Classic 1v1**, **Classic 2v2**, **Trophy Road**, **Quick Match**.
- Use an em dash (`—`) for breaks: `Not on main menu — cannot start fight`.
- Sentence case; no function names or debug error codes in status text.
- Throttle repetitive wait-loop messages to ~once per second (see `wait_for_battle_start`, `wait_for_elixir`).

## Gotchas

- `states.py` carries **module-level globals** (`mode_used_in_1v1`, `fight_mode_cycle_index`) set in one state and read in later ones; if `mode_used_in_1v1` is `None`, fight/cycle states silently skip.
- `StateHistory` throttles expensive states (upgrade, card_mastery) by randomized time increments — a manually-triggered state won't re-run until its increment elapses.
- Card availability and deck tabs use **image recognition** (`card_detection.py`, `find_image`); elixir/battle/upgrade checks use **single-pixel** sampling — an off-by-one Y breaks them.
- **Card color fingerprints are BGR.** `battle_iar` is raw `emulator.screenshot()` (OpenCV BGR). Regression: `tests/test_card_fingerprint_bgr.py`.
- **Smart battle AI** (`battle_brain.py`, toggle `smart_battle_user_toggle`, default on): `find.find_enemy_units` spots enemy troops on our half by their maroon level badge / pink HP bar, `BattleBrain` picks card + drop spot by role (defend → counter-push → win condition). Colors were measured on `tests/fixtures/card_hand_bgr_regression.png`; add fixtures from new arenas rather than loosening bounds blindly. Round 3: `find_friendly_units` (our light-blue HP bars, wide shape only — tower decorations share the color) marks pushes our troops already fight; `state_detect.read_elixir` feeds elixir discipline (`ELIXIR_COST`, defense reserve, back-tank pushes at 9+, cycling via `DeckTracker`); `OpponentModel` classifies the opponent by push pattern only. Round 4 (`docs/unit-recognition.md`): `unit_classifier` (k-NN on `find.crop_unit` crops) fills `EnemyUnit.kind`. The brain uses it for anti-air, splash, swarm and tank answers, and `OpponentDeck` uses it for the opponent's cards, cycle and named archetype. **No model ships yet** (`detection/unit_model.npz`), so kinds stay `None` and play is unchanged until one is trained from recorded `units/` crops (`python -m pyclashbot.bot.unit_dataset`). Round 5: `card_knowledge.py` is the per-card table (roles, placement, air/ground); the brain infers a per-battle win condition from the deck (never edit global `WIN_CONDITIONS` for that), runs tank-led pushes, sizes defense with `defense_budget`, remembers threats (`note_threat`) and only spells towers to finish. `fight.py` takes ONE screenshot per tick (`FightContext`/`capture_frame`), verifies each play (`record(play, deployed=...)`), fires the Hero ability via `AbilityTrigger`, and writes throttled decision JSONL under `logs/decisions`. `card_detection.identify_hand_cards_scored` uses a margin matcher; UNKNOWN crops go to `hand_crop_sink` (only when fight recording is on). `adb_base` prefers raw `screencap` (off: `PYCLASHBOT_RAW_SCREENCAP=0`).
- **Debug view** (`debug_view.py`, toggle `debug_view_user_toggle`, default off; smart battles only): `play_smart_card` hands each decision tick's already-computed values (frame, elixir, scored hand, units, threat, tower reads, ability, play/hold) to `FightContext.publish_debug` → `debug_view.build_debug_info` → `DebugFrameSink.submit`, which never blocks or raises (size-1 queue, drop-old, ~5 fps). A daemon thread renders `render_debug_frame` (pure, 2x, cv2 only) and writes `<app data>/debug/latest.jpg` + `latest.json` atomically plus a 200-frame ring in `debug/history/NNN.{jpg,json}`. The GUI's `interface/debug_window.py` polls those files. Don't add screenshots or detection passes for it — reuse what the tick already computed.
