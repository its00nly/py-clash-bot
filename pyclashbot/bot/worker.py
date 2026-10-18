import datetime as dt
import time
import traceback
from multiprocessing import Process, Queue
from multiprocessing.synchronize import Event
from typing import Any, Protocol

from pyclashbot.bot.states import StateHistory, StateOrder, state_tree
from pyclashbot.emulators import EmulatorType, get_emulator_registry
from pyclashbot.emulators.base import EmulatorNotReadyError
from pyclashbot.interface.enums import UIField
from pyclashbot.utils.logger import ProcessLogger, attach_worker_file_logging
from pyclashbot.utils.platform import is_macos
from pyclashbot.utils.session_control import SessionSettings, is_active_now, limit_reached, seconds_until_active

# Pacing decisions returned by WorkerProcess._pace() between states.
PACE_CONTINUE = "continue"
PACE_STOP = "stop"
PACE_RELAUNCH = "relaunch"


class WorkerProcess(Process):
    """Worker process for running the bot.

    Uses multiprocessing instead of threading for reliable force termination.
    Communicates stats back to main process via Queue.
    """

    def __init__(
        self,
        jobs: dict[str, Any],
        stats_queue: Queue,
        shutdown_event: Event,
        session_log_path: str,
        pause_event: Event | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.jobs = jobs
        self.stats_queue = stats_queue
        self.shutdown_event = shutdown_event
        self.pause_event = pause_event
        self.session_log_path = session_log_path
        self.session = SessionSettings.from_jobs(jobs)

    def _startup_countdown(self, logger: ProcessLogger) -> bool:
        """Hold off launching the emulator for the configured delay (#752).

        Returns False if the bot was stopped during the countdown.
        """
        remaining = self.session.startup_delay_s
        while remaining > 0:
            if remaining <= 10 or remaining % 10 == 0:
                logger.change_status(f"Starting the emulator in {remaining}s...")
            if self.shutdown_event.wait(1):
                return False
            remaining -= 1
        return True

    def _wait_while_paused(self, logger: ProcessLogger) -> None:
        if self.pause_event is None or not self.pause_event.is_set():
            return
        logger.change_status("Paused - press Resume to continue")
        while self.pause_event.is_set() and not self.shutdown_event.is_set():
            self.shutdown_event.wait(0.5)
        if not self.shutdown_event.is_set():
            logger.change_status("Resumed")

    def _wait_for_active_hours(self) -> bool:
        """Sleep until the active window opens. Returns False if the bot was stopped."""
        while not is_active_now(self.session, dt.datetime.now()):
            wait_s = min(60.0, max(1.0, seconds_until_active(self.session, dt.datetime.now())))
            if self.shutdown_event.wait(wait_s):
                return False
        return True

    def _pace(self, emulator, logger: ProcessLogger) -> str:
        """Apply pause, session limits and the active-hours window between states."""
        self._wait_while_paused(logger)
        if self.shutdown_event.is_set():
            return PACE_STOP

        started = logger.start_time or time.time()
        reason = limit_reached(self.session, logger.get_stats(), time.time() - started)
        if reason is not None:
            logger.change_status(f"Session limit reached ({reason}) - stopping")
            return PACE_STOP

        if is_active_now(self.session, dt.datetime.now()):
            return PACE_CONTINUE

        resume_at = f"{self.session.active_start_hour:02d}:00"
        logger.change_status(f"Outside active hours - closing the emulator until {resume_at}")
        try:
            emulator.stop()
        except Exception as e:
            logger.log(f"Could not stop the emulator for the off-hours break: {e}")
        if not self._wait_for_active_hours():
            return PACE_STOP
        logger.change_status("Active hours started - relaunching the emulator")
        return PACE_RELAUNCH

    def _setup_emulator(self, jobs: dict[str, Any], logger: ProcessLogger):
        """Set up the appropriate emulator based on job configuration."""
        emulator_selection = jobs.get("emulator", EmulatorType.MEMU)
        registry = get_emulator_registry()

        if emulator_selection not in registry:
            print(f"[!] Fatal error: Emulator {emulator_selection} is not supported on this platform!")
            logger.change_status(f"Emulator {emulator_selection} is not available on this platform.")
            return None

        controller_class = registry[emulator_selection]

        try:
            if emulator_selection == EmulatorType.GOOGLE_PLAY:
                print("Creating Google Play emulator")
                gp_device_serial = jobs.get(UIField.GP_DEVICE_SERIAL.value) or None
                emu = controller_class(logger=logger, device_serial=gp_device_serial)

            elif emulator_selection == EmulatorType.BLUESTACKS:
                print("Creating BlueStacks 5 emulator")
                # Default: Vulkan on macOS, OpenGL on Windows
                default_mode = "vlcn" if is_macos() else "gl"
                bs_mode = jobs.get("bluestacks_render_mode", default_mode)
                render_settings = {"graphics_renderer": bs_mode}
                bs_device_serial = jobs.get(UIField.BS_DEVICE_SERIAL.value) or None
                bs_instance = jobs.get(UIField.BS_INSTANCE.value) or None
                emu = controller_class(
                    logger=logger,
                    render_settings=render_settings,
                    device_serial=bs_device_serial,
                    instance_name=bs_instance,
                )

            elif emulator_selection == EmulatorType.MEMU:
                print("Creating MEmu emulator")
                from pyclashbot.emulators.memu import verify_memu_installation

                if not verify_memu_installation():
                    logger.change_status("MEmu is not installed! Please install it to use MEmu Emulator Mode")
                    return None
                render_mode = jobs.get("memu_render_mode", "opengl")
                emu = controller_class(logger, render_mode)

            elif emulator_selection == EmulatorType.MUMU:
                print("Creating MuMu Player 12 emulator")
                instance_index = int(jobs.get(UIField.MUMU_INSTANCE_INDEX.value, 0) or 0)
                emu = controller_class(logger=logger, instance_index=instance_index)

            elif emulator_selection == EmulatorType.ADB:
                print("Creating ADB Device controller")
                adb_serial = jobs.get(UIField.ADB_SERIAL.value) or None
                emu = controller_class(logger=logger, device_serial=adb_serial)

            else:
                return None

            # Construction is cheap; restart() boots the emulator and launches Clash.
            # First-boot failure stops the bot — there is no setup retry.
            emu.restart()
            return emu

        except EmulatorNotReadyError as e:
            print(f"{emulator_selection} emulator is not ready: {e}")
            logger.change_status(f"{emulator_selection} is not ready — fix it and start the bot again.")
            return None
        except Exception as e:
            print(f"Failed to create {emulator_selection} emulator: {e}")
            logger.change_status(f"Failed to start {emulator_selection}. Verify its installation!")
            return None

    def _run_bot_loop(self, emulator, jobs: dict[str, Any], logger: ProcessLogger) -> None:
        """Run the main bot state loop."""
        state = "start"
        state_history = StateHistory(logger)
        state_order = StateOrder()
        consecutive_restarts = 0
        max_consecutive_restarts = 5

        while not self.shutdown_event.is_set():
            pace = self._pace(emulator, logger)
            if pace == PACE_STOP:
                break
            if pace == PACE_RELAUNCH:
                try:
                    emulator.restart()
                    state = "start"
                except EmulatorNotReadyError as e:
                    logger.log(f"Relaunch after off-hours break failed: {e}")
                    state = "restart"

            try:
                new_state = state_tree(emulator, logger, state, jobs, state_history, state_order)

                # Check for restart loops
                if new_state == "restart":
                    consecutive_restarts += 1
                    if consecutive_restarts >= max_consecutive_restarts:
                        logger.error(
                            f"Too many consecutive restarts ({consecutive_restarts}) - stopping bot to prevent infinite loop"
                        )
                        break
                    logger.log(f"Restart #{consecutive_restarts} - attempting to recover")
                else:
                    consecutive_restarts = 0  # Reset counter on successful state

                # Check for error states that should stop execution
                if new_state in ["fail", None]:
                    logger.error(f"Critical error: state_tree returned '{new_state}' - stopping bot")
                    if new_state == "fail":
                        logger.add_restart_after_failure()
                    break

                state = new_state

            except Exception as e:
                logger.error(f"Exception in state_tree: {e}")
                logger.log(f"Current state was: {state}")
                print(f"[ERROR] Exception in state_tree: {e}")
                print(f"[ERROR] Current state was: {state}")
                # Try to restart from a known state
                state = "restart"
                # If we keep getting exceptions, break out
                traceback.print_exc()
                consecutive_restarts += 1
                if consecutive_restarts >= max_consecutive_restarts:
                    logger.error("Too many consecutive exceptions - stopping bot")
                    break

    def run(self) -> None:
        """Main worker process execution."""
        print("WorkerProcess run()...")
        attach_worker_file_logging(self.session_log_path)

        # Create logger that sends stats through queue
        logger = ProcessLogger(self.stats_queue)

        try:
            if not self._startup_countdown(logger):
                return
            # Respect the schedule before booting: don't launch outside active hours.
            if not is_active_now(self.session, dt.datetime.now()):
                logger.change_status(
                    f"Outside active hours - waiting until {self.session.active_start_hour:02d}:00 to start"
                )
                if not self._wait_for_active_hours():
                    return
            emulator = self._setup_emulator(self.jobs, logger)
            if emulator is None:
                return

            self._run_bot_loop(emulator, self.jobs, logger)
        except Exception as err:
            logger.error(str(err))
            traceback.print_exc()
        finally:
            logger.change_status("Bot stopped")


class _StoppableProcess(Protocol):
    """Structural interface for the process-lifecycle methods ``stop_worker_process`` needs.

    Satisfied by ``multiprocessing.Process`` (and ``WorkerProcess``), spawned context
    processes, and test fakes alike — so the helper depends on the contract, not the
    concrete class.
    """

    def is_alive(self) -> bool: ...
    def terminate(self) -> None: ...
    def join(self, timeout: float | None = ...) -> None: ...
    def kill(self) -> None: ...


class _Signal(Protocol):
    """Structural interface for the shutdown event — only ``set()`` is used here."""

    def set(self) -> None: ...


def stop_worker_process(
    process: _StoppableProcess | None,
    shutdown_event: _Signal | None = None,
    *,
    graceful_timeout: float = 2.0,
) -> None:
    """Stop the worker process via OS-level termination.

    Signals shutdown (belt-and-suspenders), then terminates and hard-kills if the
    process does not exit within ``graceful_timeout``. The OS interrupts any blocking
    call in the worker (sleeps, ADB/socket waits), so this does not depend on the
    worker cooperatively noticing the shutdown event.
    """
    if shutdown_event is not None:
        shutdown_event.set()
    if process is None or not process.is_alive():
        return
    process.terminate()  # SIGTERM / TerminateProcess
    process.join(timeout=graceful_timeout)
    if process.is_alive():
        process.kill()  # SIGKILL
        process.join(timeout=graceful_timeout)
