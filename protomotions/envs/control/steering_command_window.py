# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone keyboard window for interactive steering inference.

The simulator viewers bind many ordinary letter keys to cameras and tools. This
small Tk window owns steering input instead, so commands are only sent while the
dedicated window has focus. Tk runs entirely on its own thread; the simulation
thread exchanges plain action names and status dictionaries through queues.
"""

from __future__ import annotations

from queue import Empty, Full, Queue, SimpleQueue
import threading
from typing import Mapping, Optional


class SteeringCommandWindow:
    """Collect steering keys and display the latest target command."""

    _KEYSYM_ACTIONS = {
        "w": "move_forward",
        "s": "move_backward",
        "a": "move_left",
        "d": "move_right",
        "q": "face_left",
        "e": "face_right",
        "x": "align_facing",
        "space": "stop",
        "minus": "slower",
        "underscore": "slower",
        "equal": "faster",
        "plus": "faster",
    }

    def __init__(
        self,
        *,
        title: str = "ProtoMotions Steering",
        refresh_interval_ms: int = 50,
        startup_timeout_s: float = 5.0,
    ) -> None:
        if refresh_interval_ms <= 0:
            raise ValueError("refresh_interval_ms must be positive")
        if startup_timeout_s <= 0.0:
            raise ValueError("startup_timeout_s must be positive")

        self._title = title
        self._refresh_interval_ms = refresh_interval_ms
        self._actions: SimpleQueue[str] = SimpleQueue()
        self._status: Queue[Mapping[str, object]] = Queue(maxsize=1)
        self._ready = threading.Event()
        self._stop_requested = threading.Event()
        self._closed = threading.Event()
        self._startup_error: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._run,
            name="protomotions-steering-window",
            daemon=True,
        )
        self._thread.start()

        if not self._ready.wait(timeout=startup_timeout_s):
            self._stop_requested.set()
            raise RuntimeError(
                "Timed out while opening the standalone steering window"
            )
        if self._startup_error is not None:
            raise RuntimeError(
                "Could not open the standalone steering window. Ensure a graphical "
                "display is available and Tk support is installed."
            ) from self._startup_error

    @classmethod
    def action_for_keysym(cls, keysym: str) -> Optional[str]:
        """Translate a Tk key symbol to a steering action name."""
        return cls._KEYSYM_ACTIONS.get(keysym.lower())

    def poll_actions(self) -> list[str]:
        """Return and remove all actions queued since the previous poll."""
        actions = []
        while True:
            try:
                actions.append(self._actions.get_nowait())
            except Empty:
                return actions

    def update_status(self, status: Mapping[str, object]) -> None:
        """Publish status without ever blocking the simulation thread."""
        if self._closed.is_set():
            return
        try:
            self._status.put_nowait(dict(status))
            return
        except Full:
            pass
        try:
            self._status.get_nowait()
        except Empty:
            pass
        try:
            self._status.put_nowait(dict(status))
        except Full:
            pass

    def close(self) -> None:
        """Ask the UI thread to destroy its window and return promptly."""
        self._stop_requested.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)

    def _queue_action(self, action: str) -> None:
        self._actions.put(action)

    def _handle_key(self, event) -> None:
        action = self.action_for_keysym(event.keysym)
        if action is not None:
            self._queue_action(action)
            return "break"
        return None

    @staticmethod
    def format_status(status: Mapping[str, object]) -> str:
        """Format target values for the status label."""
        lines = [
            f"Environment: {int(status.get('env_id', 0))}",
            f"Mode: {status.get('mode', 'steering')}",
            "",
            "Target command",
            f"  local vx: {float(status.get('local_vx', 0.0)):+.2f} m/s",
            f"  local vy: {float(status.get('local_vy', 0.0)):+.2f} m/s",
            f"  speed:    {float(status.get('speed', 0.0)):.2f} m/s",
            f"  yaw rate: {float(status.get('yaw_rate', 0.0)):+.2f} rad/s",
            f"  move heading: {float(status.get('move_heading_deg', 0.0)):+.1f} deg",
            f"  face heading: {float(status.get('face_heading_deg', 0.0)):+.1f} deg",
        ]
        return "\n".join(lines)

    def _drain_latest_status(self) -> Optional[Mapping[str, object]]:
        latest = None
        while True:
            try:
                latest = self._status.get_nowait()
            except Empty:
                return latest

    def _run(self) -> None:
        try:
            import tkinter as tk

            root = tk.Tk()
            root.title(self._title)
            root.geometry("460x520")
            root.minsize(420, 480)
            root.configure(padx=16, pady=14)

            heading = tk.Label(
                root,
                text="ProtoMotions Steering",
                font=("TkDefaultFont", 16, "bold"),
            )
            heading.pack(anchor="w")
            tk.Label(
                root,
                text="Click this window, then use the keys or buttons below.",
            ).pack(anchor="w", pady=(2, 10))

            buttons = tk.Frame(root)
            buttons.pack(anchor="center", pady=(0, 12))

            def add_button(text: str, action: str, row: int, column: int) -> None:
                tk.Button(
                    buttons,
                    text=text,
                    width=12,
                    height=2,
                    command=lambda: self._queue_action(action),
                ).grid(row=row, column=column, padx=3, pady=3)

            add_button("W  Forward", "move_forward", 0, 1)
            add_button("A  Left", "move_left", 1, 0)
            add_button("S  Backward", "move_backward", 1, 1)
            add_button("D  Right", "move_right", 1, 2)
            add_button("Q  Turn left", "face_left", 2, 0)
            add_button("X  Stop turn", "align_facing", 2, 1)
            add_button("E  Turn right", "face_right", 2, 2)
            add_button("-  Slower", "slower", 3, 0)
            add_button("SPACE  Stop", "stop", 3, 1)
            add_button("+  Faster", "faster", 3, 2)

            status_var = tk.StringVar(
                value=self.format_status({"mode": "waiting for simulation"})
            )
            tk.Label(
                root,
                textvariable=status_var,
                justify="left",
                anchor="nw",
                relief="groove",
                padx=12,
                pady=10,
                font=("TkFixedFont", 11),
            ).pack(fill="both", expand=True)

            root.bind("<KeyPress>", self._handle_key)

            def close_window() -> None:
                self._stop_requested.set()
                root.destroy()

            root.protocol("WM_DELETE_WINDOW", close_window)

            def refresh() -> None:
                if self._stop_requested.is_set():
                    root.destroy()
                    return
                status = self._drain_latest_status()
                if status is not None:
                    status_var.set(self.format_status(status))
                root.after(self._refresh_interval_ms, refresh)

            root.after(self._refresh_interval_ms, refresh)
            root.after(100, root.focus_force)
            self._ready.set()
            root.mainloop()
        except BaseException as exc:
            self._startup_error = exc
        finally:
            self._ready.set()
            self._closed.set()


__all__ = ["SteeringCommandWindow"]
