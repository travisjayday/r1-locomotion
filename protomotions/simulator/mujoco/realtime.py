# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Wall-clock pacing for client-driven MuJoCo passive viewers."""

from __future__ import annotations

import time
from collections.abc import Callable


class RealTimePacer:
    """Synchronize monotonically increasing simulation time to wall time."""

    def __init__(
        self,
        *,
        real_time_factor: float = 1.0,
        max_lag_s: float = 0.1,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if real_time_factor <= 0.0:
            raise ValueError("real_time_factor must be positive")
        if max_lag_s < 0.0:
            raise ValueError("max_lag_s must be non-negative")
        self.real_time_factor = real_time_factor
        self.max_lag_s = max_lag_s
        self._clock = clock
        self._sleeper = sleeper
        self._wall_origin: float | None = None
        self._sim_origin: float | None = None

    def reset(self, sim_time: float) -> None:
        self._wall_origin = self._clock()
        self._sim_origin = float(sim_time)

    def wait(self, sim_time: float) -> None:
        sim_time = float(sim_time)
        if (
            self._wall_origin is None
            or self._sim_origin is None
            or sim_time < self._sim_origin
        ):
            self.reset(sim_time)
            return

        target_wall = self._wall_origin + (
            sim_time - self._sim_origin
        ) / self.real_time_factor
        delay = target_wall - self._clock()
        if delay > 0.0:
            self._sleeper(delay)
        elif delay < -self.max_lag_s:
            # Do not accelerate indefinitely after a debugger pause, expensive
            # reset, or temporarily overloaded frame. Start a fresh epoch.
            self.reset(sim_time)


__all__ = ["RealTimePacer"]
