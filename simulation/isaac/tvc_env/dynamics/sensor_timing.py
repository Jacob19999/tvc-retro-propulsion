"""Effective sensor timing on the physics grid.

Sensor models advance once per physics substep, so a transport latency is
rounded to whole substeps and an output rate cannot exceed the physics rate.
At the 120 Hz mission step the WTGAHRS1's 5 ms latency becomes 8.3 ms, its
200 Hz output becomes 120 Hz, and the VN-110E's 2 ms latency becomes 0
(Paper/physics_simulation, Sec. XI-B (ii)). This module reports the timing a
model actually delivers and warns when it differs materially from the
datasheet value, so a run record never silently carries the wrong timing.
"""

from __future__ import annotations

import math
import warnings

# Relative latency error above which a run is warned (and absolute floor, s).
LATENCY_TOLERANCE = 0.25
LATENCY_FLOOR_S = 1e-4


def effective_timing(dt: float, rate_hz: float, latency_s: float) -> dict:
    """Requested and delivered output rate (Hz) and latency (s) on a physics grid of step dt."""
    dt = float(dt)
    delay = int(round(float(latency_s) / dt))
    return dict(requested_rate_hz=float(rate_hz), effective_rate_hz=min(float(rate_hz), 1.0 / dt),
                requested_latency_s=float(latency_s), effective_latency_s=delay * dt, delay_substeps=delay,
                physics_dt_s=dt)


def warn_if_quantized(name: str, timing: dict) -> None:
    """Warn when the physics grid changes a sensor's latency or rate materially."""
    problems = []
    requested, effective = timing["requested_latency_s"], timing["effective_latency_s"]
    if abs(effective - requested) > max(LATENCY_TOLERANCE * requested, LATENCY_FLOOR_S):
        problems.append(f"latency {requested * 1e3:.1f} ms runs as {effective * 1e3:.1f} ms")
    if timing["effective_rate_hz"] < timing["requested_rate_hz"] - 1e-9:
        problems.append(f"output rate {timing['requested_rate_hz']:.0f} Hz runs as {timing['effective_rate_hz']:.0f} Hz")
    if problems:
        hz = 1.0 / timing["physics_dt_s"]
        warnings.warn(f"{name}: {'; '.join(problems)} on the {hz:.0f} Hz physics grid. "
                      "Use a finer physics step for timing-sensitive conclusions.", RuntimeWarning, stacklevel=3)


def check(name: str, dt: float, rate_hz: float, latency_s: float) -> dict:
    """effective_timing plus the warning; returns the timing record."""
    timing = effective_timing(dt, rate_hz, latency_s)
    if math.isfinite(timing["requested_rate_hz"]):
        warn_if_quantized(name, timing)
    return timing
