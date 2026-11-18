"""Fig. convex_plan and convex_search: one landing plan from the production planner, and its free-final-time search.

The planner limits are derived from the 8S mission plant exactly as ConvexGuidanceController
derives them at a re-plan (_maybe_replan): rho1/rho2 from the measured bus voltage, the
thrust-rate bound from the yaw-safe duty slew, the tilt-rate bound from the vanes' precession
authority. Solved with Clarabel; no PhysX.
"""
import math
import time
from dataclasses import replace

import numpy as np

from plant import C, FIGURES, Plant, style
from tvc_env.controllers.convex_guidance import _Segment

plt = style()
p = Plant("planned_8s")
ctrl = p.convex_controller()
veh = ctrl.vehicle
volts = 32.0                                   # loaded 8S bus near full charge (fig_battery)
available = veh.available_thrust_n(volts)
ceiling = ctrl._planning_ceiling(available)
limits = replace(ctrl._limits, thrust_rate_n_s=ctrl._thrust_rate(volts), thrust_min_n=ctrl._planning_floor(ceiling),
                 thrust_max_n=ceiling, emergency_thrust_max_n=available,
                 emergency_thrust_rate_n_s=ctrl._thrust_rate(volts, ctrl.spool_rate))
guidance = ctrl.guidance
guidance.limits = limits
gate = ctrl.gate
g = limits.gravity
m = limits.mass_kg

start, velocity = np.array([8.0, 5.0, 18.0]), np.array([0.0, 0.0, 0.0])
t0 = time.perf_counter()
plan = guidance.plan(start, velocity, veh.weight_n, gate)
solve_s = time.perf_counter() - t0

# Dense samples of the first-order-hold plan.
ts = np.linspace(0.0, plan.duration, 400)
samples = [plan.sample(float(t)) for t in ts]
pos = np.array([s[0] for s in samples])
vel = np.array([s[1] for s in samples])
acc = np.array([s[2] for s in samples])
thrust = np.linalg.norm(acc, axis=1) * m
tilt = np.degrees(np.arctan2(np.linalg.norm(acc[:, :2], axis=1), acc[:, 2]))
apex = np.asarray(gate.apex)
horizontal = np.linalg.norm(pos[:, :2] - apex[:2], axis=1)
node_h = np.linalg.norm(plan.position[:, :2] - apex[:2], axis=1)
# Sink-rate envelope -v_z <= sqrt(v_gate^2 + 2 a (z - z_gate)), a = fraction * (rho2/m - g).
a_env = limits.landing_sink_brake_fraction * (limits.thrust_max_n / m - g)
v_gate = abs(gate.velocity[2])
z_gate = gate.position[2]
envelope = np.sqrt(np.maximum(v_gate ** 2 + 2 * a_env * (pos[:, 2] - z_gate), 0.0))

fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.95))
ax = axes[0]
h_cone = np.linspace(0, 18, 50)
ax.fill_betweenx(h_cone + apex[2], h_cone * math.tan(limits.glide_slope_rad), 20, color=C["grey"], alpha=0.15, lw=0)
ax.plot(horizontal, pos[:, 2], color=C["blue"])
ax.plot(node_h, plan.position[:, 2], "o", ms=2.2, color=C["blue"])
ax.plot([0], [z_gate], "v", color=C["black"], ms=4)
ax.set_xlim(0, 11)
ax.set_ylim(0, 19)
ax.set_xlabel("Distance from pad (m)")
ax.set_ylabel("Altitude (m)")
ax.set_title("(a) Path, glide cone", fontsize=8)

ax = axes[1]
ax.plot(ts, thrust, color=C["blue"], label=r"$m\|u\|$")
ax.plot(plan.times, plan.sigma * m, "o", ms=2.2, mfc="none", color=C["orange"], label=r"$m\sigma$")
for level, name in ((limits.thrust_min_n, r"$\rho_1$"), (veh.weight_n, "$W$"), (limits.thrust_max_n, r"$\rho_2$")):
    ax.axhline(level, color=C["grey"], ls=":", lw=0.8)
    ax.text(plan.duration, level + 0.4, name, ha="right", fontsize=6.5, color=C["grey"])
ax.set_xlabel("Time (s)")
ax.set_ylabel("Thrust (N)")
ax.set_title("(b) Thrust and slack", fontsize=8)
ax.set_ylim(23.0, 38.5)
ax.legend(loc="upper left", frameon=False, fontsize=6.5)

ax = axes[2]
ax.plot(ts, tilt, color=C["blue"])
ax.axhline(math.degrees(limits.max_tilt_rad), color=C["grey"], ls=":", lw=0.8)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Planned tilt (deg)")
ax.set_title("(c) Thrust pointing", fontsize=8)

ax = axes[3]
ax.plot(ts, np.linalg.norm(vel, axis=1), color=C["blue"], label=r"$\|v\|$")
ax.plot(ts, -vel[:, 2], color=C["green"], ls="--", label=r"$-v_z$")
ax.plot(ts, envelope, color=C["red"], ls=":", label="sink envelope")
ax.axhline(limits.max_speed_m_s, color=C["grey"], ls=":", lw=0.8)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Speed (m/s)")
ax.set_title("(d) Speed bounds", fontsize=8)
ax.set_ylim(0, 4.5)
ax.legend(loc="lower center", frameon=False, fontsize=6)
fig.tight_layout(w_pad=0.6)
fig.savefig(FIGURES / "convex_plan.pdf")

# ---- free-final-time search: cost of the fixed-duration SOCP against landing duration ----------------
durations = np.linspace(3.0, 20.0, 69)
costs = []
stats = dict(solves=0)
for t_land in durations:
    sol = guidance._solve(start, velocity, veh.weight_n, gate, [_Segment("land", float(t_land), guidance.landing_nodes)],
                          stats)
    costs.append(np.nan if sol is None else sol.cost)
costs = np.array(costs)
feasible = ~np.isnan(costs)
t_min = float(durations[feasible][0])

fig, ax = plt.subplots(figsize=(3.45, 2.1))
ax.axvspan(durations[0], t_min, color=C["red"], alpha=0.12, lw=0)
ax.text(0.5 * (durations[0] + t_min), np.nanmax(costs) * 0.98, "infeasible", ha="center", va="top", fontsize=6.5,
        color=C["red"])
ax.plot(durations[feasible], costs[feasible], color=C["blue"], label="Fixed-duration SOCP")
ax.plot([plan.duration], [plan.cost], "o", color=C["orange"], ms=5, label=f"Line search: {plan.duration:.2f} s")
ax.set_xlabel(r"Landing duration $t_f$ (s)")
ax.set_ylabel("Planned energy (Wh)")
ax.legend(loc="upper right", frameon=False)
fig.tight_layout()
fig.savefig(FIGURES / "convex_search.pdf")

dt_nodes = np.diff(plan.times)
rate = np.linalg.norm(np.diff(plan.thrust_accel, axis=0), axis=1) * m / dt_nodes
slew = np.degrees(np.linalg.norm(np.diff(plan.thrust_accel[:, :2], axis=0), axis=1) / dt_nodes / g)
print(f"bus {volts} V: available {available:.2f} N, rho1 {limits.thrust_min_n:.2f} N, W {veh.weight_n:.2f} N, "
      f"rho2 {limits.thrust_max_n:.2f} N, thrust rate {limits.thrust_rate_n_s:.2f} N/s "
      f"(duty slew {ctrl._duty_rate(volts):.4f} /s), tilt-rate bound {math.degrees(limits.tilt_rate_rad_s):.2f} deg/s, "
      f"emergency rate {limits.emergency_thrust_rate_n_s:.1f} N/s, sink-envelope a {a_env:.3f} m/s^2")
print(f"plan from {start.tolist()}: {plan.mode}, {plan.duration:.2f} s, {plan.energy_wh:.3f} Wh, "
      f"gap {plan.convexification_gap:.1e}, {plan.solves} SOCPs in {solve_s * 1e3:.0f} ms ({plan.iterations} IPM iterations "
      f"in the selected solve), max tilt {tilt.max():.2f} deg, max |v| {np.linalg.norm(vel, axis=1).max():.2f} m/s, "
      f"max thrust rate {rate.max():.2f} N/s, max horizontal slew {slew.max():.2f} deg/s")
second = np.diff(plan.sigma * m, 2)
late = plan.times[1:-1] > 0.5 * plan.duration
print(f"node-to-node thrust alternation (second difference) in the second half: max {np.abs(second[late]).max():.2f} N")
print(f"search: infeasible below {t_min:.2f} s; grid minimum {durations[np.nanargmin(costs)]:.2f} s "
      f"({np.nanmin(costs):.4f} Wh) vs line search {plan.duration:.2f} s ({plan.cost:.4f} Wh)")

# Lossless-convexification check over several starts.
for s0, v0 in (([2.0, -1.5, 10.0], [0, 0, 0]), ([20.0, 0.0, 25.0], [-3.0, 0.0, -2.0]), ([-0.28, 0.82, 18.0], [0, 0, -1.0])):
    t0 = time.perf_counter()
    pl = guidance.plan(s0, v0, veh.weight_n, gate)
    took = time.perf_counter() - t0
    norm = np.linalg.norm(pl.thrust_accel, axis=1) * m
    k = int(np.argmax((pl.sigma * m - norm) / (pl.sigma * m)))
    print(f"start {s0} v {v0}: {pl.mode}, {pl.duration:.2f} s, gap {pl.convexification_gap:.1e} "
          f"(node {k}: sigma {pl.sigma[k] * m:.2f} N, |u| {norm[k]:.2f} N), {pl.solves} SOCPs, {took * 1e3:.0f} ms")
