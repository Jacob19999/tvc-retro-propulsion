"""Planner studies for the AIAA draft: dispersed-start Monte Carlo, node-count convergence, duration structure.

Uses the production planner (tvc_env.controllers.convex_guidance) with limits derived from the 8S
mission plant exactly as the flight computer derives them at a re-plan (see
Paper/physics_simulation/scripts/fig_convex_plan.py). Solved with Clarabel; no PhysX, no closed loop.
"""
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "physics_simulation" / "scripts"))
from plant import C, Plant, style  # noqa: E402
import tvc_env.controllers.convex_guidance as cg  # noqa: E402
from tvc_env.controllers.convex_guidance import _Segment  # noqa: E402

FIGURES = HERE.parents[1] / "figures"
plt = style()
p = Plant("planned_8s")
ctrl = p.convex_controller()
veh = ctrl.vehicle
gate = ctrl.gate
guidance = ctrl.guidance


def limits_at(volts):
    available = veh.available_thrust_n(volts)
    ceiling = ctrl._planning_ceiling(available)
    return replace(ctrl._limits, thrust_rate_n_s=ctrl._thrust_rate(volts), thrust_min_n=ctrl._planning_floor(ceiling),
                   thrust_max_n=ceiling, emergency_thrust_max_n=available,
                   emergency_thrust_rate_n_s=ctrl._thrust_rate(volts, ctrl.spool_rate))


def violations(plan, lim):
    """Largest constraint excess between nodes, from dense samples of the first-order-hold plan."""
    m, g = lim.mass_kg, lim.gravity
    ts = np.linspace(0.0, plan.duration, 40 * (len(plan.times) - 1) + 1)
    s = [plan.sample(float(t)) for t in ts]
    pos, vel, acc = (np.array([x[i] for x in s]) for i in range(3))
    thrust = np.linalg.norm(acc, axis=1) * m
    apex = np.asarray(gate.apex)
    a_env = lim.landing_sink_brake_fraction * (lim.thrust_max_n / m - g)
    env = np.sqrt(np.maximum(abs(gate.velocity[2]) ** 2 + 2 * a_env * (pos[:, 2] - gate.position[2]), 0.0))
    return dict(
        floor=float(np.max(lim.thrust_min_n - thrust)),              # N below rho1 (chord of the annulus)
        speed=float(np.max(np.linalg.norm(vel, axis=1) - lim.max_speed_m_s)),
        sink=float(np.max(-vel[:, 2] - env)),
        glide=float(np.max(np.linalg.norm(pos[:, :2] - apex[:2], axis=1)
                           - math.tan(lim.glide_slope_rad) * (pos[:, 2] - apex[2]))),
        tilt=float(np.degrees(np.max(np.arctan2(np.linalg.norm(acc[:, :2], axis=1), acc[:, 2])))))


# ---- (A) dispersed starts ------------------------------------------------------------------------
rng = np.random.default_rng(20261116)
N_MC = 400
cases = []
for _ in range(N_MC):
    rho, az = rng.uniform(0.0, 25.0), rng.uniform(0.0, 2 * math.pi)
    z = rng.uniform(5.0, 40.0)
    vh, vaz = rng.uniform(0.0, 3.0), rng.uniform(0.0, 2 * math.pi)
    vz = rng.uniform(-3.0, 1.0)
    volts = rng.uniform(26.0, 33.0)
    cases.append((np.array([rho * math.cos(az), rho * math.sin(az), z]),
                  np.array([vh * math.cos(vaz), vh * math.sin(vaz), vz]), volts))


def run(tolerance):
    cg.LCVX_GAP_TOLERANCE = tolerance
    rows = []
    for r0, v0, volts in cases:
        guidance.limits = limits_at(volts)
        t0 = time.perf_counter()
        plan = guidance.plan(r0, v0, veh.weight_n, gate)
        took = time.perf_counter() - t0
        rows.append(None if plan is None else dict(
            mode=plan.mode, gap=plan.convexification_gap, solves=plan.solves, ms=1e3 * took, it=plan.iterations,
            duration=plan.duration, energy=plan.energy_wh, miss=plan.terminal_miss_m))
    return rows


raw = run(math.inf)        # relaxation as solved, guard off
guarded = run(1e-3)        # production setting
cg.LCVX_GAP_TOLERANCE = 1e-3


def summarize(name, rows):
    opt = [r for r in rows if r is not None and r["mode"] == "optimal"]
    soft = [r for r in rows if r is not None and r["mode"] != "optimal"]
    gaps = np.array([r["gap"] for r in opt])
    ms = np.array([r["ms"] for r in opt])
    solves = np.array([r["solves"] for r in opt])
    its = np.array([r["it"] for r in opt])
    print(f"{name}: {len(rows)} starts, optimal {len(opt)}, soft-terminal {len(soft)}, none {sum(r is None for r in rows)}")
    print(f"  gap: median {np.median(gaps):.1e}, max {gaps.max():.1e}, >1e-6: {(gaps > 1e-6).sum()}, >1e-3: {(gaps > 1e-3).sum()}")
    print(f"  solve time ms: median {np.median(ms):.0f}, p95 {np.percentile(ms, 95):.0f}, max {ms.max():.0f}; "
          f"SOCPs median {np.median(solves):.0f}, max {solves.max()}; IPM iterations median {np.median(its):.0f}, max {its.max()}")
    return opt, soft


raw_opt, _ = summarize("guard off", raw)
g_opt, g_soft = summarize("guard on ", guarded)
both = [(a, b) for a, b in zip(raw, guarded) if a and b and a["mode"] == b["mode"] == "optimal" and a["gap"] > 1e-3]
if both:
    dE = np.array([100 * (b["energy"] / a["energy"] - 1) for a, b in both])
    dT = np.array([b["duration"] - a["duration"] for a, b in both])
    print(f"  guarded vs raw on the {len(both)} gapped starts: energy +{np.median(dE):.2f}% median, +{dE.max():.2f}% max; "
          f"duration +{np.median(dT):.2f} s median, +{dT.max():.2f} s max; residual gap max "
          f"{max(b['gap'] for _, b in both):.1e}")
gapped = np.array([c[0:2] for c, a in zip(cases, raw) if a and a["mode"] == "optimal" and a["gap"] > 1e-3], dtype=float)
if len(gapped):
    off = np.linalg.norm(gapped[:, 0, :2], axis=1) / gapped[:, 0, 2]
    print(f"  gapped starts: offset/altitude median {np.median(off):.2f} (all starts "
          f"{np.median([np.linalg.norm(c[0][:2]) / c[0][2] for c in cases]):.2f}), "
          f"v_z median {np.median(gapped[:, 1, 2]):.2f} m/s")

fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.35))
ax = axes[0]
edges = np.logspace(-11, 0, 45)
ax.hist(np.maximum([r["gap"] for r in raw_opt], 1e-11), bins=edges, color=C["orange"], alpha=0.8, label="as solved")
ax.hist(np.maximum([r["gap"] for r in g_opt], 1e-11), bins=edges, histtype="step", color=C["blue"], lw=1.2, label="with gap guard")
ax.axvline(1e-3, color=C["grey"], ls=":", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel(r"Convexification gap $\max_k(\sigma_k-\|u_k\|)/\sigma_k$")
ax.set_ylabel("Plans")
ax.set_title("(a) Losslessness", fontsize=8)
ax.legend(frameon=False, fontsize=6, loc="center right", bbox_to_anchor=(1.0, 0.62))
ax = axes[1]
ax.hist([r["ms"] for r in g_opt], bins=30, color=C["blue"], alpha=0.85)
ax.set_xlabel("Wall time per plan (ms)")
ax.set_ylabel("Plans")
ax.set_title("(b) Planning time", fontsize=8)
ax = axes[2]
xy = np.array([(np.linalg.norm(c[0][:2]), c[0][2]) for c in cases])
is_opt = np.array([r is not None and r["mode"] == "optimal" for r in guarded])
is_gap = np.array([a is not None and a["mode"] == "optimal" and a["gap"] > 1e-3 for a in raw])
ax.plot(xy[is_opt & ~is_gap, 0], xy[is_opt & ~is_gap, 1], ".", ms=2.5, color=C["blue"], label="lossless")
ax.plot(xy[is_gap, 0], xy[is_gap, 1], "s", ms=3, mfc="none", color=C["orange"], label="gap, repaired")
ax.plot(xy[~is_opt, 0], xy[~is_opt, 1], "x", ms=3.5, color=C["red"], label="soft-terminal")
ax.set_xlabel("Initial offset from pad (m)")
ax.set_ylabel("Initial altitude (m)")
ax.set_title("(c) Outcome by start", fontsize=8)
ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, columnspacing=0.8, handletextpad=0.2)
fig.tight_layout(w_pad=0.8)
fig.savefig(FIGURES / "planner_mc.pdf")

# ---- (B) node-count convergence --------------------------------------------------------------------
guidance.limits = lim = limits_at(32.0)
start, vel0 = np.array([8.0, 5.0, 18.0]), np.zeros(3)
nodes = [8, 12, 16, 24, 32, 48, 64, 96]
conv = []
for n in nodes:
    guidance.landing_nodes = n
    t0 = time.perf_counter()
    plan = guidance.plan(start, vel0, veh.weight_n, gate)
    took = 1e3 * (time.perf_counter() - t0)
    v = violations(plan, lim)
    alt = float(np.abs(np.diff(plan.sigma * lim.mass_kg, 2))[len(plan.sigma) // 2:].max())
    conv.append(dict(n=n, T=plan.duration, E=plan.energy_wh, ms=took, alt=alt, **v))
    print(f"N={n:3d}: t_f {plan.duration:.3f} s, {plan.energy_wh:.4f} Wh, {took:.0f} ms, gap {plan.convexification_gap:.1e}, "
          f"between nodes: thrust below rho1 {v['floor']:+.3f} N, speed {v['speed']:+.3f} m/s, sink {v['sink']:+.3f} m/s, "
          f"glide {v['glide']:+.3f} m, alternation {alt:.2f} N")
guidance.landing_nodes = 24
ref = conv[-1]
fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.2))
ax = axes[0]
ax.plot(nodes[:-1], [100 * abs(c["E"] / ref["E"] - 1) for c in conv[:-1]], "o-", ms=3, color=C["blue"])
ax.axvline(24, color=C["grey"], ls=":", lw=0.8)
ax.set_yscale("log")
ax.set_xlabel("Landing-leg intervals $N$")
ax.set_ylabel(f"Energy vs $N={nodes[-1]}$ (%)")
ax.set_title("(a) Convergence", fontsize=8)
ax = axes[1]
ax.plot(nodes, [max(c["sink"], 0.0) for c in conv], "o-", ms=3, color=C["red"], label="sink envelope (m/s)")
ax.plot(nodes, [max(c["speed"], 0.0) for c in conv], "s--", ms=3, color=C["green"], label="speed bound (m/s)")
ax.axvline(24, color=C["grey"], ls=":", lw=0.8)
ax.set_xlabel("Landing-leg intervals $N$")
ax.set_ylabel("Excess between nodes")
ax.legend(frameon=False, fontsize=6)
ax.set_title("(b) Inter-node violation", fontsize=8)
ax = axes[2]
ax.plot(nodes, [c["ms"] for c in conv], "o-", ms=3, color=C["blue"])
ax.axvline(24, color=C["grey"], ls=":", lw=0.8)
ax.set_xlabel("Landing-leg intervals $N$")
ax.set_ylabel("Wall time per plan (ms)")
ax.set_title("(c) Cost of resolution", fontsize=8)
fig.tight_layout(w_pad=0.8)
fig.savefig(FIGURES / "planner_nodes.pdf")

# ---- (C) structure of the duration search ------------------------------------------------------------
cg.LCVX_GAP_TOLERANCE = math.inf
checked = holes = nonmono = 0
slopes, curves = [], []
for r0, v0, volts in cases[:60]:
    guidance.limits = limits_at(volts)
    stats = dict(solves=0)
    ts = np.linspace(2.0, 30.0, 57)
    cost = np.array([(lambda s: np.nan if s is None else s.cost)(
        guidance._solve(r0, v0, veh.weight_n, gate, [_Segment("land", float(t), 24)], stats)) for t in ts])
    ok = ~np.isnan(cost)
    if not ok.any():
        continue
    checked += 1
    first = int(np.argmax(ok))
    last = len(ok) - 1 - int(np.argmax(ok[::-1]))
    holes += int((~ok[first:last + 1]).any())
    c = cost[first:last + 1][ok[first:last + 1]]
    nonmono += int((np.diff(c) < -1e-6).any())
    if len(curves) < 10 and ok.sum() > 8:
        curves.append((ts[ok], cost[ok]))
    if ok[first:].sum() > 4:
        slopes.append(np.polyfit(ts[first:][ok[first:]], cost[first:][ok[first:]], 1)[0])
cg.LCVX_GAP_TOLERANCE = 1e-3
# Reference start (Fig. plan): fine duration grid and the production line search.
guidance.limits = limits_at(32.0)
ref_plan = guidance.plan(start, vel0, veh.weight_n, gate)
ref_t = np.linspace(3.0, 20.0, 69)
ref_c = np.array([(lambda s: np.nan if s is None else s.cost)(
    guidance._solve(start, vel0, veh.weight_n, gate, [_Segment("land", float(t), 24)], dict(solves=0))) for t in ref_t])
ref_ok = ~np.isnan(ref_c)
t_min = float(ref_t[ref_ok][0])
hover_wh_s = (veh.energy_model().power_ref_w * (veh.weight_n / veh.energy_model().thrust_ref_n) ** 1.5
              + veh.energy_model().auxiliary_w) / 3600.0
fig, ax = plt.subplots(figsize=(4.2, 2.3))
for i, (t, c) in enumerate(curves):
    ax.plot(t, c, color=C["grey"], lw=0.7, alpha=0.7, label="Dispersed starts" if i == 0 else None)
    ax.plot(t[:1], c[:1], ".", ms=3, color=C["grey"])
ax.plot(ref_t[ref_ok], ref_c[ref_ok], color=C["blue"], lw=1.4, label="Start of Fig. 6")
ax.plot([ref_plan.duration], [ref_plan.cost], "o", ms=4.5, color=C["orange"], label=f"Line search: {ref_plan.duration:.2f} s")
ax.axvspan(ref_t[0] - 1.0, t_min, color=C["red"], alpha=0.10, lw=0)
ax.set_xlim(2.0, 30.0)
ax.set_xlabel(r"Landing duration $t_f$ (s)")
ax.set_ylabel("Planned energy (Wh)")
ax.legend(loc="lower right", frameon=False, fontsize=6.5)
fig.tight_layout()
fig.savefig(FIGURES / "duration_search.pdf")
print(f"reference start: infeasible below {t_min:.2f} s; line search {ref_plan.duration:.2f} s ({ref_plan.cost:.4f} Wh) vs "
      f"grid minimum {ref_c[ref_ok][0]:.4f} Wh; hover power {hover_wh_s:.4f} Wh/s")
print(f"duration structure over {checked} starts (0.5 s grid, 2-30 s): feasible set not an interval in {holes}, "
      f"cost not non-decreasing beyond the boundary in {nonmono}; slope {np.median(slopes):.3f} Wh/s median "
      f"({np.min(slopes):.3f}..{np.max(slopes):.3f})")
guidance.close()
