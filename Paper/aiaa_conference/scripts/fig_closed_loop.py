"""Dispersed closed-loop landings on a reduced-order replica of the mission plant.

The flight computer is the production ConvexGuidanceController (planner, tracking loop, gyroscopic LQR, mixer,
deadband inverse), built from the 8S mission plant exactly as the mission runner builds it. The plant is a
reduced-order replica, not PhysX. It extends the replica of tests/unit/test_convex_guidance._land_on_vanes:

  * rigid body with the rotor's gyroscopic torque (implicit midpoint) and spool reaction;
  * vanes as effort torques at the probed momentum-plant authority, with the side force those torques imply
    at the vane lever arm (the non-minimum-phase lateral path);
  * the production servo model (lag, slew, deadband) and a 25 ms vane-joint lag;
  * the current-limited motor of the mission plant on a constant bus voltage;
  * a steady wind, intake momentum drag and axial/crossflow body drag.

Not modelled: contact (a run ends at touchdown height), the sensor chain (the controller reads true state),
battery sag within a flight, gusts, vane hinge dynamics and the coupled-jet model itself.

Starts are the first N of the planner study's dispersed starts (same seed), so each run's initial plan is known.

Usage: fig_closed_loop.py [N]         the production controller (default 400 starts);
       fig_closed_loop.py <variant> [N]   the same starts with one of VARIANTS (pd, latency, authority, rotor, com);
       fig_closed_loop.py report      re-print the numbers and figure from the stored results.
"""
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "physics_simulation" / "scripts"))
FIGURES = HERE.parents[1] / "figures"

DT, CONTROL_SUBSTEPS = 1.0 / 120.0, 4
RHO, CD, A_PERP, A_PAR = 1.225, 1.0, 0.042, 0.011
LEVER = 0.11                      # vane centre of pressure below the centre of mass, m
TOUCHDOWN = 0.3125                # body-origin height at touchdown, m
JET_SPEED_MAX, RAW_THRUST_MAX = 128.0, 48.0
N_RUNS, WORKERS = 400, 3
# Controls and model-mismatch variants: the controller is unchanged and unaware of the plant change.
VARIANTS = {
    "pd": dict(law="pd"),                 # the mission's earlier proportional-derivative attitude law
    "latency": dict(latency=0.4),         # every re-plan takes over 0.4 s after it was requested
    "authority": dict(authority=0.7),     # plant vanes 30% weaker than the controller's model
    "rotor": dict(rotor=1.2),             # plant rotor inertia 20% above the controller's model
    "com": dict(com=(0.005, 0.0)),        # 5 mm lateral centre-of-mass offset
}


def cases(n):
    rng = np.random.default_rng(20261116)          # the planner study's starts
    wind_rng = np.random.default_rng(7)
    out = []
    for i in range(n):
        rho, az = rng.uniform(0.0, 25.0), rng.uniform(0.0, 2 * math.pi)
        z = rng.uniform(5.0, 40.0)
        vh, vaz = rng.uniform(0.0, 3.0), rng.uniform(0.0, 2 * math.pi)
        vz = rng.uniform(-3.0, 1.0)
        volts = rng.uniform(26.0, 33.0)
        ws, waz = wind_rng.uniform(0.0, 4.0), wind_rng.uniform(0.0, 2 * math.pi)
        out.append(dict(i=i, r0=[rho * math.cos(az), rho * math.sin(az), z],
                        v0=[vh * math.cos(vaz), vh * math.sin(vaz), vz], volts=volts,
                        wind=[ws * math.cos(waz), ws * math.sin(waz), 0.0]))
    return out


_PLANT = None


def fly(case, max_time=120.0):
    """One landing. Returns the outcome record."""
    global _PLANT
    import torch
    from scipy.spatial.transform import Rotation
    from plant import Plant, _load
    from tvc_env.controllers.convex_adapter import _FRD_TO_ISAAC
    from tvc_env.dynamics.actuator_servo import ServoModel

    torch.set_num_threads(1)
    if _PLANT is None:
        _PLANT = Plant("planned_8s")
    p = _PLANT
    settings = _load("configs/controllers/convex_guidance.yaml")
    settings["guidance"]["solver_workers"] = 2
    if case.get("law"):
        settings["attitude"]["law"] = case["law"]
    if case.get("latency"):
        settings["guidance"]["plan_latency_s"] = float(case["latency"])
    ctrl = p.convex_controller(settings=settings)
    veh = ctrl.vehicle
    scale = float(case.get("authority", 1.0))
    K, Kpsi = scale * veh.vane_authority_nm_per_rad, scale * veh.yaw_authority_nm_per_rad
    h = float(case.get("rotor", 1.0)) * veh.rotor_inertia * veh.omega_max      # plant rotor momentum at full speed, N m s
    com = np.array([*case.get("com", (0.0, 0.0)), 0.0])
    q_bar, p_max = p.motor_limit_nm, p.shaft_power
    servo_cfg = p.c["servo"]
    servo = ServoModel(tau_servo=float(servo_cfg["tau_servo"]), max_angular_velocity=float(servo_cfg["max_angular_velocity"]),
                       max_command_angle=float(servo_cfg["max_command_angle"]), deadband=float(servo_cfg["deadband"]))
    joint_lag = float(servo_cfg["drive_damping"]) / float(servo_cfg["drive_stiffness"])
    slew = float(servo_cfg["max_angular_velocity"])
    inertia = np.diag([veh.inertia_rp_kg_m2, veh.inertia_rp_kg_m2, veh.inertia_yaw_kg_m2])
    I_frd = _FRD_TO_ISAAC.T @ inertia @ _FRD_TO_ISAAC
    m, g = veh.mass_kg, veh.gravity
    volts, wind = float(case["volts"]), np.array(case["wind"])
    v_ratio = volts / veh.reference_voltage_v

    r, v = np.array(case["r0"], float), np.array(case["v0"], float)
    R, w = np.eye(3), np.zeros(3)
    target, fins, joint_rate = torch.zeros(1, 4), np.zeros(4), np.zeros(4)
    rotor = min(veh.hover_throttle, v_ratio)
    t, energy_j = 0.0, 0.0
    worst_att = worst_tilt = worst_track = worst_speed = 0.0
    pq, soft_steps, steps, replans, max_solve_ms, max_gap = [], 0, 0, 0, 0.0, 0.0
    outcome = "timeout"
    started = time.perf_counter()
    while t < max_time:
        obs = np.concatenate([-r, _quat(R), _FRD_TO_ISAAC.T @ (R.T @ v), _FRD_TO_ISAAC.T @ w, [r[2]],
                              fins, joint_rate, [rotor], [0.0]])
        action = ctrl.compute_action(torch.tensor(obs[None], dtype=torch.float32), bus_voltage_v=volts)[0].numpy()
        tel = ctrl.last_telemetry or {}
        steps += 1
        worst_att = max(worst_att, tel.get("attitude_error_deg") or 0.0)
        if tel.get("phase") == "POWERED_DESCENT":
            worst_track = max(worst_track, tel.get("tracking_error_m") or 0.0)
        solver = tel.get("solver")
        if solver:
            soft_steps += solver["mode"] != "optimal"
            if tel.get("replanned"):
                replans += 1
                max_solve_ms = max(max_solve_ms, solver["solve_ms"])
                if solver["mode"] == "optimal":
                    max_gap = max(max_gap, solver["convexification_gap"])
        if t >= 2.0:
            pq.append(np.degrees(_FRD_TO_ISAAC.T @ w)[:2])
        duty = float(action[4])
        for _ in range(CONTROL_SUBSTEPS):
            target = servo.update(target, torch.tensor(action[None, :4]), DT)
            joint_rate = np.clip((target[0].double().numpy() - fins) / joint_lag, -slew, slew)
            fins = fins + joint_rate * DT
            # Current-limited motor (mission plant): first-order speed command, torque limit, coast at zero duty.
            command = min(1.0, max(0.0, duty * v_ratio))
            q_aero = p_max / veh.omega_max * rotor ** 2
            q_motor = float(np.clip(h * (command - rotor) / 0.15 + q_aero, -q_bar, q_bar)) if duty > 0.0 else 0.0
            previous = rotor
            rotor = min(1.0, max(0.0, rotor + (q_motor - q_aero) / h * DT))
            energy_j += (p_max * rotor ** 3 / veh.motor_efficiency + veh.auxiliary_power_w) * DT
            thrust = veh.full_thrust_n * rotor ** 2
            effort = np.array([(fins[2] - fins[0]) / 2, (fins[3] - fins[1]) / 2, fins.mean()])
            vane_torque = np.array([K, K, Kpsi]) * effort * rotor ** 2
            torque_frd = (vane_torque - np.array([0.0, 0.0, h * (rotor - previous) / DT])
                          + np.cross(com, [0.0, 0.0, -thrust]))
            H = h * rotor
            Hx = np.array([[0.0, -H, 0.0], [H, 0.0, 0.0], [0.0, 0.0, 0.0]])
            w_frd = _FRD_TO_ISAAC.T @ w
            rhs = I_frd @ w_frd + DT * (torque_frd - np.cross(w_frd, I_frd @ w_frd)) + 0.5 * DT * Hx @ w_frd
            w_new = _FRD_TO_ISAAC @ np.linalg.solve(I_frd - 0.5 * DT * Hx, rhs)
            R = R @ Rotation.from_rotvec(0.5 * (w + w_new) * DT).as_matrix()
            w = w_new
            # Side force of the vanes: the force at the lever arm that produces the roll/pitch torque.
            side = R @ (_FRD_TO_ISAAC @ np.array([vane_torque[1] / LEVER, -vane_torque[0] / LEVER, 0.0]))
            # Aerodynamics: intake momentum drag (linear) and axial/crossflow body drag (quadratic).
            v_rel = v - wind
            mdot = RAW_THRUST_MAX / JET_SPEED_MAX * rotor
            v_body = R.T @ v_rel
            perp = np.array([v_body[0], v_body[1], 0.0])
            body = -0.5 * RHO * CD * (A_PERP * np.linalg.norm(perp) * perp
                                      + A_PAR * abs(v_body[2]) * np.array([0.0, 0.0, v_body[2]]))
            force = thrust * R[:, 2] + side - mdot * v_rel + R @ body
            v = v + (force / m + np.array([0.0, 0.0, -g])) * DT
            r = r + v * DT
            t += DT
            worst_tilt = max(worst_tilt, math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2])))))
            worst_speed = max(worst_speed, float(np.linalg.norm(v)))
            if r[2] <= TOUCHDOWN:
                outcome = "touchdown"
                break
        if outcome == "touchdown":
            break
        if np.abs(w).max() > 20.0:
            outcome = "lost_attitude"
            break
    ctrl.close()
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))
    return dict(i=case["i"], outcome=outcome, phase=(ctrl.last_telemetry or {}).get("phase"), height=float(r[2]),
                impact=float(-v[2]), total_speed=float(np.linalg.norm(v)),
                horizontal_speed=float(np.linalg.norm(v[:2])), pad=float(np.linalg.norm(r[:2])), tilt=tilt,
                rate=float(np.degrees(np.linalg.norm((_FRD_TO_ISAAC.T @ w)[:2]))), time=t,
                energy_wh=energy_j / 3600.0, worst_att=worst_att, worst_tilt=worst_tilt, worst_track=worst_track,
                worst_speed=worst_speed, pq_rms=float(np.sqrt(np.mean(np.square(pq)))) if pq else 0.0,
                soft_fraction=soft_steps / max(steps, 1), replans=replans, max_solve_ms=max_solve_ms, max_gap=max_gap,
                wall_s=time.perf_counter() - started, volts=volts, wind=float(np.linalg.norm(wind)),
                offset=float(np.linalg.norm(case["r0"][:2])), altitude=case["r0"][2], vz0=case["v0"][2])


def _quat(R):
    w = math.sqrt(max(0.0, 1 + np.trace(R))) / 2
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])


def classify(rec):
    if rec["outcome"] != "touchdown":
        return rec["outcome"]
    if rec["impact"] > 3.0 or rec["tilt"] > math.degrees(0.5):
        return "crash"
    return "landed" if rec["impact"] <= 0.25 and rec["pad"] <= 0.5 else "off_nominal"


def main():
    args = sys.argv[1:]
    variant = args.pop(0) if args and args[0] in VARIANTS else None
    n = int(args[0]) if args else N_RUNS
    todo = [dict(c, **VARIANTS.get(variant, {})) for c in cases(n)]
    tag = "" if variant is None else f"_{variant}"
    started = time.perf_counter()
    if n <= 2:
        records = [fly(c) for c in todo]
    else:
        with ProcessPoolExecutor(WORKERS) as pool:
            records = list(pool.map(fly, todo, chunksize=4))
    for rec in records:
        rec["class"] = classify(rec)
    (FIGURES / f"closed_loop{tag}_results.json").write_text(json.dumps(records, indent=0))
    report(records, time.perf_counter() - started, figure=variant is None)


def report(records, wall=None, figure=True):
    from plant import C, style
    plt = style()
    n = len(records)
    counts = {k: sum(r["class"] == k for r in records) for k in ("landed", "off_nominal", "crash", "lost_attitude", "timeout")}
    print(f"{n} closed-loop runs" + (f" in {wall / 60:.1f} min wall" if wall else "") + ": "
          + ", ".join(f"{k} {v}" for k, v in counts.items()))
    down = [r for r in records if r["outcome"] == "touchdown"]

    def stats(name, values, unit, fmt=".3f"):
        if not values:
            return
        a = np.asarray(values, float)
        print(f"  {name}: median {np.median(a):{fmt}}, p95 {np.percentile(a, 95):{fmt}}, max {a.max():{fmt}} {unit}")

    stats("downward speed, all height crossings", [r["impact"] for r in down], "m/s")
    if down and all("total_speed" in r and "horizontal_speed" in r for r in down):
        stats("total body speed, all height crossings", [r["total_speed"] for r in down], "m/s")
        stats("horizontal body speed, all height crossings", [r["horizontal_speed"] for r in down], "m/s")
    stats("pad error, all height crossings", [r["pad"] for r in down], "m")
    stats("tilt at height crossing", [r["tilt"] for r in down], "deg", ".2f")
    stats("peak attitude error", [r["worst_att"] for r in down], "deg", ".2f")
    stats("peak tilt in flight", [r["worst_tilt"] for r in down], "deg", ".2f")
    stats("peak tracking error in powered descent", [r["worst_track"] for r in down], "m")
    stats("roll/pitch rate RMS after 2 s", [r["pq_rms"] for r in down], "deg/s", ".2f")
    stats("flight time", [r["time"] for r in down], "s", ".1f")
    stats("electrical energy", [r["energy_wh"] for r in down], "Wh", ".2f")
    stats("largest re-plan solve time per run", [r["max_solve_ms"] for r in down], "ms", ".0f")
    print(f"  re-plans per run: median {np.median([r['replans'] for r in records]):.0f}; runs that ever tracked a "
          f"soft-terminal plan: {sum(r['soft_fraction'] > 0 for r in records)}; largest gap of a tracked hard-terminal plan "
          f"{max(r['max_gap'] for r in records):.1e}")
    bad = [r for r in records if r["class"] != "landed"]
    for r in bad[:40]:
        print(f"  not nominal #{r['i']}: {r['class']}, last downward speed {r['impact']:.2f} m/s, pad {r['pad']:.2f} m, start offset "
              f"{r['offset']:.1f} m alt {r['altitude']:.1f} m vz {r['vz0']:+.1f} m/s, bus {r['volts']:.1f} V, wind "
              f"{r['wind']:.1f} m/s, soft-plan share {r['soft_fraction']:.2f}, t {r['time']:.1f} s, last phase "
              f"{r.get('phase')} at {r.get('height', float('nan')):.2f} m")
    for field, bins in (("volts", ((26, 28), (28, 30), (30, 33))), ("wind", ((0, 2), (2, 4)))):
        for lo, hi in bins:
            subset = [r for r in records if lo <= r[field] < hi]
            if not subset:
                continue
            crossings = [r for r in subset if r["outcome"] == "touchdown"]
            passed = sum(r["class"] == "landed" for r in subset)
            if field == "volts":
                label = f"bus {lo}-{hi} V"
                summary = f", downward speed median {np.median([r['impact'] for r in crossings]):.3f} m/s" if crossings else ""
            else:
                label = f"wind {lo}-{hi} m/s"
                pad = [r["pad"] for r in crossings]
                summary = f", pad error median {np.median(pad):.3f} m, p95 {np.percentile(pad, 95):.3f} m" if pad else ""
            print(f"  {label}: {passed}/{len(subset)} passed{summary}")

    if not figure:
        return
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.2))
    ax = axes[0]
    ax.plot([r["pad"] for r in down], [r["impact"] for r in down], ".", ms=3, color=C["blue"])
    ax.axhline(0.25, color=C["red"], ls=":", lw=0.9)
    ax.axvline(0.5, color=C["red"], ls=":", lw=0.9)
    ax.text(0.02, 0.255, "success limits", fontsize=6, color=C["red"], va="bottom")
    ax.set_xlim(0.0, 0.55)
    ax.set_ylim(0.0, 0.3)
    ax.set_xlabel("Pad error (m)")
    ax.set_ylabel("Downward speed (m/s)")
    ax.set_title("(a) Height crossing", fontsize=8)
    ax = axes[1]
    ax.plot([r["offset"] for r in down], [r["worst_tilt"] for r in down], ".", ms=3, color=C["green"])
    ax.axhline(15.0, color=C["grey"], ls=":", lw=0.8)
    ax.axhline(22.0, color=C["grey"], ls="--", lw=0.8)
    ax.set_ylim(0.0, 24.0)
    ax.set_xlabel("Initial offset from pad (m)")
    ax.set_ylabel("Peak tilt in flight (deg)")
    ax.set_title("(b) Tilt", fontsize=8)
    ax = axes[2]
    pd_file = FIGURES / "closed_loop_pd_results.json"
    bins = np.linspace(0.0, 16.0, 33)
    if pd_file.exists():
        pd_runs = json.loads(pd_file.read_text())
        ids = {r["i"] for r in pd_runs}
        ax.hist([r["pq_rms"] for r in records if r["i"] in ids], bins=bins, color=C["blue"], alpha=0.85, label="scheduled LQR")
        ax.hist([r["pq_rms"] for r in pd_runs], bins=bins, color=C["red"], alpha=0.75, label="earlier PD law")
        ax.legend(frameon=False, fontsize=6, loc="upper center")
    else:
        ax.hist([r["pq_rms"] for r in records], bins=bins, color=C["blue"], alpha=0.85)
    ax.set_xlabel("Roll/pitch rate RMS (deg/s)")
    ax.set_ylabel("Runs")
    ax.set_title("(c) Body rates, 100 starts", fontsize=8)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(FIGURES / "closed_loop.pdf")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        recs = json.loads((FIGURES / "closed_loop_results.json").read_text())
        report(recs)
    else:
        main()
