"""Sensitivity of the mission plant to the jet-speed convention (printed; no figure).

The plant takes the nozzle jet speed from the open-rotor far-wake relation u = sqrt(2 T / (rho A)) with the
fan's swept area A (configs/hardware/planned_8s.yaml: 128 m/s). A ducted fan whose exit area is about the
swept area exhausts at u_e = sqrt(T / (rho A)) instead: the duct prevents the wake contraction. This script
rebuilds the production models with that jet speed and prints what changes.
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "physics_simulation" / "scripts"))
import plant as plant_module  # noqa: E402
from plant import DTYPE, Plant  # noqa: E402
from tvc_env.dynamics.propulsion_edf import duct_coriolis_force, inlet_momentum_drag  # noqa: E402

RHO, AREA, THRUST = 1.225, 4.78e-3, 48.0
u_open = math.sqrt(2 * THRUST / (RHO * AREA))
u_duct = math.sqrt(THRUST / (RHO * AREA))
print(f"swept area {AREA * 1e6:.0f} mm^2: open-rotor far wake {u_open:.1f} m/s, ducted exit {u_duct:.1f} m/s; "
      f"ideal jet power {0.5 * THRUST * u_open:.0f} W vs {0.5 * THRUST * u_duct:.0f} W")

base_configs = plant_module.configs


def build(speed):
    def patched(profile="planned_8s"):
        c = base_configs(profile)
        c["edf"]["edf"]["exhaust_speed_nominal"] = speed
        return c
    plant_module.configs = patched
    try:
        return Plant("planned_8s")
    finally:
        plant_module.configs = base_configs


def report(name, p):
    hover = p.hover_fraction()
    out, raw = p.jet_at(torch.zeros(1, 4, dtype=DTYPE), hover)
    mdot = float(out.mass_flow_per_fin.sum(-1))
    net = p.neutral_net_thrust(1.0)
    roll, yaw = p.vane_authority()
    q_area = 0.5 * p.aero.air_density * (p.aero.exhaust_speed if hasattr(p.aero, "exhaust_speed") else 0.0) ** 2
    w = torch.zeros(1, 3, dtype=DTYPE)
    w[0, 0] = 0.2
    jet, _ = p.jet_at(torch.zeros(1, 4, dtype=DTYPE), hover, body_rate=w)
    vane_damp = -float((p.moment_about_com(jet.forces) - p.moment_about_com(out.forces))[0, 0]) / 0.2
    side = float(-inlet_momentum_drag(torch.tensor([mdot]), torch.tensor([[4.0, 0.0, 0.0]]))[0, 0])
    cross = 2 * mdot / (RHO * 1.0 * 0.042)
    z_e, z_i = 0.05 - float(p.com[2]), float(p.inlet[2] - p.com[2])
    print(f"{name}: hover fraction {hover:.4f}, mdot {mdot:.4f} kg/s, net static thrust {net:.2f} N "
          f"(loss {100 * (1 - net / THRUST):.1f}%), authority roll {roll:.3f} yaw {yaw:.3f} N m/rad at full speed, "
          f"vane-flow damping {vane_damp:.2e}, jet damping mdot z_e^2 {mdot * z_e ** 2:.2e} N m s/rad, "
          f"intake drag at 4 m/s {side:.2f} N, intake = body drag at {cross:.1f} m/s")
    return dict(hover=hover, mdot=mdot, net=net, roll=roll, yaw=yaw)


a = report("open-rotor (production)", build(128.0))
b = report("ducted exit           ", build(u_duct))
print("ratios ducted/production: " + ", ".join(f"{k} {b[k] / a[k]:.3f}" for k in a))
