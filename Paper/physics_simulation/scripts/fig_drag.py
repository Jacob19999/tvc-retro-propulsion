"""Fig. drag: crosswise aerodynamic force budget at hover, and the rotational damping audit (printed)."""
import torch

from plant import C, DTYPE, FIGURES, Plant, style
from tvc_env.dynamics.coupled_jet import cylinder_rotational_drag
from tvc_env.dynamics.propulsion_edf import duct_coriolis_force, inlet_momentum_drag
from tvc_env.common.frames import frd_to_isaac

plt = style()
p = Plant("planned_8s")
hover = p.hover_fraction()
out, raw = p.jet_at(torch.zeros(1, 4, dtype=DTYPE), hover)
mdot = out.mass_flow_per_fin.sum(-1).float()                     # (1,)
wind = p.wind()
legacy_iso = dict(cd=1.0, area=0.02)                               # old disturbance-file body_drag
legacy_calm = dict(cd=1.0, area=0.011)                             # old still-air path (end-on area)

speeds = torch.linspace(0.0, 15.0, 61)
inlet, body, iso, calm = [], [], [], []
level = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
for v in speeds.tolist():
    v_frd = torch.tensor([[v, 0.0, 0.0]])                          # crosswise air-relative velocity, body FRD
    inlet.append(float(-inlet_momentum_drag(mdot, v_frd)[0, 0]))
    body.append(float(-wind.compute_drag_force(frd_to_isaac(v_frd), level)[0, 0]))
    iso.append(0.5 * 1.225 * legacy_iso["cd"] * legacy_iso["area"] * v * v)
    calm.append(0.5 * 1.225 * legacy_calm["cd"] * legacy_calm["area"] * v * v)
total = [a + b for a, b in zip(inlet, body)]
crossing = next(s for s, a, b in zip(speeds.tolist(), inlet, body) if s > 0 and b >= a)

fig, ax = plt.subplots(figsize=(3.45, 2.4))
ax.plot(speeds, total, color=C["black"], label="Current plant: total")
ax.plot(speeds, inlet, color=C["blue"], label=r"Intake momentum drag $\dot m\,v$")
ax.plot(speeds, body, color=C["green"], label=r"Body crossflow, $A_\perp=0.042$ m$^2$")
ax.plot(speeds, iso, color=C["orange"], ls="--", label=r"Before: wind on, $A=0.02$ m$^2$")
ax.plot(speeds, calm, color=C["red"], ls=":", label=r"Before: still air, $A=0.011$ m$^2$")
ax.axvline(crossing, color=C["grey"], lw=0.8, ls=":")
ax.set_xlabel("Crosswise air-relative speed (m/s)")
ax.set_ylabel("Crosswise force (N)")
ax.set_title(f"Hover, $\\dot m$ = {float(mdot):.3f} kg/s", fontsize=8)
ax.legend(loc="upper left", frameon=False, fontsize=6.3)
fig.tight_layout()
fig.savefig(FIGURES / "drag.pdf")

l_in = float(p.inlet[2] - p.com[2])
print(f"hover fraction {hover:.4f}, mdot {float(mdot):.4f} kg/s, inlet-to-COM {l_in:+.3f} m")
print(f"intake = body crossflow drag at {crossing:.2f} m/s; at 1 m/s ratio {inlet[4] / body[4]:.1f}")
print(f"at 4 m/s: intake {inlet[16]:.3f} N, body {body[16]:.3f} N, old iso {iso[16]:.3f} N, old calm {calm[16]:.3f} N; "
      f"intake moment {inlet[16] * abs(l_in):.3f} N m")

# ---- Rotational damping audit (body rate -> restoring moment about the COM), hover --------------
rate = 0.2
damping = {}
for axis, name in ((0, "roll"), (2, "yaw")):
    w = torch.zeros(1, 3, dtype=DTYPE)
    w[0, axis] = rate
    jet, _ = p.jet_at(torch.zeros(1, 4, dtype=DTYPE), hover, body_rate=w)
    still, _ = p.jet_at(torch.zeros(1, 4, dtype=DTYPE), hover)
    m = p.moment_about_com(jet.forces) - p.moment_about_com(still.forces)
    damping[f"vane flow, {name}"] = -float(m[0, axis]) / rate
    cyl = cylinder_rotational_drag(w.float(), p.body["length"], p.body["diameter"], p.body["cd_body"])
    damping[f"cylinder crossflow, {name} (at 1 rad/s)"] = -float(cyl[0, axis]) / rate * (1.0 / rate)
    r_in = (p.inlet - p.com).float()
    f_in = inlet_momentum_drag(mdot, torch.linalg.cross(w.float(), r_in[None]))
    damping[f"intake momentum drag, {name}"] = -float(torch.linalg.cross(r_in[None], f_in)[0, axis]) / rate
    # Duct Coriolis reaction (mission plant since 2026-10-03), at the duct midpoint.
    r_exit = (torch.tensor(p.c["plant"]["dynamics"]["inlet_momentum_drag"]["duct_exit_position_frd"], dtype=DTYPE)
              - p.com).float()
    f_cor = duct_coriolis_force(mdot, w.float(), r_in, r_exit)
    mid = 0.5 * (r_in + r_exit)
    damping[f"duct Coriolis reaction, {name}"] = -float(torch.linalg.cross(mid[None], f_cor)[0, axis]) / rate
exit_z = float(p.jet.exit_position[2] - p.com[2])
vane_z = float(p.cops[0, 2] - p.com[2])
print("rotational damping coefficients (N m s/rad), hover:")
for k, v in damping.items():
    print(f"  {k:40s} {v:.3e}")
for name in ("roll", "yaw"):
    total = sum(v for k, v in damping.items() if k.endswith(name) or k.endswith(f"{name} (at 1 rad/s)"))
    print(f"  modelled total, {name:34s} {total:.3e}   (legacy damper / total: {0.27 / total:.0f}x; "
          f"nutation decay c/I_xx {total / float(p.inertia[0, 0]):.3f} /s)")
print(f"  legacy linear damper                    {0.27:.3e}")
print(f"  classical jet damping mdot*l_e^2, l_e={exit_z:.2f}..{vane_z:.2f} m: "
      f"{float(mdot) * exit_z ** 2:.3e} .. {float(mdot) * vane_z ** 2:.3e}")
print(f"  duct Coriolis damping mdot*(l_e^2-l_in^2) for the same exit stations: "
      f"{float(mdot) * (exit_z ** 2 - l_in ** 2):+.3e} .. {float(mdot) * (vane_z ** 2 - l_in ** 2):+.3e}")
