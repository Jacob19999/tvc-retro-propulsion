"""Fig. vane_force: per-vane side force and roll authority, legacy airfoil vs momentum-bounded jet."""
import math

import torch

from plant import C, DTYPE, FIGURES, Plant, style

plt = style()
p = Plant("planned_8s")
alpha = torch.linspace(0.0, 0.262, 60, dtype=DTYPE)

# (a) One vane deflected (FwdFin), full rotor speed, vehicle at rest.
angles = torch.zeros(len(alpha), 4, dtype=DTYPE)
angles[:, 0] = alpha
legacy = p.dispatch.compute_body_frame_forces(angles, torch.ones(len(alpha), dtype=DTYPE)).forces_body[:, 0, :2].norm(dim=-1)
jet, raw = p.jet_at(angles, 1.0)
zero, _ = p.jet_at(torch.zeros(1, 4, dtype=DTYPE), 1.0)
momentum = (jet.forces[:, 0, :2] - zero.forces[:, 0, :2]).norm(dim=-1)   # remove the de-swirl offset
quarter = float(raw[0]) / 4
bound = quarter * torch.sin(alpha)
q_area = .5 * p.aero.air_density * p.aero.exhaust_speed ** 2 * p.aero.fin_area * p.aero.duct_confinement_factor
turn = 1 - math.exp(-q_area * p.aero.C_N_alpha / quarter)
drag = 1 - math.exp(-q_area * p.aero.C_D_0 / quarter)
closed = quarter * (turn - drag) * torch.sin(alpha) * torch.cos(alpha)

# (b) Roll authority (dM_x/de of the -e/+e X-vane pair) against rotor fraction.
fractions = torch.linspace(0.2, 1.0, 17, dtype=DTYPE)
e = 0.035
auth_jet, auth_leg = [], []
for f in fractions.tolist():
    plus, _ = p.jet_at(torch.tensor([[-e, 0, e, 0]], dtype=DTYPE), f)
    minus, _ = p.jet_at(torch.tensor([[e, 0, -e, 0]], dtype=DTYPE), f)
    auth_jet.append(float(p.moment_about_com(plus.forces)[0, 0] - p.moment_about_com(minus.forces)[0, 0]) / (2 * e))
    lp = p.dispatch.compute_body_frame_forces(torch.tensor([[-e, 0, e, 0]], dtype=DTYPE), torch.tensor([f], dtype=DTYPE))
    lm = p.dispatch.compute_body_frame_forces(torch.tensor([[e, 0, -e, 0]], dtype=DTYPE), torch.tensor([f], dtype=DTYPE))
    auth_leg.append(float(p.moment_about_com(lp.forces_body)[0, 0] - p.moment_about_com(lm.forces_body)[0, 0]) / (2 * e))
hover = p.hover_fraction()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.4))
deg = torch.rad2deg(alpha)
ax1.plot(deg, legacy, color=C["red"], label=r"Legacy airfoil $qSk\,C_N(\alpha)$")
ax1.plot(deg, bound, color=C["grey"], ls="--", label=r"Streamtube bound $(T/4)\sin\alpha$")
ax1.plot(deg, momentum, color=C["blue"], label="Coupled jet (code)")
ax1.plot(deg[::6], closed[::6], "o", ms=3, mfc="none", color=C["black"], label=r"$(T/4)(\tau_t-\delta_D)\sin\alpha\cos\alpha$")
ax1.set_xlabel("Vane deflection (deg)")
ax1.set_ylabel("Side force on one vane (N)")
ax1.set_title("(a) Full rotor speed, $T=48$ N", fontsize=8)
ax1.legend(loc="upper left", frameon=False)

ax2.semilogy(fractions, auth_leg, color=C["red"], label="Legacy airfoil")
ax2.semilogy(fractions, auth_jet, color=C["blue"], label="Coupled jet")
ax2.axvline(hover, color=C["grey"], ls=":", lw=1)
ax2.text(hover - 0.02, 0.3, f"hover\n{hover:.3f}", ha="right", fontsize=7, color=C["grey"])
ax2.set_xlabel(r"Rotor fraction $\omega/\omega_{\max}$")
ax2.set_ylabel(r"Roll authority (N$\,$m/rad)")
ax2.set_title(r"(b) $\partial M_x/\partial e$ about the COM", fontsize=8)
ax2.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(FIGURES / "vane_force.pdf")

print(f"turn={turn:.5f} drag={drag:.4f} loading={q_area * p.aero.C_N_alpha / quarter:.3f}")
print(f"15 deg: legacy {float(legacy[-1]):.2f} N, momentum {float(momentum[-1]):.3f} N, closed {float(closed[-1]):.3f} N, bound {float(bound[-1]):.3f} N")
print(f"roll authority full: jet {auth_jet[-1]:.3f}, legacy {auth_leg[-1]:.3f}, ratio {auth_leg[-1] / auth_jet[-1]:.2f}")
