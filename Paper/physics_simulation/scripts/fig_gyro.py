"""Fig. gyro: energy and world angular momentum of the unforced body + rotor under three integrators.

Offline replica of the integrator split, not PhysX: the locked-body Euler term
is advanced explicitly (as PhysX does), and the rotor torque is supplied by
the production functions in tvc_env.dynamics.rotor_reaction. PhysX's
additional angular-momentum-magnitude projection is not reproduced here; the
production 'coupled_cayley' mode inverts it (correct_physx_projection=True).
"""
import torch

from plant import C, DTYPE, FIGURES, Plant, style
from tvc_env.dynamics.rotor_reaction import (cayley_body_orientation, compute_coupled_midpoint_torques,
                                             compute_midpoint_gyroscopic_torque)
from tvc_env.common.frames import frd_to_isaac
from tvc_env.common.quaternions import multiply, normalize, rotate_vector

plt = style()
p = Plant("planned_8s")
I = p.inertia[None]                                   # (1, 3, 3) FRD
I_inv = torch.linalg.inv(I)
spin = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
omega_rotor = p.hover_fraction() * p.edf.omega_max
H = (spin * p.edf.rotor_inertia * omega_rotor)[None]   # (1, 3) rotor angular momentum, FRD


def energy(w):
    return float(0.5 * (w[0] @ I[0] @ w[0]))


def euler_body(w):
    return -torch.linalg.cross(w, (I @ w[..., None])[..., 0])


def exp_quat(q, w_frd, dt):
    """First-order orientation update from one body rate (the drift Cayley replaces)."""
    v = frd_to_isaac(w_frd) * dt
    angle = v.norm(dim=-1, keepdim=True).clamp(min=1e-15)
    dq = torch.cat((torch.cos(angle / 2), torch.sin(angle / 2) * v / angle), dim=-1)
    return normalize(multiply(q, dq))


def world_momentum(q, w):
    body = (I @ w[..., None])[..., 0] + H
    return rotate_vector(q, frd_to_isaac(body))


def simulate(method, hz, w0, seconds=2.0, orientation="exp"):
    dt = 1.0 / hz
    w = torch.tensor([w0], dtype=DTYPE)
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=DTYPE)
    L0 = world_momentum(q, w)
    t, e, dl = [0.0], [1.0], [0.0]
    e0 = energy(w)
    for k in range(int(round(seconds * hz))):
        if method == "explicit":
            tau = -torch.linalg.cross(w, H)
        elif method == "rotor_midpoint":
            tau = compute_midpoint_gyroscopic_torque(torch.tensor([omega_rotor], dtype=DTYPE), w,
                                                     p.edf.rotor_inertia, spin, I, dt)
        else:
            gyro, corr = compute_coupled_midpoint_torques(w, I, H, torch.zeros_like(w), dt)
            tau = gyro + corr
        w_new = w + dt * (I_inv @ (tau + euler_body(w))[..., None])[..., 0]
        if orientation == "cayley":
            q = cayley_body_orientation(q, w, w_new, dt)
        else:
            q = exp_quat(q, w_new, dt)
        w = w_new
        t.append((k + 1) * dt)
        e.append(energy(w) / e0)
        dl.append(float((world_momentum(q, w) - L0).norm()))
    return t, e, dl


w0 = [6.28, 6.28, -40.0]   # yaw counter to the rotor spin: the unstable case
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.4))
for method, color, label in (("explicit", C["red"], r"Explicit $-\omega\times H$"),
                             ("rotor_midpoint", C["orange"], "Rotor-only midpoint"),
                             ("coupled", C["blue"], "Coupled implicit midpoint")):
    for hz, ls in ((120, "-"), (480, "--")):
        t, e, _ = simulate(method, hz, w0)
        ax1.semilogy(t, e, color=color, ls=ls, label=f"{label}, {hz} Hz")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel(r"$E_{\mathrm{rot}}(t)/E_{\mathrm{rot}}(0)$")
ax1.set_title(r"(a) Unforced, $\omega_0=(6.28, 6.28, -40)$ rad/s, hover rotor", fontsize=8)
ax1.legend(loc="upper left", frameon=False, fontsize=6)

for orient, color, label in (("exp", C["purple"], "Coupled midpoint + end-rate orientation"),
                             ("cayley", C["blue"], "Coupled midpoint + Cayley orientation")):
    t, _, dl = simulate("coupled", 120, w0, orientation=orient)
    ax2.semilogy(t[1:], [max(x, 1e-16) for x in dl[1:]], color=color, label=label)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel(r"$\|L_W(t)-L_W(0)\|$ (N$\,$m$\,$s)")
ax2.set_title("(b) World angular momentum, 120 Hz", fontsize=8)
ax2.legend(loc="center right", frameon=False, fontsize=6.5)
fig.tight_layout()
fig.savefig(FIGURES / "gyro.pdf")

for method in ("explicit", "rotor_midpoint", "coupled"):
    for hz in (120, 480):
        for wz in (40.0, -40.0):
            _, e, _ = simulate(method, hz, [6.28, 6.28, wz])
            print(f"{method:15s} {hz:4d} Hz r={wz:+.0f}: E(2s)/E0 = {e[-1]:.6g}")
for orient in ("exp", "cayley"):
    _, _, dl = simulate("coupled", 120, w0, orientation=orient)
    print(f"world L drift after 2 s ({orient}): {dl[-1]:.3e} N m s")
