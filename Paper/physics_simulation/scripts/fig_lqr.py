"""Fig. lqr: closed-loop roll/pitch growth rate of the PD law and the gyroscopic LQR against rotor fraction.

Both laws act on the attitude model the flight computer identifies (AttitudePlant: transverse
inertia, rotor momentum H = I_r omega_max f, vane authority K f^2, servo lag and vane-joint lag),
discretized with a zero-order hold at the 30 Hz control period, as attitude_lqr.py designs it.
The PD law is the mission's pre-2026-09-26 law (kp 0.5, kd 0.08, gyro cross-feed 0.03), with its
efforts scaled from the legacy reference authority to the momentum vanes.
"""
import cmath
import math

import numpy as np
import yaml

from plant import C, FIGURES, SIM, Plant, style
from tvc_env.controllers.attitude_lqr import _zoh

plt = style()
p = Plant("planned_8s")
ctrl = p.convex_controller()
lqr = ctrl._lqr
att = yaml.safe_load((SIM / "configs/controllers/convex_guidance.yaml").read_text())["attitude"]
kp, kd, comp = att["kp_att"], att["kd_att"], att["gyro_comp_rp"]
scale = att["reference_vane_authority_nm_per_rad"] / ctrl.vehicle.vane_authority_nm_per_rad
dt = ctrl.dt


def pd_modes(f):
    A, B = lqr.roll_pitch_model(f)
    Ad, Bd = _zoh(A, B, dt)
    gain = np.zeros((2, 10))     # u = scale [-kd p - kp e_x + comp q, -kd q - kp e_y - comp p]
    gain[0, 2], gain[0, 4], gain[0, 5] = -kp, -kd, comp
    gain[1, 3], gain[1, 5], gain[1, 4] = -kp, -kd, -comp
    closed = Ad[2:, 2:] + Bd[2:] @ (scale * gain)[:, 2:]   # the PD law has no integral state
    out = []
    for z in np.linalg.eigvals(closed):
        s = cmath.log(complex(z)) / dt
        out.append((s.real, abs(s.imag) / (2 * math.pi)))
    return sorted(out, reverse=True)


def damping_ratio(mode):
    sigma, freq = mode
    wn = math.hypot(sigma, 2 * math.pi * freq)
    return -sigma / wn if wn > 0 else 1.0


fractions = np.linspace(0.3, 1.0, 29)
pd_growth, lqr_growth, lqr_zeta = [], [], []
for f in fractions:
    pd_growth.append(pd_modes(f)[0][0])
    modes = lqr.modes(f)
    lqr_growth.append(modes[0][0])
    oscillatory = [m for m in modes if m[1] > 0.5]
    lqr_zeta.append(min(damping_ratio(m) for m in oscillatory))
hover = p.hover_fraction()

fig, ax = plt.subplots(figsize=(3.45, 2.2))
ax.axhline(0.0, color=C["black"], lw=0.8)
ax.plot(fractions, pd_growth, color=C["red"], label="PD law (pre-2026-09-26)")
ax.plot(fractions, lqr_growth, color=C["blue"], label="Gyroscopic LQR")
ax.axvline(hover, color=C["grey"], ls=":", lw=0.8)
ax.set_xlabel(r"Rotor fraction $\omega/\omega_{\max}$")
ax.set_ylabel(r"Slowest-mode growth rate (s$^{-1}$)")
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(FIGURES / "lqr.pdf")

hover_pd = pd_modes(hover)
hover_lqr = lqr.modes(hover)
osc = [m for m in hover_lqr if m[1] > 0.5]
print(f"hover {hover:.3f}: PD slowest {hover_pd[0][0]:+.2f} /s at {hover_pd[0][1]:.2f} Hz; "
      f"unstable PD modes: {[(round(a, 2), round(b, 2)) for a, b in hover_pd if a > 0]}")
print(f"LQR slowest modes at hover: {[(round(a, 2), round(b, 2)) for a, b in hover_lqr[:4]]}; "
      f"min oscillatory damping ratio {min(damping_ratio(m) for m in osc):.2f}")
print(f"PD growth over 0.3-1.0: {min(pd_growth):+.2f} .. {max(pd_growth):+.2f} /s; "
      f"LQR slowest decay over 0.3-1.0: {max(lqr_growth):+.2f} .. {min(lqr_growth):+.2f} /s; "
      f"min LQR damping ratio over 0.3-1.0 {min(lqr_zeta):.2f}")
gain = lqr.roll_pitch_gain(hover)
print("LQR roll row at hover [z_x z_y e_x e_y p q s_x s_y j_x j_y]:", np.round(gain[0], 3).tolist())
print(f"rate-command slew bound {math.degrees(ctrl._slew_rate):.1f} deg/s; planned tilt-rate bound "
      f"{math.degrees(ctrl._limits.tilt_rate_rad_s):.2f} deg/s; yaw reserve {ctrl._yaw_reserve:.3f} rad of "
      f"{ctrl._vane_limit:.3f} rad vane limit")
