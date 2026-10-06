"""Fig. spool: rotor response and body spool reaction for three motor models."""
import torch

from plant import C, FIGURES, Plant, style

plt = style()
p = Plant("planned_8s")
dt = 1.0 / 120.0           # mission physics step (configs/env/single_env_debug.yaml)
t_end = 7.0
steps = int(round(t_end / dt))
hover = p.hover_fraction()


def duty(t):
    """Cold spool-up at full duty, settle to hover duty, then disarm (zero throttle)."""
    if t < 2.5:
        return 1.0
    if t < 4.5:
        return hover
    return 0.0


def run(edf, battery=None):
    omega = torch.zeros(1, dtype=torch.float32)
    t_hist, f_hist, tau_hist, v_hist, i_hist = [], [], [], [], []
    for k in range(steps):
        t = k * dt
        throttle = torch.tensor([duty(t)], dtype=torch.float32)
        if battery is None:
            new = edf.update(omega, throttle, dt)
        else:
            new = battery.update_motor(edf, omega, throttle, dt)
            v_hist.append(float(battery.voltage_v[0]))
            i_hist.append(float(battery.current_a[0]))
        tau_hist.append(-edf.rotor_inertia * float(new - omega) / dt)   # spool reaction on the body (N m, FRD z)
        omega = new
        t_hist.append(t + dt)
        f_hist.append(float(omega) / edf.omega_max)
    return t_hist, f_hist, tau_hist, v_hist, i_hist


free = run(p.edf)
limited = run(p.torque_limited_edf())
battery = p.battery()
coupled = run(p.torque_limited_edf(), battery)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.4))
for (t, f, *_), color, label in ((free, C["red"], r"First-order, $\tau_m=0.15$ s (unbounded)"),
                                 (limited, C["orange"], r"Torque-limited, $|Q_m|\leq 0.76$ N$\,$m"),
                                 (coupled, C["blue"], "Torque-limited + 8S LiPo (mission plant)")):
    ax1.plot(t, f, color=color, label=label)
ax1.axhline(hover, color=C["grey"], ls=":", lw=1)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel(r"Rotor fraction $\omega/\omega_{\max}$")
ax1.set_ylim(0, 1.05)
ax1.set_title("(a) Duty 1.0, then hover duty, then 0 at 4.5 s", fontsize=8)

for (t, _, tau, *_), color in ((free, C["red"]), (limited, C["orange"]), (coupled, C["blue"])):
    ax2.plot(t, tau, color=color)
yaw_max = 0.348   # common-mode vane yaw torque at 15 deg, hover rotor speed (fig_vane_force / plant.py)
ax2.axhspan(-yaw_max, yaw_max, color=C["green"], alpha=0.15, lw=0)
ax2.text(6.9, yaw_max + 0.25, "vane yaw authority at hover", ha="right", fontsize=6.5, color=C["green"])
ax2.set_xlabel("Time (s)")
ax2.set_ylabel(r"Spool reaction $-I_r\dot\omega$ (N$\,$m)")
ax2.set_ylim(-7, 6)
ax2.set_title("(b) Yaw torque the rotor puts on the body", fontsize=8)
fig.legend(*ax1.get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(FIGURES / "spool.pdf")


def first_time(t, f, level):
    return next(tt for tt, ff in zip(t, f) if ff >= level)


for name, (t, f, tau, v, i) in (("free", free), ("limited", limited), ("coupled", coupled)):
    print(f"{name}: t(0.84)={first_time(t, f, hover):.3f}s, peak spool-up torque {min(tau):.2f} N m, "
          f"peak cut torque {max(tau):.2f} N m, fraction 1 s after cut {f[int(round(5.5 / dt)) - 1]:.3f}")
print(f"coupled: min bus {min(coupled[3]):.2f} V, peak current {max(coupled[4]):.1f} A, "
      f"bus at hover {coupled[3][int(round(4.4 / dt))]:.2f} V")
