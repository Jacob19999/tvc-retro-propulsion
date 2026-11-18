"""Fig. battery: sustained hover on the 8S pack with a voltage-compensated duty (as the convex controller flies)."""
import torch

from plant import C, FIGURES, Plant, style

plt = style()
p = Plant("planned_8s")
edf = p.torque_limited_edf()
battery = p.battery()
cfg = battery.config
hover = p.hover_fraction()
dt, decimation = 1.0 / 120.0, 4
omega = torch.tensor([hover * edf.omega_max], dtype=torch.float32)
battery.carry_load(torch.tensor([0]), omega, edf)      # spawn on the loaded bus (reset_logic)

log = dict(t=[], v=[], soc=[], i=[], f=[], temp=[], wh=[])
duty = 0.0
t = 0.0
for step in range(int(600.0 / (dt * decimation))):
    # Flight computer at 30 Hz: invert omega = duty * V_bus / V_ref * omega_max for the hover rotor speed.
    duty = min(1.0, hover * cfg["reference_voltage_v"] / max(float(battery.voltage_v[0]), 1e-3))
    for _ in range(decimation):
        omega = battery.update_motor(edf, omega, torch.tensor([duty]), dt)
        t += dt
    if step % 15 == 0:
        log["t"].append(t); log["v"].append(float(battery.voltage_v[0])); log["soc"].append(float(battery.soc[0]))
        log["i"].append(float(battery.current_a[0])); log["f"].append(float(omega[0]) / edf.omega_max)
        log["temp"].append(float(battery.temperature_c[0])); log["wh"].append(float(battery.energy_wh[0]))
    if bool(battery.cutoff[0]) or float(omega[0]) / edf.omega_max < 0.98 * hover:
        break

fig, ax = plt.subplots(figsize=(3.45, 2.3))
ax.plot(log["t"], log["v"], color=C["blue"], label="Loaded bus voltage")
ax.plot(log["t"], [cfg["cells"] * x for x in [3.3] * len(log["t"])], color=C["grey"], ls=":", lw=0.8)
ax.set_xlabel("Hover time (s)")
ax.set_ylabel("Bus voltage (V)", color=C["blue"])
ax2 = ax.twinx()
ax2.plot(log["t"], [100 * s for s in log["soc"]], color=C["orange"], label="State of charge")
ax2.plot(log["t"], [100 * f for f in log["f"]], color=C["green"], ls="--", label="Rotor fraction")
ax2.set_ylabel("SOC, rotor fraction (%)")
ax2.grid(False)
lines = ax.get_lines()[:1] + ax2.get_lines()
ax.legend(lines, [l.get_label() for l in lines], loc="lower left", frameon=False, fontsize=6.5)
fig.tight_layout()
fig.savefig(FIGURES / "battery.pdf")

end = len(log["t"]) - 1
print(f"hover fraction {hover:.4f}; held for {log['t'][end]:.0f} s; final SOC {log['soc'][end]:.3f}, "
      f"bus {log['v'][0]:.2f} -> {log['v'][end]:.2f} V, current {log['i'][0]:.1f} -> {log['i'][end]:.1f} A, "
      f"energy {log['wh'][end]:.1f} Wh, pack temperature {log['temp'][end]:.1f} C, last duty {duty:.3f}")
print(f"initial electrical power {log['v'][0] * log['i'][0]:.0f} W")
