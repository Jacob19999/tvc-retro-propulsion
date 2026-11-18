"""Fig. imu: what the flight computer receives from the IMU chain (tvc_env.dynamics.imu_model).

(a) Body roll rate during a 1.07 Hz, 15 deg/s coning-like oscillation: truth vs the delivered gyro register
    of two hardware profiles (low-pass, sample-and-hold, latency, sensor errors).
(b) Tilt error of the onboard attitude filter during a sustained 10 deg bank with the consistent horizontal
    acceleration g tan(10 deg): accelerometer aiding drags the estimate toward level.
Kinematics are scripted (no PhysX); the IMU chain is the production code at the 120 Hz mission step.
"""
import math

import torch

from plant import C, FIGURES, style
from tvc_env.dynamics.imu_model import ImuModel, resolve_imu_config, STANDARD_GRAVITY

plt = style()
dt = 1.0 / 120.0
torch.manual_seed(7)


def make(profile):
    generator = torch.Generator().manual_seed(11)
    return ImuModel(1, "cpu", dt, resolve_imu_config({"profile": profile}), generator)


def quat_roll(phi):
    return torch.tensor([[math.cos(phi / 2), math.sin(phi / 2), 0.0, 0.0]])


def tilt_between(q_a, q_b):
    """Angle between the body-up axes of two Isaac wxyz attitudes (rad)."""
    def up(q):
        w, x, y, z = q[0].tolist()
        return torch.tensor([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    return float(torch.arccos(torch.clamp(up(q_a) @ up(q_b), -1.0, 1.0)))


# ---- (a) oscillating roll rate ------------------------------------------------------------------
freq, amp = 1.07, math.radians(15.0)
rate_series = {}
for profile in ("wtgahrs1", "vn110e"):
    imu = make(profile)
    phi = 0.0
    zero = torch.zeros(1, 3)
    imu.reset(torch.tensor([0]), quat_roll(phi), zero, zero)
    t_hist, true_hist, meas_hist = [], [], []
    for k in range(int(3.0 / dt)):
        t = (k + 1) * dt
        p_rate = amp * math.sin(2 * math.pi * freq * t)
        phi += p_rate * dt
        # Isaac roll about +x is FRD roll about +x (x is shared); FRD rate = (p, 0, 0).
        imu.step(quat_roll(phi), zero, torch.tensor([[p_rate, 0.0, 0.0]]))
        t_hist.append(t)
        true_hist.append(math.degrees(float(imu.true_rate_frd[0, 0])))
        meas_hist.append(math.degrees(float(imu.gyro_frd[0, 0])))
    rate_series[profile] = (t_hist, true_hist, meas_hist)

# ---- (b) sustained banked acceleration ----------------------------------------------------------
bank = math.radians(10.0)
tilt_series = {}
for profile in ("wtgahrs1", "bno085", "vn110e"):
    imu = make(profile)
    q = quat_roll(bank)
    accel = STANDARD_GRAVITY * math.tan(bank)
    imu.reset(torch.tensor([0]), q, torch.zeros(1, 3), torch.zeros(1, 3))
    t_hist, err_hist = [], []
    for k in range(int(20.0 / dt)):
        t = (k + 1) * dt
        v = torch.tensor([[0.0, -accel * t, 0.0]])     # body up leans toward world -y for +roll
        imu.step(q, v, torch.zeros(1, 3))
        if k % 4 == 0:
            t_hist.append(t)
            err_hist.append(math.degrees(tilt_between(imu.quaternion_wxyz, q)))
    tilt_series[profile] = (t_hist, err_hist)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.4))
for profile, color, ls, label in (("wtgahrs1", C["red"], "-", "WTGAHRS1: 20 Hz LPF, 120 Hz out, 8.3 ms"),
                                  ("vn110e", C["blue"], "--", "VN-110E: 240 Hz LPF, 120 Hz out, 0 ms")):
    t, truth, meas = rate_series[profile]
    ax1.plot(t, [m - r for m, r in zip(meas, truth)], color=color, ls=ls, label=label)
ax1.set_xlim(1.0, 3.0)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Gyro error (deg/s)")
ax1.set_title("(a) Delivered gyro error, 1.07 Hz, 15 deg/s roll", fontsize=8)
ax1.set_ylim(-1.7, 2.4)
ax1.legend(loc="upper center", frameon=False, fontsize=6.3, ncol=1)
for profile, color, label in (("wtgahrs1", C["red"], "WTGAHRS1"), ("bno085", C["orange"], "BNO085"),
                              ("vn110e", C["blue"], "VN-110E")):
    ax2.plot(*tilt_series[profile], color=color, label=label)
ax2.axhline(10.0, color=C["grey"], ls=":", lw=0.8)
ax2.set_xlabel("Time in the bank (s)")
ax2.set_ylabel("Onboard tilt error (deg)")
ax2.set_title(r"(b) 10 deg bank, $a=g\tan 10^\circ$", fontsize=8)
ax2.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(FIGURES / "imu.pdf")


def rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs))


for profile in ("wtgahrs1", "vn110e"):
    t, truth, meas = rate_series[profile]
    sel = [i for i, tt in enumerate(t) if tt > 1.0]
    print(f"{profile}: gyro error rms {rms([meas[i] - truth[i] for i in sel]):.2f} deg/s on a "
          f"{rms([truth[i] for i in sel]):.2f} deg/s rms signal")
for profile, (t, err) in tilt_series.items():
    at = {s: err[min(range(len(t)), key=lambda i: abs(t[i] - s))] for s in (1, 2, 5, 10, 20)}
    print(f"{profile}: tilt error " + ", ".join(f"{s}s {v:.1f} deg" for s, v in at.items()))
