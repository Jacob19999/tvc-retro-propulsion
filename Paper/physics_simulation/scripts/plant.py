"""Build the mission-plant force models from the repository's own configuration.

Every figure and number in the physics/simulation paper comes from these
objects: the same classes (tvc_env.dynamics.*) and the same YAML files that
Isaac missions load, merged the way mission control merges them
(configs/params -> configs/hardware/planned_8s.yaml -> configs/env/mission_plant.yaml).
No PhysX is involved; these are the pure-tensor models that feed it.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
SIM = REPO / "simulation" / "isaac"
sys.path.insert(0, str(SIM))

from tvc_env.dynamics.coupled_jet import CoupledJet  # noqa: E402
from tvc_env.dynamics.fin_aero import FinAeroModel  # noqa: E402
from tvc_env.dynamics.fin_force_dispatch import FinForceDispatch  # noqa: E402
from tvc_env.dynamics.propulsion_edf import EDFModel  # noqa: E402
from tvc_env.dynamics.battery_lipo import LiPoBattery  # noqa: E402
from tvc_env.dynamics.wind_model import WindModel  # noqa: E402
from tvc_env.envs.task_registry import deep_merge  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures"
GRAVITY = 9.81  # PhysX scene gravity (configs/physics)
DTYPE = torch.float64


def _load(rel: str) -> dict:
    return yaml.safe_load((SIM / rel).read_text(encoding="utf-8"))


def configs(profile: str = "planned_8s") -> dict:
    """Vehicle, EDF, servo, battery, metadata and mission-plant sections for a hardware profile."""
    vehicle = _load("configs/vehicle/edf_drone_v2.yaml")["vehicle"]
    edf = _load("configs/params/edf_90mm.yaml")
    servo = _load("configs/params/servo_mg996r.yaml")
    battery = _load("configs/params/battery_6s.yaml")["battery"]
    if profile == "planned_8s":
        hw = _load("configs/hardware/planned_8s.yaml")
        edf = deep_merge(edf, {"edf": hw["edf"]})
        servo = deep_merge(servo, {"servo": hw["servo"]})
        battery = deep_merge(battery, hw["battery"])
    plant = _load("configs/env/mission_plant.yaml")
    meta = _load("assets/metadata/edf_drone_v2.asset.yaml")["asset"]
    return dict(vehicle=vehicle, edf=edf, servo=servo["servo"], battery=battery,
                plant=plant, metadata=meta)


class Plant:
    """The force models the mission plant flies, for one hardware profile."""

    mass = 3.104  # kg: 3.1 kg Body + four 1 g vanes (USD MassAPI, validate_usd_mass_props.py)

    def __init__(self, profile: str = "planned_8s"):
        c = configs(profile)
        self.c = c
        self.profile = profile
        vehicle_cfg = {"vehicle": c["vehicle"]}
        self.dispatch = FinForceDispatch.from_metadata_and_config(c["metadata"], vehicle_cfg, c["edf"], dtype=DTYPE)
        self.aero: FinAeroModel = self.dispatch.aero_model
        self.cops = self.dispatch.cop_positions                      # (4, 3) FRD, m
        self.normals = self.dispatch.normal_dirs
        e = c["edf"]["edf"]
        self.edf = EDFModel(max_thrust=e["max_thrust"], tau_motor=e["tau_motor"], omega_max=e["omega_max"],
                            k_T=e.get("k_T"), k_Q=e.get("k_Q"), rotor_inertia=e["rotor_inertia"])
        limit = c["plant"]["dynamics"]["motor_torque_limit"]
        self.motor_limit_nm = float(limit["max_torque_nm"])
        self.shaft_power = float(c["battery"]["shaft_power_at_max_w"])
        jet_cfg = dict(c["plant"]["dynamics"]["coupled_jet"])
        jet_cfg["shaft_power_at_max_w"] = self.shaft_power
        self.jet = CoupledJet(jet_cfg, self.aero, self.normals)
        self.com = torch.tensor(c["vehicle"]["body_com_offset"], dtype=DTYPE)   # FRD, m
        self.inertia = torch.diag(torch.tensor([c["vehicle"]["inertia_tensor"][k] for k in ("Ixx", "Iyy", "Izz")],
                                               dtype=DTYPE))
        self.inlet = torch.tensor(c["plant"]["dynamics"]["inlet_momentum_drag"]["inlet_position_frd"], dtype=DTYPE)
        self.body = c["vehicle"]["body"]

    # ---- configuration helpers ------------------------------------------------
    def torque_limited_edf(self, brake: bool = False) -> EDFModel:
        """EDF with the mission plant's current-limited motor (base_env._initialize_physics_systems)."""
        edf = copy.copy(self.edf)
        edf.max_motor_torque = self.motor_limit_nm
        edf.aero_torque_at_max = self.shaft_power / edf.omega_max
        edf.zero_throttle_brake = brake
        return edf

    def battery(self, n: int = 1) -> LiPoBattery:
        cfg = dict(self.c["battery"])
        cfg["max_current_a"] = min(cfg["max_current_a"], 120.0)   # mission_control.models.battery_config cap
        return LiPoBattery(cfg, n)

    def wind(self) -> WindModel:
        return WindModel(**WindModel.body_drag_from_vehicle(self.body), num_envs=1)

    # ---- jet evaluation ----------------------------------------------------------
    def jet_at(self, angles, fraction: float = 1.0, body_rate=None, cop_velocity=None):
        """Coupled-jet output for vane angles (n, 4) at a rotor fraction, vehicle otherwise at rest."""
        angles = torch.as_tensor(angles, dtype=DTYPE)
        if angles.ndim == 1:
            angles = angles[None]
        n = angles.shape[0]
        frac = torch.full((n,), float(fraction), dtype=DTYPE)
        omega = frac * self.edf.omega_max
        raw = self.edf.compute_thrust(omega)
        rate = torch.zeros(n, 3, dtype=DTYPE) if body_rate is None else torch.as_tensor(body_rate, dtype=DTYPE).expand(n, 3)
        cops = self.cops[None].expand(n, -1, -1)
        if cop_velocity is None:
            cop_velocity = torch.linalg.cross(rate[:, None].expand(-1, 4, -1), cops)
        return self.jet.compute(angles, frac, omega, raw, cops, cop_velocity, rate), raw

    def moment_about_com(self, forces):
        """Sum of r x F about the nominal COM for per-vane forces (n, 4, 3)."""
        return torch.linalg.cross((self.cops - self.com)[None].expand_as(forces), forces).sum(1)

    def neutral_net_thrust(self, fraction: float = 1.0) -> float:
        out, raw = self.jet_at(torch.zeros(1, 4), fraction)
        return float(raw[0] - out.forces[0, :, 2].sum())

    def hover_fraction(self) -> float:
        """Level-hover rotor fraction (TVCDirectRLEnv.nominal_hover_throttle)."""
        return (self.mass * GRAVITY / self.neutral_net_thrust(1.0)) ** 0.5

    def vane_authority(self, effort: float = 0.035) -> tuple[float, float]:
        """(roll/pitch, yaw) torque per rad of effort at full rotor speed (TVCDirectRLEnv.vane_authority)."""
        def torque(angles):
            out, _ = self.jet_at(torch.tensor([angles], dtype=DTYPE), 1.0)
            return self.moment_about_com(out.forces)[0]
        e = effort
        roll = float(torque([-e, 0, e, 0])[0] - torque([e, 0, -e, 0])[0]) / (2 * e)
        yaw = float(torque([e] * 4)[2] - torque([-e] * 4)[2]) / (2 * e)
        return roll, yaw

    def vehicle_model(self):
        """The flight computer's VehicleModel, built as apps/run_mission.py builds it."""
        from tvc_env.controllers.convex_adapter import VehicleModel
        roll, yaw = self.vane_authority()
        b = self.c["battery"]
        s = self.c["servo"]
        return VehicleModel(
            mass_kg=self.mass, hover_throttle=self.hover_fraction(), reference_voltage_v=float(b["reference_voltage_v"]),
            shaft_power_at_max_w=float(b["shaft_power_at_max_w"]), motor_efficiency=float(b["motor_efficiency"]),
            auxiliary_power_w=float(b["auxiliary_power_w"]), rotor_inertia=self.edf.rotor_inertia,
            omega_max=self.edf.omega_max, vane_authority_nm_per_rad=roll, yaw_authority_nm_per_rad=yaw,
            inertia_rp_kg_m2=float(self.inertia[0, 0]), inertia_yaw_kg_m2=float(self.inertia[2, 2]),
            angular_damping_nm_s_per_rad=float(self.c["plant"]["dynamics"]["body_angular_damping"]),
            servo_lag_s=float(s["tau_servo"]),
            vane_joint_lag_s=float(s["drive_damping"]) / float(s["drive_stiffness"]))

    def convex_controller(self, touchdown_height_m: float = 0.3125, settings: dict | None = None):
        """ConvexGuidanceController with the repository settings, at the 30 Hz mission control step."""
        from tvc_env.controllers.convex_adapter import ConvexGuidanceController
        settings = settings or _load("configs/controllers/convex_guidance.yaml")
        return ConvexGuidanceController(settings, self.vehicle_model(), (0.0, 0.0, 0.0), touchdown_height_m,
                                        1.0 / 30.0, max_command_angle=float(self.c["servo"]["max_command_angle"]),
                                        servo_deadband_rad=float(self.c["servo"]["deadband"]))


def style():
    """Matplotlib settings for single-column IEEE figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.2, "axes.grid": True,
        "grid.linewidth": 0.3, "grid.alpha": 0.5, "figure.dpi": 150, "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })
    return plt


# Colour-blind-safe series colours (Okabe-Ito).
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00", purple="#CC79A7",
         sky="#56B4E9", black="#000000", grey="#7f7f7f")
