"""Intake momentum drag of the EDF (physics review 2026-09-29)."""

import math
from pathlib import Path

import torch
import yaml

from tvc_env.dynamics.propulsion_edf import inlet_momentum_drag

ROOT = Path(__file__).resolve().parents[2]


def test_force_opposes_air_relative_inlet_velocity_and_never_adds_energy():
    torch.manual_seed(0)
    mass_flow = torch.rand(64) * 0.5
    velocity = torch.randn(64, 3) * 4.0
    force = inlet_momentum_drag(mass_flow, velocity)
    assert torch.allclose(force, -mass_flow[:, None] * velocity)
    power = (force * velocity).sum(-1)
    assert torch.all(power <= 0.0)
    assert torch.allclose(power, -mass_flow * velocity.square().sum(-1))


def test_hover_crosswind_drag_dominates_body_form_drag_at_low_speed():
    # Planned 8S: 48 N / 128 m/s at full rotor, hover thrust ~ weight.
    weight = 3.104 * 9.81
    fraction = math.sqrt(weight / 48.0)
    mass_flow = weight / (128.0 * fraction)             # T / u, u scales with rotor speed
    ram = float(inlet_momentum_drag(torch.tensor([mass_flow]), torch.tensor([[1.0, 0.0, 0.0]]))[0, 0])
    body = yaml.safe_load((ROOT / 'configs/vehicle/edf_drone_v2.yaml').read_text())['vehicle']['body']
    form = 0.5 * 1.225 * body['cd_lateral'] * body['lateral_reference_area'] * 1.0 ** 2
    assert 0.25 < -ram < 0.35
    assert -ram > 8.0 * form


def test_mission_plant_enables_it_at_the_inlet_above_the_com():
    plant = yaml.safe_load((ROOT / 'configs/env/mission_plant.yaml').read_text())
    inlet = plant['dynamics']['inlet_momentum_drag']
    assert inlet['enabled'] is True
    vehicle = yaml.safe_load((ROOT / 'configs/vehicle/edf_drone_v2.yaml').read_text())['vehicle']
    # FRD z is down: the intake lip sits above the COM, so a crosswind pitches the nose away from it.
    assert inlet['inlet_position_frd'][2] < vehicle['body_com_offset'][2]


def _duct(n=32, seed=1):
    torch.manual_seed(seed)
    mass_flow = torch.rand(n, dtype=torch.float64) * 0.5
    rate = torch.randn(n, 3, dtype=torch.float64) * 3.0
    velocity = torch.randn(n, 3, dtype=torch.float64) * 4.0
    inlet = torch.tensor([0.0, 0.0, -0.10], dtype=torch.float64)   # relative to the COM, FRD
    exit = torch.tensor([0.0, 0.0, 0.06], dtype=torch.float64)
    return mass_flow, rate, velocity, inlet, exit


def test_duct_coriolis_force_is_the_integrated_coriolis_reaction_at_the_duct_midpoint():
    from tvc_env.dynamics.propulsion_edf import duct_coriolis_force
    mass_flow, rate, _, inlet, exit = _duct()
    force = duct_coriolis_force(mass_flow, rate, inlet, exit)
    # Midpoint-rule integral of -r x (2 w x a v_d) dm along the duct, dm = mdot / v_d ds.
    s = (torch.arange(2000, dtype=torch.float64) + 0.5) / 2000
    points = inlet + s[:, None] * (exit - inlet)
    ds = float((exit - inlet).norm()) / 2000
    axis = (exit - inlet) / (exit - inlet).norm()
    density = -2.0 * mass_flow[:, None, None] * torch.linalg.cross(rate[:, None], axis.expand(1, 2000, 3)) * ds
    assert torch.allclose(density.sum(1), force, atol=1e-12)
    moment = torch.linalg.cross(points.expand_as(density), density).sum(1)
    midpoint = 0.5 * (inlet + exit)
    assert torch.allclose(moment, torch.linalg.cross(midpoint.expand_as(force), force), atol=1e-9)


def test_intake_plus_duct_coriolis_gives_classical_jet_damping():
    from tvc_env.dynamics.propulsion_edf import duct_coriolis_force
    mass_flow, rate, _, inlet, exit = _duct()
    rate[:, 2] = 0.0                                          # transverse rotation about the COM
    intake = inlet_momentum_drag(mass_flow, torch.linalg.cross(rate, inlet.expand_as(rate)))
    coriolis = duct_coriolis_force(mass_flow, rate, inlet, exit)
    midpoint = 0.5 * (inlet + exit)
    moment = (torch.linalg.cross(inlet.expand_as(intake), intake)
              + torch.linalg.cross(midpoint.expand_as(coriolis), coriolis))
    # -mdot l_exit^2 w: set by the nozzle station only. The intake term alone gave -mdot l_in^2 w.
    assert torch.allclose(moment, -mass_flow[:, None] * exit[2] ** 2 * rate, atol=1e-12)


def test_intake_plus_duct_coriolis_acts_like_nozzle_momentum_drag_and_never_adds_energy():
    from tvc_env.dynamics.propulsion_edf import duct_coriolis_force
    mass_flow, rate, velocity, inlet, exit = _duct()
    inlet_velocity = velocity + torch.linalg.cross(rate, inlet.expand_as(rate))
    midpoint = 0.5 * (inlet + exit)
    power = ((inlet_momentum_drag(mass_flow, inlet_velocity) * inlet_velocity).sum(-1)
             + (duct_coriolis_force(mass_flow, rate, inlet, exit)
                * (velocity + torch.linalg.cross(rate, midpoint.expand_as(rate)))).sum(-1))
    nozzle_velocity = velocity + torch.linalg.cross(rate, exit.expand_as(rate))
    assert torch.allclose(power, -mass_flow * nozzle_velocity.square().sum(-1), atol=1e-10)
    assert torch.all(power <= 1e-12)


def test_mission_plant_adds_the_duct_coriolis_reaction_at_the_jet_exit():
    plant = yaml.safe_load((ROOT / 'configs/env/mission_plant.yaml').read_text())
    inlet = plant['dynamics']['inlet_momentum_drag']
    assert inlet['duct_coriolis'] is True
    assert inlet['duct_exit_position_frd'] == plant['dynamics']['coupled_jet']['exit_position_frd']
    assert inlet['duct_exit_position_frd'][2] > inlet['inlet_position_frd'][2]


def _env_stub(duct_coriolis):
    """Just what TVCDirectRLEnv._inlet_momentum_drag reads: a level vehicle at rest, still air."""
    from types import SimpleNamespace
    from tvc_env.common.frames import isaac_velocity_to_frd
    plant = yaml.safe_load((ROOT / 'configs/env/mission_plant.yaml').read_text())
    com_frd = yaml.safe_load((ROOT / 'configs/vehicle/edf_drone_v2.yaml').read_text())['vehicle']['body_com_offset']
    config = dict(plant['dynamics']['inlet_momentum_drag'], duct_coriolis=duct_coriolis)
    env = SimpleNamespace(
        _config=SimpleNamespace(config=dict(dynamics=plant['dynamics'])),
        _body_iface=SimpleNamespace(get_root_linear_velocity_world=lambda: torch.zeros(1, 3),
                                    get_linear_velocity_body_frd=lambda v, q: isaac_velocity_to_frd(v)),
        _wind_model=None,
        _drone=SimpleNamespace(data=SimpleNamespace(
            body_com_pos_w=torch.tensor([[[com_frd[0], -com_frd[1], -com_frd[2]]]]))),
        _art_map=SimpleNamespace(body_index=0),
        _aero_model=None)
    return env, config, torch.tensor(com_frd)


def test_environment_applies_classical_jet_damping_about_the_com():
    from types import SimpleNamespace
    from tvc_env.envs.direct_rl_env import TVCDirectRLEnv
    mass_flow, rate = 0.3157, torch.tensor([[0.4, -0.3, 0.0]])
    jet = SimpleNamespace(mass_flow_per_fin=torch.full((1, 4), mass_flow / 4))
    q, pos = torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.zeros(1, 3)
    moments = {}
    for coriolis in (False, True):
        env, config, com = _env_stub(coriolis)
        force, torque_origin = TVCDirectRLEnv._inlet_momentum_drag(
            env, config, torch.tensor([30.0]), torch.tensor([0.84]), jet, q, pos, rate)
        # The wrench dispatcher moves the origin moment to the COM: + (origin - COM) x F.
        moments[coriolis] = torque_origin + torch.linalg.cross(-com.expand_as(force), force)
        if coriolis:
            assert torch.allclose(env._duct_coriolis_debug, force - inlet_momentum_drag(
                torch.tensor([mass_flow]), torch.linalg.cross(rate, (torch.tensor(config['inlet_position_frd']) - com)[None])))
    l_in = config['inlet_position_frd'][2] - float(com[2])        # -0.10 m: intake above the COM
    l_exit = config['duct_exit_position_frd'][2] - float(com[2])  # +0.06 m: nozzle below it
    assert torch.allclose(moments[False], -mass_flow * l_in ** 2 * rate, atol=1e-7)
    assert torch.allclose(moments[True], -mass_flow * l_exit ** 2 * rate, atol=1e-7)


def test_physx_default_does_not_clamp_physical_body_rates():
    from tvc_env.sim.scene_builder import SceneConfig
    cfg = SceneConfig.from_yaml({'env': {}, 'physics': {}})
    # A warm rotor spawn hands the body ~680 deg/s of yaw; the old 100 deg/s
    # default silently clipped (and removed energy from) every faster rotation.
    assert cfg.max_angular_velocity_deg_s >= 3600.0
