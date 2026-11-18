"""Sensor timing on the physics grid: rounding is reported and warned, never silent."""

import warnings

import pytest
import torch

from tvc_env.dynamics.imu_model import ImuModel, resolve_imu_config
from tvc_env.dynamics.sensor_timing import effective_timing


def _imu(profile, hz):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = ImuModel(1, "cpu", 1.0 / hz, resolve_imu_config({"profile": profile}), torch.Generator().manual_seed(0))
    return model, [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]


def test_mission_step_rounds_the_wtgahrs1_latency_and_caps_its_rate_with_a_warning():
    model, messages = _imu("wtgahrs1", 120)
    assert model.timing["effective_latency_s"] == pytest.approx(1 / 120)      # 5 ms runs as 8.3 ms
    assert model.timing["effective_rate_hz"] == pytest.approx(120.0)          # 200 Hz runs as 120 Hz
    assert model.timing["delay_substeps"] == model._delay
    assert any("latency 5.0 ms runs as 8.3 ms" in m and "200 Hz runs as 120 Hz" in m for m in messages)


def test_vn110e_latency_vanishes_at_the_mission_step():
    model, messages = _imu("vn110e", 120)
    assert model.timing["effective_latency_s"] == 0.0
    assert any("latency 2.0 ms runs as 0.0 ms" in m for m in messages)


def test_a_fine_physics_step_keeps_the_datasheet_timing_without_a_warning():
    model, messages = _imu("vn110e", 960)
    assert model.timing["effective_latency_s"] == pytest.approx(2 / 960)
    assert model.timing["effective_rate_hz"] == pytest.approx(400.0)
    assert not messages


def test_effective_timing_rounds_to_whole_substeps():
    timing = effective_timing(1 / 120, 100.0, 0.015)
    assert timing["delay_substeps"] == 2
    assert timing["effective_latency_s"] == pytest.approx(2 / 120)
    assert timing["effective_rate_hz"] == 100.0
