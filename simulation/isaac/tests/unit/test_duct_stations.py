"""The duct stations the plant uses match the authored USD geometry (needs the USD bindings)."""
from pathlib import Path

import pytest
import yaml

pytest.importorskip('pxr')

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def stations():
    import sys
    sys.path.insert(0, str(ROOT / 'tools'))
    from measure_duct_stations import measure
    return measure(ROOT / 'assets/usd/drone_v2_physics.usd')


def test_intake_is_the_open_duct_lip_not_the_top_of_the_frame(stations):
    inlet = yaml.safe_load((ROOT / 'configs/env/mission_plant.yaml').read_text())['dynamics']['inlet_momentum_drag']
    assert stations['points_over_intake'] == 0
    assert inlet['inlet_position_frd'][2] == pytest.approx(stations['intake_lip_frd_z'], abs=0.005)
    # The top of the whole mesh is the frame, well above the duct (the 2026-09-29 inlet).
    assert stations['mesh_top_frd_z'] < stations['intake_lip_frd_z'] - 0.05


def test_nozzle_exit_matches_the_jet_exit_and_the_vanes_sit_in_the_free_jet(stations):
    dynamics = yaml.safe_load((ROOT / 'configs/env/mission_plant.yaml').read_text())['dynamics']
    exit_z = dynamics['inlet_momentum_drag']['duct_exit_position_frd'][2]
    assert exit_z == pytest.approx(stations['nozzle_exit_frd_z'], abs=0.01)
    assert dynamics['coupled_jet']['exit_position_frd'][2] == pytest.approx(stations['nozzle_exit_frd_z'], abs=0.01)
    # The coupled jet's frozen wake assumes the vanes are downstream of the nozzle.
    for vane in stations['vanes'].values():
        assert vane['frd_z'][0] > stations['nozzle_exit_frd_z']
