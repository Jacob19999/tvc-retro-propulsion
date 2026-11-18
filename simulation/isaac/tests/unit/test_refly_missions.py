"""tools/refly_missions.py: restore recorded missions, re-fly them, compare (Isaac replaced by a stub)."""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import refly_missions as refly  # noqa: E402
from mission_control.models import default_mission, validate_mission, vane_model_overrides  # noqa: E402


def _record(runs, mission_id, request, summary, dynamics=None):
    """A recorded mission as run_mission.py and the mission service leave it."""
    path = runs / mission_id
    path.mkdir(parents=True)
    old_plant = copy.deepcopy(vane_model_overrides({}))
    old_plant['dynamics']['inlet_momentum_drag'] = dict(enabled=True, inlet_position_frd=[0.0, 0.0, -0.11])
    metadata = dict(request=request, dynamics=dynamics if dynamics is not None else old_plant['dynamics'],
                    physics=old_plant['physics'])
    (path / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    (path / 'request.json').write_text(json.dumps(request), encoding='utf-8')
    (path / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    return path


@pytest.fixture
def runs(tmp_path):
    runs = tmp_path / 'runs'
    request = validate_mission(default_mission())
    _record(runs, 'aaaaaaaaaaaa', request,
            dict(outcome='LANDED', success=True, impact_speed=.178, pad_distance=.017, landing_duration_s=8.3,
                 flight_energy_wh=5.4))
    legacy = dict(request, vane_model='legacy')
    _record(runs, 'bbbbbbbbbbbb', legacy, dict(outcome='LANDED', success=True))
    return runs


def test_restore_returns_the_validated_request_and_skips_what_the_service_rejects(runs):
    request, reason = refly.restore('aaaaaaaaaaaa', runs)
    assert reason is None and request == validate_mission(default_mission())
    request, reason = refly.restore('bbbbbbbbbbbb', runs)
    assert request is None and 'rejected' in reason
    assert refly.restore('cccccccccccc', runs) == (None, 'no recording')


def test_dry_run_lists_the_mission_plant_changes_against_the_recording(runs, tmp_path, monkeypatch):
    monkeypatch.setattr(refly, 'fly', lambda *a, **k: pytest.fail('a dry run must not fly'))
    out = tmp_path / 'out'
    assert refly.main(['aaaaaaaaaaaa', 'bbbbbbbbbbbb', '--runs', str(runs), '--out', str(out), '--dry-run']) == 0
    rows = {row['id']: row for row in json.loads((out / 'refly_report.json').read_text())}
    assert 'rejected' in rows['bbbbbbbbbbbb']['skipped']
    changes = {c['key']: (c['before'], c['after']) for c in rows['aaaaaaaaaaaa']['plant_changes']}
    assert changes['dynamics.inlet_momentum_drag.inlet_position_frd'] == ([0.0, 0.0, -0.11], [0.0, 0.0, -0.017])
    assert changes['dynamics.inlet_momentum_drag.duct_coriolis'] == (None, True)
    # Only what the mission plant defines is compared; unchanged settings are not listed.
    assert 'dynamics.coupled_jet.enabled' not in changes
    report = (out / 'refly_report.md').read_text()
    assert 'inlet_position_frd' in report and 'skipped' in report


def test_reflight_compares_outcomes_and_the_flown_plant(runs, tmp_path, monkeypatch):
    flown = []

    def fake_fly(request, output, python=sys.executable):
        flown.append(request['name'])
        output.mkdir(parents=True, exist_ok=True)
        (output / 'metadata.json').write_text(json.dumps(dict(request=request, **vane_model_overrides({}))))
        (output / 'summary.json').write_text(json.dumps(dict(outcome='LANDED', success=True, impact_speed=.170,
                                                             pad_distance=.030, landing_duration_s=8.4,
                                                             flight_energy_wh=5.5)))
        return 0

    monkeypatch.setattr(refly, 'fly', fake_fly)
    out = tmp_path / 'out'
    refly.main(['aaaaaaaaaaaa', 'bbbbbbbbbbbb', '--runs', str(runs), '--out', str(out)])
    assert flown == [default_mission()['name']]          # the rejected legacy request is not flown
    row = next(r for r in json.loads((out / 'refly_report.json').read_text()) if r['id'] == 'aaaaaaaaaaaa')
    assert row['before']['impact_speed'] == .178 and row['after']['impact_speed'] == .170
    keys = {c['key'] for c in row['plant_changes']}
    assert {'dynamics.inlet_momentum_drag.inlet_position_frd', 'dynamics.inlet_momentum_drag.duct_coriolis'} <= keys
    assert '0.178 → 0.170' in (out / 'refly_report.md').read_text()


def test_benchmark_lists_the_documented_momentum_plant_missions():
    assert len(refly.BENCHMARK) == 10
    assert all(len(mission_id) == 12 and int(mission_id, 16) >= 0 for mission_id in refly.BENCHMARK)
