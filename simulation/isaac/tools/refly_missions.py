"""Re-fly recorded mission-control missions on the current plant and compare the outcomes.

Each recorded mission (runs/mission_control/<id>/) is restored from its request
and resolved metadata, exactly as mission control restores it, flown again
through apps/run_mission.py, and compared with its recording: outcome, impact
speed, pad distance, landing time and energy, plus every dynamics/physics
setting that differs between the two runs. Use it to validate a plant change
in Isaac (for example the 2026-10-05 duct stations and Coriolis reaction).

Run with the Isaac Python (the same interpreter that runs missions), from simulation/isaac:

    python tools/refly_missions.py --benchmark
    python tools/refly_missions.py 408e6b926eee c8233615eabf --out runs/refly/duct_review
    python tools/refly_missions.py --benchmark --dry-run     # restore and diff only, no Isaac

Missions are flown one at a time. Requests the current mission service rejects
(legacy vanes, PID) are reported as skipped, not flown.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mission_control.models import read_json, recorded_request, validate_mission, vane_model_overrides  # noqa: E402

RUNS = ROOT / "runs/mission_control"
# Momentum-vane convex benchmark (docs/convex_guidance_2026-09-25.md, fourth pass, final controller)
# and the two missions re-flown by physics review 2026-09-29.
BENCHMARK = {
    "408e6b926eee": "Default 18 m",
    "e2ba15c45abb": "Hot 3 m hop",
    "49cc16aca7f5": "7 m offset + crosswind",
    "fdb559daede1": "Legacy 6S pack, 8 m",
    "c8233615eabf": "50 m start",
    "0036c82ca31d": "Route: fly-through + hover",
    "1b9c6c33b85b": "Trial: 50 m, three hovers",
    "d4e2b8d8ceac": "Wind + sensor noise + CoM shift",
    "a93eb786cb50": "High-energy capture (2026-09-29)",
    "614281ca9910": "Hover 10 m in gusts (2026-09-29)",
}
METRICS = ("outcome", "success", "impact_speed", "pad_distance", "landing_duration_s", "flight_energy_wh")


def flatten(value, prefix: str = "") -> dict:
    """Nested mapping -> {'a.b.c': leaf}."""
    if not isinstance(value, dict):
        return {prefix: value}
    out = {}
    for key, item in value.items():
        out.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    return out


def plant_diff(before: dict, after: dict, after_keys_only: bool = False) -> list[tuple[str, object, object]]:
    """Dynamics/physics settings that differ: (key, before, after).

    after_keys_only compares only the settings `after` defines (a partial plant, such as
    the mission-plant overlay, against a fully resolved recording).
    """
    a = flatten({k: before.get(k) or {} for k in ("dynamics", "physics")})
    b = flatten({k: after.get(k) or {} for k in ("dynamics", "physics")})
    keys = set(b) if after_keys_only else set(a) | set(b)
    return [(key, a.get(key), b.get(key)) for key in sorted(keys) if a.get(key) != b.get(key)]


def restore(mission_id: str, runs: Path = RUNS) -> tuple[dict | None, str | None]:
    """The validated request to re-fly, or (None, reason) when it cannot be flown now."""
    source = runs / mission_id
    if not (source / "metadata.json").exists() and not (source / "request.json").exists():
        return None, "no recording"
    try:
        return validate_mission(recorded_request(source)), None
    except (ValueError, TypeError, KeyError) as exc:
        return None, f"rejected by the current mission service: {exc}"


def fly(request: dict, output: Path, python: str = sys.executable) -> int:
    """One Isaac mission through apps/run_mission.py; returns its exit code."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
    with (output / "isaac.log").open("wb") as log:
        return subprocess.run([python, str(ROOT / "apps/run_mission.py"), "--request", str(output / "request.json"),
                               "--output", str(output)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode


def compare(mission_id: str, label: str, before_dir: Path, after_dir: Path | None, dry_run: bool) -> dict:
    """Before/after metrics and the plant settings that changed."""
    before_meta = read_json(before_dir / "metadata.json", {}) or {}
    before = read_json(before_dir / "summary.json", {}) or {}
    if dry_run or after_dir is None:
        # What the next flight will change: the mission plant sections it would fly.
        after_meta, after = vane_model_overrides({}), {}
    else:
        after_meta = read_json(after_dir / "metadata.json", {}) or {}
        after = read_json(after_dir / "summary.json", {}) or {}
    return dict(id=mission_id, label=label,
                before={k: before.get(k) for k in METRICS}, after={k: after.get(k) for k in METRICS},
                plant_changes=[dict(key=k, before=a, after=b) for k, a, b in
                               plant_diff(before_meta, after_meta, after_keys_only=dry_run or after_dir is None)])


def _cell(value):
    if isinstance(value, float):
        return f"{value:.3f}"
    return "—" if value is None else str(value)


def markdown(rows: list[dict]) -> str:
    lines = ["| Mission | Before | After | Impact m/s (before → after) | Pad m (before → after) | Energy Wh (before → after) |",
             "|---|---|---|---|---|---|"]
    for row in rows:
        if row.get("skipped"):
            lines.append(f"| {row['label']} (`{row['id']}`) | skipped: {row['skipped']} | | | | |")
            continue
        b, a = row["before"], row["after"]
        lines.append(f"| {row['label']} (`{row['id']}`) | {_cell(b['outcome'])} | {_cell(a['outcome'])} | "
                     f"{_cell(b['impact_speed'])} → {_cell(a['impact_speed'])} | "
                     f"{_cell(b['pad_distance'])} → {_cell(a['pad_distance'])} | "
                     f"{_cell(b['flight_energy_wh'])} → {_cell(a['flight_energy_wh'])} |")
    changes = {(c["key"], json.dumps(c["before"]), json.dumps(c["after"]))
               for row in rows for c in row.get("plant_changes", [])}
    if changes:
        lines += ["", "Plant settings that differ from the recordings:", ""]
        lines += [f"- `{key}`: {before} → {after}" for key, before, after in sorted(changes)]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ids", nargs="*", help="recorded mission ids (runs/mission_control/<id>)")
    parser.add_argument("--benchmark", action="store_true", help="the momentum-vane convex benchmark and 2026-09-29 re-flights")
    parser.add_argument("--runs", type=Path, default=RUNS, help="recorded missions directory")
    parser.add_argument("--out", type=Path, default=None, help="output directory (default runs/refly/<timestamp>)")
    parser.add_argument("--dry-run", action="store_true", help="restore and diff the plant only; fly nothing")
    args = parser.parse_args(argv)
    missions = dict(BENCHMARK) if args.benchmark else {}
    missions.update({i: missions.get(i, i) for i in args.ids})
    if not missions:
        parser.error("give mission ids or --benchmark")
    out = args.out or ROOT / "runs/refly" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (mission_id, label) in enumerate(missions.items(), 1):
        request, reason = restore(mission_id, args.runs)
        if request is None:
            print(f"[{index}/{len(missions)}] {mission_id} {label}: skipped ({reason})", flush=True)
            rows.append(dict(id=mission_id, label=label, skipped=reason))
            continue
        after_dir = None
        if not args.dry_run:
            print(f"[{index}/{len(missions)}] {mission_id} {label}: flying...", flush=True)
            after_dir = out / mission_id
            code = fly(request, after_dir)
            if code != 0:
                print(f"    run_mission.py exited {code}; see {after_dir / 'isaac.log'}", flush=True)
        row = compare(mission_id, label, args.runs / mission_id, after_dir, args.dry_run)
        rows.append(row)
        print(f"    {row['before']['outcome']} -> {row['after']['outcome']}", flush=True)
    (out / "refly_report.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    report = markdown(rows)
    (out / "refly_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Report: {out / 'refly_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
