"""Measure the EDF duct's intake and nozzle stations and the vane extents from the USD mesh.

The intake momentum drag (dynamics.inlet_momentum_drag.inlet_position_frd) and
the duct Coriolis reaction (duct_exit_position_frd, coupled_jet.exit_position_frd)
need the duct's axial stations. This reads them from the authored geometry:
the duct wall is the band of mesh points at the duct radius that covers the
full circle; above and below it the same radius holds only discrete frame
rods and the four vane mounts.

Run with any Python that has the USD bindings (Isaac's, or `pip install usd-core`):

    python tools/measure_duct_stations.py [assets/usd/drone_v2_physics.usd]

Positions are printed in body FRD relative to the Body origin (FRD z = -Isaac z).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DUCT_RADIUS_BAND = (0.043, 0.060)   # m: 90 mm duct wall, inner to outer
SLICE_M = 0.005
FULL_RING = 33                       # of 36 ten-degree azimuth bins: a continuous wall


def body_frame_points(stage, prim, body_inv):
    from pxr import UsdGeom
    cache = UsdGeom.XformCache()
    matrix = np.array(cache.GetLocalToWorldTransform(prim) * body_inv, dtype=float)
    points = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
    return (np.c_[points, np.ones(len(points))] @ matrix)[:, :3]


def measure(usd_path: Path) -> dict:
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(usd_path))
    root = stage.GetDefaultPrim().GetPath()
    body = stage.GetPrimAtPath(root.AppendChild("Body"))
    body_inv = UsdGeom.XformCache().GetLocalToWorldTransform(body).GetInverse()
    meshes = [p for p in Usd.PrimRange(body) if p.IsA(UsdGeom.Mesh)]
    points = np.vstack([body_frame_points(stage, p, body_inv) for p in meshes])
    r, z = np.hypot(points[:, 0], points[:, 1]), points[:, 2]
    azimuth = np.degrees(np.arctan2(points[:, 1], points[:, 0])) % 360.0
    band = (r >= DUCT_RADIUS_BAND[0]) & (r < DUCT_RADIUS_BAND[1])
    full = []
    for top in np.arange(z.max(), z.min(), -SLICE_M):
        sel = band & (z < top) & (z >= top - SLICE_M)
        bins = np.histogram(azimuth[sel], bins=np.arange(0.0, 361.0, 10.0))[0]
        if (bins > 0).sum() >= FULL_RING:
            full.append((top - SLICE_M, top))
    if not full:
        raise RuntimeError("No continuous duct wall found at the duct radius")
    wall_top = max(hi for _, hi in full)
    wall_bottom = min(lo for lo, _ in full)
    # Refine the ends to the extreme wall points inside the end slices.
    ring = band & (z <= wall_top) & (z >= wall_bottom)
    lip, exit_ = float(z[ring].max()), float(z[ring].min())
    vanes = {}
    for name in ("FwdFin", "RightFin", "AftFin", "LeftFin"):
        prim = stage.GetPrimAtPath(root.AppendChild(name))
        q = np.vstack([body_frame_points(stage, p, body_inv) for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Mesh)])
        span = np.hypot(q[:, 0], q[:, 1])
        vanes[name] = dict(frd_z=(-float(q[:, 2].max()), -float(q[:, 2].min())),
                           radius=(float(span.min()), float(span.max())))
    blocked = int(((r < DUCT_RADIUS_BAND[0]) & (z > lip + 0.006)).sum())
    return dict(intake_lip_frd_z=-lip, nozzle_exit_frd_z=-exit_, vanes=vanes, points_over_intake=blocked,
                mesh_top_frd_z=-float(z.max()))


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "assets/usd/drone_v2_physics.usd"
    m = measure(path)
    print(f"duct intake lip   FRD z = {m['intake_lip_frd_z']:+.3f} m  (continuous wall top)")
    print(f"duct nozzle exit  FRD z = {m['nozzle_exit_frd_z']:+.3f} m  (continuous wall bottom)")
    print(f"mesh top          FRD z = {m['mesh_top_frd_z']:+.3f} m  (frame, not the duct)")
    print(f"mesh points inside the duct radius above the lip: {m['points_over_intake']}")
    for name, v in m["vanes"].items():
        print(f"{name:9s} FRD z {v['frd_z'][0]:+.3f}..{v['frd_z'][1]:+.3f} m, radius {v['radius'][0]:.3f}..{v['radius'][1]:.3f} m")


if __name__ == "__main__":
    main()
