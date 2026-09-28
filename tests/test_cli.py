import csv

import numpy as np
import pytest

from helpers import QUADRANTS
from pointille.cli import main, run, build_parser
from pointille.io.gltf_append import read_glb

trimesh = pytest.importorskip("trimesh")


def cli(*argv):
    return run(build_parser().parse_args([str(a) for a in argv]))


def test_cli_end_to_end(textured_glb, tmp_path):
    s = cli(textured_glb, "-n", 200, "--color", "point", "-o", tmp_path)
    assert s["count"] == 200 and s["scale"] == 1.0
    glb = tmp_path / "model_pointille.glb"
    rows = list(csv.DictReader((tmp_path / "model_drones.csv").open()))
    assert len(rows) == 200
    assert (tmp_path / "model_drones.ply").stat().st_size > 200 * 15

    scene = trimesh.load(str(glb), force="scene")
    cloud = next(g for g in scene.geometry.values() if isinstance(g, trimesh.PointCloud))
    r = np.linalg.norm(cloud.vertices - [0, 2, 0], axis=1)
    assert np.all((r > 0.99) & (r < 1.0 + 1e-5))  # on the (translated) sphere

    checked = 0
    for row in rows:
        # CSV is Z-up: (x, y, z)_csv = (x, -z, y)_gltf
        gy, gz = float(row["z"]) - 2.0, -float(row["y"])
        if min(abs(gz), abs(gy)) < 0.2:
            continue
        quad = ("top" if gz > 0 else "bottom") + "_" + ("left" if gy > 0 else "right")
        assert (int(row["r"]), int(row["g"]), int(row["b"])) == QUADRANTS[quad]
        checked += 1
    assert checked > 60


def test_cli_scales_to_min_distance(textured_glb, tmp_path):
    s = cli(textured_glb, "-n", 100, "--min-dist", 2.0, "-o", tmp_path)
    assert s["min_distance"] == pytest.approx(2.0)
    doc, _ = read_glb((tmp_path / "model_pointille.glb").read_bytes())
    assert doc["nodes"][-1]["name"] == "Pointille_Root"
    pts = np.array([[float(r["x"]), float(r["y"]), float(r["z"])]
                    for r in csv.DictReader((tmp_path / "model_drones.csv").open())])
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2) + np.eye(len(pts)) * 1e9
    assert d.min() == pytest.approx(2.0, abs=1e-3)


def test_cli_facade_and_obj(tmp_path, capsys):
    box = trimesh.creation.box((2, 2, 2))
    path = tmp_path / "box.obj"
    box.export(str(path))
    assert main([str(path), "-n", "60", "--facade", "--view", "front"]) == 0
    assert "Дронов: 60" in capsys.readouterr().out
    pts = np.array([[float(r["x"]), float(r["y"]), float(r["z"])]
                    for r in csv.DictReader((tmp_path / "box_drones.csv").open())])
    # OBJ is read as Y-up: its +Z face is the front, i.e. y = -1 in the Z-up CSV
    assert np.allclose(pts[:, 1], -1.0, atol=1e-4)
    assert (tmp_path / "box_drones.glb").exists()


def test_cli_rejects_missing_file(tmp_path):
    with pytest.raises(SystemExit):
        cli(tmp_path / "nope.glb", "-n", 5)
