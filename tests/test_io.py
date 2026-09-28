import csv
import json

import numpy as np
import pytest

from pointille.io.export import csv_text, ply_bytes, to_z_up
from pointille.io.gltf_append import append_drones, empty_glb, read_glb

trimesh = pytest.importorskip("trimesh")


def drones(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)), rng.random((n, 3))


def test_append_keeps_original_bytes_and_json(textured_glb):
    original = textured_glb.read_bytes()
    doc0, bin0 = read_glb(original)
    pos, col = drones()
    doc1, bin1 = read_glb(append_drones(original, pos, col, extras={"count": 50}))

    assert bin1[: len(bin0)] == bin0
    for key, items in doc0.items():
        if isinstance(items, list) and key not in ("scenes", "buffers"):  # both only grow
            assert doc1[key][: len(items)] == items, key
    roots0 = doc0["scenes"][0]["nodes"]
    assert doc1["scenes"][0]["nodes"][: len(roots0)] == roots0
    for key in ("images", "textures", "samplers"):
        assert doc1.get(key) == doc0.get(key)
    assert doc1["buffers"][0]["byteLength"] == len(bin1)

    node = doc1["nodes"][-1]
    assert node["name"] == "Pointille_Drones"
    assert node["extras"]["pointille"]["count"] == 50
    prim = doc1["meshes"][node["mesh"]]["primitives"][0]
    assert prim["mode"] == 0 and set(prim["attributes"]) == {"POSITION", "COLOR_0"}
    assert len(doc1["nodes"]) - 1 in doc1["scenes"][doc1.get("scene", 0)]["nodes"]


def test_appended_glb_loads(textured_glb):
    pos, col = drones()
    out = textured_glb.parent / "loaded.glb"
    out.write_bytes(append_drones(textured_glb.read_bytes(), pos, col))
    scene = trimesh.load(str(out), force="scene")
    kinds = sorted(type(g).__name__ for g in scene.geometry.values())
    assert kinds == ["PointCloud", "Trimesh"]
    cloud = next(g for g in scene.geometry.values() if type(g).__name__ == "PointCloud")
    assert len(cloud.vertices) == 50
    assert np.allclose(np.sort(cloud.vertices, axis=0), np.sort(pos, axis=0), atol=1e-5)


def test_append_with_scale_wraps_roots(textured_glb):
    original = textured_glb.read_bytes()
    doc0, _ = read_glb(original)
    pos, col = drones()
    doc1, _ = read_glb(append_drones(original, pos, col, scale=10.0, pivot=[0, -1, 0]))
    root = doc1["nodes"][-1]
    assert root["name"] == "Pointille_Root"
    assert root["scale"] == [10.0] * 3
    assert np.allclose(root["translation"], [0, 9, 0])
    assert doc1["scenes"][0]["nodes"] == [len(doc1["nodes"]) - 1]
    assert set(root["children"]) == set(doc0["scenes"][0]["nodes"]) | {len(doc1["nodes"]) - 2}
    for i, n in enumerate(doc0["nodes"]):
        assert doc1["nodes"][i] == n


def test_points_only_glb():
    pos, col = drones(10)
    doc, bin_chunk = read_glb(append_drones(empty_glb(), pos, col))
    assert doc["buffers"][0]["byteLength"] == len(bin_chunk) == 10 * 24
    assert doc["scenes"][0]["nodes"] == [0]


def test_csv_and_ply():
    pos = np.array([[1.0, 2.0, 3.0], [0.5, 0.0, -1.0]])
    col = np.array([[255, 0, 16], [1, 2, 3]], np.uint8)
    rows = list(csv.DictReader(csv_text(pos, col).splitlines()))
    assert rows[0] == {"id": "0", "x": "1.0000", "y": "2.0000", "z": "3.0000",
                       "r": "255", "g": "0", "b": "16", "hex": "#FF0010"}
    data = ply_bytes(pos, col)
    header, body = data.split(b"end_header\n")
    assert b"element vertex 2" in header
    rec = np.frombuffer(body, dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    assert np.allclose(rec["p"], pos) and np.array_equal(rec["c"], col)


def test_y_up_to_z_up():
    # glTF front (+Z) becomes the show frame's -Y, glTF up (+Y) becomes +Z
    assert np.allclose(to_z_up([[0, 0, 1], [0, 1, 0]], "y"), [[0, -1, 0], [0, 0, 1]])
