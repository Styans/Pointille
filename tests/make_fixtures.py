"""Regenerate tests/fixtures/quadrant_sphere.glb (used by the browser e2e test).

A UV sphere at (0, 2, 0) whose texture has four solid quadrants:
top-left red, top-right green, bottom-left blue, bottom-right white.
"""

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from helpers import quadrant_texture


def quadrant_sphere_glb() -> bytes:
    sphere = trimesh.creation.uv_sphere(radius=1.0, count=[32, 64])
    v = sphere.vertices / np.linalg.norm(sphere.vertices, axis=1, keepdims=True)
    u = (np.arctan2(v[:, 1], v[:, 0]) / (2 * np.pi)) % 1.0
    w = np.arccos(np.clip(v[:, 2], -1, 1)) / np.pi
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(quadrant_texture(128)), baseColorFactor=[255, 255, 255, 255]
    )
    sphere.visual = trimesh.visual.TextureVisuals(uv=np.stack([u, 1.0 - w], axis=1), material=material)
    scene = trimesh.Scene()
    scene.add_geometry(sphere, node_name="model", transform=trimesh.transformations.translation_matrix([0, 2, 0]))
    return scene.export(file_type="glb")


def write_obj(folder: Path) -> None:
    """The same sphere as OBJ + MTL + PNG (three.js loads these with flipY on)."""
    import io

    folder.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(io.BytesIO(quadrant_sphere_glb()), file_type="glb").to_geometry()
    mesh.export(str(folder / "ball.obj"))


if __name__ == "__main__":
    fixtures = Path(__file__).parent / "fixtures"
    (fixtures / "quadrant_sphere.glb").write_bytes(quadrant_sphere_glb())
    write_obj(fixtures / "obj")
    print(fixtures)
