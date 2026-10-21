"""Plain formats every drone-show and 3D tool can read."""

from __future__ import annotations

import numpy as np

# model frame -> Z-up frame used by drone-show software (and Blender)
Y_UP_TO_Z_UP = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])


def to_z_up(points: np.ndarray, up: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ Y_UP_TO_Z_UP.T if up == "y" else points.copy()


def csv_text(positions: np.ndarray, colors_srgb8: np.ndarray) -> str:
    lines = ["id,x,y,z,r,g,b,hex"]
    for i, (p, c) in enumerate(zip(positions, colors_srgb8)):
        r, g, b = (int(v) for v in c)
        lines.append(f"{i},{p[0]:.4f},{p[1]:.4f},{p[2]:.4f},{r},{g},{b},#{r:02X}{g:02X}{b:02X}")
    return "\n".join(lines) + "\n"


def write_csv(path, positions, colors_srgb8) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(csv_text(positions, colors_srgb8))


def ply_bytes(positions: np.ndarray, colors_srgb8: np.ndarray) -> bytes:
    n = len(positions)
    header = (
        "ply\nformat binary_little_endian 1.0\ncomment Pointille drones\n"
        f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    rec = np.zeros(n, dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    rec["p"] = positions
    rec["c"] = colors_srgb8
    return header + rec.tobytes()


def write_ply(path, positions, colors_srgb8) -> None:
    with open(path, "wb") as f:
        f.write(ply_bytes(positions, colors_srgb8))
