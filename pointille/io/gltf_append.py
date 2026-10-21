"""Add a drone point cloud to a GLB without touching the original model.

Only *appends*: new bytes go after the existing binary chunk and new entries
go at the end of the JSON arrays, so every existing mesh, material, texture
and image keeps its data and its index.
"""

from __future__ import annotations

import base64
import json
import struct
from typing import Optional, Tuple

import numpy as np

GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942
ARRAY_BUFFER = 34962
FLOAT = 5126


class GlbError(ValueError):
    pass


def read_glb(data: bytes) -> Tuple[dict, Optional[bytes]]:
    if len(data) < 20:
        raise GlbError("file is too short to be a GLB")
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC:
        raise GlbError("not a binary glTF (.glb) file")
    if version != 2:
        raise GlbError(f"unsupported glTF version {version}")
    offset, doc, bin_chunk = 12, None, None
    while offset + 8 <= min(length, len(data)):
        size, kind = struct.unpack_from("<II", data, offset)
        body = data[offset + 8 : offset + 8 + size]
        if kind == CHUNK_JSON and doc is None:
            doc = json.loads(body.decode("utf-8"))
        elif kind == CHUNK_BIN and bin_chunk is None:
            bin_chunk = bytes(body)
        offset += 8 + size
    if doc is None:
        raise GlbError("GLB has no JSON chunk")
    return doc, bin_chunk


def write_glb(doc: dict, bin_chunk: Optional[bytes]) -> bytes:
    js = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    js += b" " * (-len(js) % 4)
    out = [struct.pack("<II", len(js), CHUNK_JSON), js]
    if bin_chunk is not None:
        bin_chunk = bin_chunk + b"\0" * (-len(bin_chunk) % 4)
        out += [struct.pack("<II", len(bin_chunk), CHUNK_BIN), bin_chunk]
    body = b"".join(out)
    return struct.pack("<III", GLB_MAGIC, 2, 12 + len(body)) + body


def empty_glb() -> bytes:
    return write_glb({"asset": {"version": "2.0", "generator": "Pointille"}, "scenes": [{"nodes": []}], "scene": 0}, None)


def append_drones(
    glb: bytes,
    positions: np.ndarray,
    colors_linear: np.ndarray,
    name: str = "Pointille_Drones",
    scale: float = 1.0,
    pivot=None,
    extras: Optional[dict] = None,
) -> bytes:
    """Return a new GLB = original + a POINTS mesh with per-drone colors.

    ``positions`` are in the scene's world space (glTF, Y-up). With
    ``scale != 1`` the original root nodes and the drones are put under a new
    root node that scales everything about ``pivot``.
    """
    doc, bin_chunk = read_glb(glb)
    positions = np.ascontiguousarray(positions, dtype="<f4").reshape(-1, 3)
    colors = np.ascontiguousarray(np.clip(colors_linear, 0, 1), dtype="<f4").reshape(-1, 3)
    if len(positions) != len(colors):
        raise ValueError("positions and colors differ in length")

    for key in ("buffers", "bufferViews", "accessors", "meshes", "materials", "nodes", "scenes"):
        doc.setdefault(key, [])

    payload = positions.tobytes() + colors.tobytes()
    buffers = doc["buffers"]
    embedded = bin_chunk is not None and buffers and "uri" not in buffers[0]
    if embedded or (bin_chunk is None and not buffers):
        if bin_chunk is None:
            bin_chunk = b""
            buffers.append({"byteLength": 0})
        base = len(bin_chunk) + (-len(bin_chunk) % 4)
        bin_chunk = bin_chunk + b"\0" * (base - len(bin_chunk)) + payload
        buffers[0]["byteLength"] = len(bin_chunk)
        buf_index = 0
    else:  # first buffer is external: carry ours as a data URI
        base = 0
        buffers.append({
            "byteLength": len(payload),
            "uri": "data:application/octet-stream;base64," + base64.b64encode(payload).decode(),
        })
        buf_index = len(buffers) - 1

    views = doc["bufferViews"]
    views.append({"buffer": buf_index, "byteOffset": base, "byteLength": positions.nbytes, "target": ARRAY_BUFFER})
    views.append({"buffer": buf_index, "byteOffset": base + positions.nbytes, "byteLength": colors.nbytes, "target": ARRAY_BUFFER})
    acc = doc["accessors"]
    acc.append({
        "bufferView": len(views) - 2, "componentType": FLOAT, "count": len(positions), "type": "VEC3",
        "min": positions.min(axis=0).tolist() if len(positions) else [0, 0, 0],
        "max": positions.max(axis=0).tolist() if len(positions) else [0, 0, 0],
    })
    acc.append({"bufferView": len(views) - 1, "componentType": FLOAT, "count": len(colors), "type": "VEC3"})

    doc["materials"].append({
        "name": name,
        "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0, "roughnessFactor": 1},
        "extensions": {"KHR_materials_unlit": {}},
    })
    used = doc.setdefault("extensionsUsed", [])
    if "KHR_materials_unlit" not in used:
        used.append("KHR_materials_unlit")
    doc["meshes"].append({
        "name": name,
        "primitives": [{
            "attributes": {"POSITION": len(acc) - 2, "COLOR_0": len(acc) - 1},
            "mode": 0,
            "material": len(doc["materials"]) - 1,
        }],
    })
    nodes = doc["nodes"]
    drone_node = {"name": name, "mesh": len(doc["meshes"]) - 1}
    if extras:
        drone_node["extras"] = {"pointille": extras}
    nodes.append(drone_node)
    drone_index = len(nodes) - 1

    scenes = doc["scenes"]
    if not scenes:
        children = {c for n in nodes for c in n.get("children", ())}
        scenes.append({"nodes": [i for i in range(len(nodes) - 1) if i not in children]})
        doc["scene"] = 0
    scene = scenes[doc.get("scene", 0)]
    roots = scene.setdefault("nodes", [])
    if scale != 1.0:
        p = np.zeros(3) if pivot is None else np.asarray(pivot, dtype=np.float64)
        nodes.append({
            "name": "Pointille_Root",
            "translation": (p * (1.0 - scale)).tolist(),
            "scale": [float(scale)] * 3,
            "children": list(roots) + [drone_index],
        })
        scene["nodes"] = [len(nodes) - 1]
    else:
        roots.append(drone_index)
    return write_glb(doc, bin_chunk)


def required_extensions(glb: bytes) -> list:
    doc, _ = read_glb(glb)
    return list(doc.get("extensionsRequired", []))
