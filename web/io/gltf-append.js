// Add a drone point cloud to a GLB without touching the original model:
// bytes are appended after the binary chunk and new entries go at the end of
// the JSON arrays. Mirrors pointille/io/gltf_append.py.

const GLB_MAGIC = 0x46546c67;
const CHUNK_JSON = 0x4e4f534a;
const CHUNK_BIN = 0x004e4942;

export function readGlb(buffer) {
  const view = new DataView(buffer);
  if (buffer.byteLength < 20 || view.getUint32(0, true) !== GLB_MAGIC) throw new Error("not a binary glTF (.glb) file");
  if (view.getUint32(4, true) !== 2) throw new Error("unsupported glTF version");
  const length = Math.min(view.getUint32(8, true), buffer.byteLength);
  let offset = 12, doc = null, bin = null;
  while (offset + 8 <= length) {
    const size = view.getUint32(offset, true);
    const kind = view.getUint32(offset + 4, true);
    const body = new Uint8Array(buffer, offset + 8, size);
    if (kind === CHUNK_JSON && !doc) doc = JSON.parse(new TextDecoder().decode(body));
    else if (kind === CHUNK_BIN && !bin) bin = body.slice();
    offset += 8 + size;
  }
  if (!doc) throw new Error("GLB has no JSON chunk");
  return { doc, bin };
}

export function writeGlb(doc, bin) {
  let json = new TextEncoder().encode(JSON.stringify(doc));
  const jsonPad = (4 - (json.length % 4)) % 4;
  const binLen = bin ? bin.length + ((4 - (bin.length % 4)) % 4) : 0;
  const total = 12 + 8 + json.length + jsonPad + (bin ? 8 + binLen : 0);
  const out = new Uint8Array(total);
  const view = new DataView(out.buffer);
  view.setUint32(0, GLB_MAGIC, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, total, true);
  view.setUint32(12, json.length + jsonPad, true);
  view.setUint32(16, CHUNK_JSON, true);
  out.set(json, 20);
  out.fill(0x20, 20 + json.length, 20 + json.length + jsonPad);
  if (bin) {
    const o = 20 + json.length + jsonPad;
    view.setUint32(o, binLen, true);
    view.setUint32(o + 4, CHUNK_BIN, true);
    out.set(bin, o + 8);
  }
  return out.buffer;
}

export function emptyGlb() {
  return writeGlb({ asset: { version: "2.0", generator: "Pointille" }, scenes: [{ nodes: [] }], scene: 0 }, null);
}

// positions: flat xyz in the scene's world space (glTF, Y-up); colors: flat linear rgb.
export function appendDrones(glb, positions, colorsLinear, { name = "Pointille_Drones", scale = 1, pivot = [0, 0, 0], extras = null } = {}) {
  const { doc, bin } = readGlb(glb);
  const n = positions.length / 3;
  const pos = Float32Array.from(positions);
  const col = Float32Array.from(colorsLinear, (c) => Math.min(Math.max(c, 0), 1));
  for (const key of ["buffers", "bufferViews", "accessors", "meshes", "materials", "nodes", "scenes"]) doc[key] ??= [];

  const payload = new Uint8Array(pos.byteLength + col.byteLength);
  payload.set(new Uint8Array(pos.buffer), 0);
  payload.set(new Uint8Array(col.buffer), pos.byteLength);
  const buffers = doc.buffers;
  let newBin = bin, base, bufIndex;
  const embedded = bin && buffers.length && !("uri" in buffers[0]);
  if (embedded || (!bin && !buffers.length)) {
    const old = bin || new Uint8Array(0);
    if (!bin) buffers.push({ byteLength: 0 });
    base = old.length + ((4 - (old.length % 4)) % 4);
    newBin = new Uint8Array(base + payload.length);
    newBin.set(old, 0);
    newBin.set(payload, base);
    buffers[0].byteLength = newBin.length;
    bufIndex = 0;
  } else {
    base = 0;
    let b64 = "";
    for (let i = 0; i < payload.length; i += 0x8000) b64 += String.fromCharCode(...payload.subarray(i, i + 0x8000));
    buffers.push({ byteLength: payload.length, uri: "data:application/octet-stream;base64," + btoa(b64) });
    bufIndex = buffers.length - 1;
  }

  const views = doc.bufferViews;
  views.push({ buffer: bufIndex, byteOffset: base, byteLength: pos.byteLength, target: 34962 });
  views.push({ buffer: bufIndex, byteOffset: base + pos.byteLength, byteLength: col.byteLength, target: 34962 });
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++)
    for (let a = 0; a < 3; a++) {
      min[a] = Math.min(min[a], pos[3 * i + a]);
      max[a] = Math.max(max[a], pos[3 * i + a]);
    }
  const acc = doc.accessors;
  acc.push({ bufferView: views.length - 2, componentType: 5126, count: n, type: "VEC3", min: n ? min : [0, 0, 0], max: n ? max : [0, 0, 0] });
  acc.push({ bufferView: views.length - 1, componentType: 5126, count: n, type: "VEC3" });
  doc.materials.push({
    name,
    pbrMetallicRoughness: { baseColorFactor: [1, 1, 1, 1], metallicFactor: 0, roughnessFactor: 1 },
    extensions: { KHR_materials_unlit: {} },
  });
  doc.extensionsUsed ??= [];
  if (!doc.extensionsUsed.includes("KHR_materials_unlit")) doc.extensionsUsed.push("KHR_materials_unlit");
  doc.meshes.push({
    name,
    primitives: [{ attributes: { POSITION: acc.length - 2, COLOR_0: acc.length - 1 }, mode: 0, material: doc.materials.length - 1 }],
  });
  const node = { name, mesh: doc.meshes.length - 1 };
  if (extras) node.extras = { pointille: extras };
  doc.nodes.push(node);
  const droneIndex = doc.nodes.length - 1;

  if (!doc.scenes.length) {
    const children = new Set(doc.nodes.flatMap((nd) => nd.children || []));
    doc.scenes.push({ nodes: doc.nodes.slice(0, -1).map((_, i) => i).filter((i) => !children.has(i)) });
    doc.scene = 0;
  }
  const scene = doc.scenes[doc.scene ?? 0];
  scene.nodes ??= [];
  if (scale !== 1) {
    doc.nodes.push({
      name: "Pointille_Root",
      translation: pivot.map((p) => p * (1 - scale)),
      scale: [scale, scale, scale],
      children: [...scene.nodes, droneIndex],
    });
    scene.nodes = [doc.nodes.length - 1];
  } else {
    scene.nodes.push(droneIndex);
  }
  return writeGlb(doc, newBin);
}
