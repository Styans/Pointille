import assert from "node:assert/strict";
import { test } from "node:test";

import { csvText, plyBytes, toZUp } from "../../web/io/export.js";
import { appendDrones, emptyGlb, readGlb, writeGlb } from "../../web/io/gltf-append.js";

const pos = Float64Array.from([0, 1, 2, 3, 4, 5]);
const col = Float64Array.from([1, 0, 0, 0, 0.5, 1]);

test("points-only GLB", () => {
  const { doc, bin } = readGlb(appendDrones(emptyGlb(), pos, col, { extras: { count: 2 } }));
  assert.equal(doc.buffers[0].byteLength, bin.length);
  assert.equal(bin.length, 2 * 24);
  assert.deepEqual(doc.scenes[0].nodes, [0]);
  assert.equal(doc.nodes[0].extras.pointille.count, 2);
  const prim = doc.meshes[0].primitives[0];
  assert.equal(prim.mode, 0);
  assert.deepEqual(Object.keys(prim.attributes).sort(), ["COLOR_0", "POSITION"]);
  assert.deepEqual(Array.from(new Float32Array(bin.buffer, bin.byteOffset, 6)), [0, 1, 2, 3, 4, 5]);
});

test("append keeps the original model", () => {
  const originalBin = Uint8Array.from({ length: 30 }, (_, i) => i + 1);
  const original = {
    asset: { version: "2.0" },
    buffers: [{ byteLength: 30 }],
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: 30 }],
    images: [{ bufferView: 0, mimeType: "image/png" }],
    nodes: [{ name: "model", children: [1] }, { name: "part" }],
    scenes: [{ nodes: [0] }],
    scene: 0,
  };
  const glb = writeGlb(structuredClone(original), originalBin);
  const { doc, bin } = readGlb(appendDrones(glb, pos, col));
  assert.deepEqual(bin.subarray(0, 30), originalBin);
  assert.equal(doc.bufferViews[1].byteOffset, 32); // 4-byte aligned after the original data
  assert.deepEqual(doc.images, original.images);
  assert.deepEqual(doc.nodes.slice(0, 2), original.nodes);
  assert.deepEqual(doc.scenes[0].nodes, [0, 2]);

  const scaled = readGlb(appendDrones(glb, pos, col, { scale: 10, pivot: [0, -1, 0] })).doc;
  const root = scaled.nodes.at(-1);
  assert.equal(root.name, "Pointille_Root");
  assert.deepEqual(root.scale, [10, 10, 10]);
  assert.deepEqual(root.translation.map((v) => v + 0), [0, 9, 0]);
  assert.deepEqual(root.children, [0, 2]);
  assert.deepEqual(scaled.scenes[0].nodes, [3]);
});

test("csv, ply and axes", () => {
  const c8 = Uint8Array.from([255, 0, 16, 1, 2, 3]);
  const lines = csvText(pos, c8).trim().split("\n");
  assert.equal(lines[0], "id,x,y,z,r,g,b,hex");
  assert.equal(lines[1], "0,0.0000,1.0000,2.0000,255,0,16,#FF0010");
  const ply = plyBytes(pos, c8);
  const text = new TextDecoder().decode(ply.subarray(0, 200));
  assert.ok(text.includes("element vertex 2"));
  assert.equal(ply.length, text.indexOf("end_header\n") + 11 + 30);
  assert.deepEqual(Array.from(toZUp([0, 0, 1, 0, 1, 0]), (v) => v + 0), [0, -1, 0, 0, 0, 1]);
});
