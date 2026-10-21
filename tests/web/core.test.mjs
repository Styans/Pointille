import assert from "node:assert/strict";
import { test } from "node:test";

import { colorsSrgb8, distribute, scaledPositions, summary } from "../../web/core/api.js";
import { glow, linearToSrgb, sampleTexture, srgbToLinear } from "../../web/core/color.js";
import { closestOnTriangle } from "../../web/core/relax.js";
import { poissonRadius } from "../../web/core/sampling.js";
import { Grid, nearest } from "../../web/core/spatial.js";
import { QUADRANTS, concat, cube, gridPlane, quadrantTexture, uvSphere } from "./helpers.mjs";

const sphere = uvSphere();
const ideal = (r) => 2 * poissonRadius(r.area, r.count);

test("exact count", () => {
  for (const count of [1, 2, 17, 500]) {
    for (const relaxIterations of [0, 10]) {
      const r = distribute(sphere, { count, relaxIterations, colorMode: "none" });
      assert.equal(r.count, count);
      assert.equal(r.positions.length, count * 3);
      assert.equal(r.colorsLinear.length, count * 3);
    }
  }
});

test("points lie on the surface", () => {
  const r = distribute(sphere, { count: 400, colorMode: "none" });
  const T = sphere.triangles;
  for (let i = 0; i < r.count; i++) {
    const t = r.triangleIds[i], b = [0, 1, 2].map((a) => r.barycentrics[3 * i + a]);
    assert.ok(Math.abs(b[0] + b[1] + b[2] - 1) < 1e-9 && Math.min(...b) > -1e-9);
    for (let a = 0; a < 3; a++) {
      const p = b[0] * T[9 * t + a] + b[1] * T[9 * t + 3 + a] + b[2] * T[9 * t + 6 + a];
      assert.ok(Math.abs(p - r.positions[3 * i + a]) < 1e-9);
    }
  }
});

test("uniform spacing", () => {
  const raw = distribute(sphere, { count: 800, relaxIterations: 0, colorMode: "none" });
  const relaxed = distribute(sphere, { count: 800, colorMode: "none" });
  const d = ideal(raw);
  assert.ok(raw.minDistance > 0.65 * d, `raw min ${raw.minDistance / d}`);
  assert.ok(relaxed.minDistance > 0.75 * d, `relaxed min ${relaxed.minDistance / d}`);
  assert.ok(relaxed.meanDistance > 0.87 * d, `relaxed mean ${relaxed.meanDistance / d}`);
});

test("deterministic by seed", () => {
  const a = distribute(sphere, { count: 200, seed: 7 });
  const b = distribute(sphere, { count: 200, seed: 7 });
  const c = distribute(sphere, { count: 200, seed: 8 });
  assert.deepEqual(a.positions, b.positions);
  assert.notDeepEqual(a.positions, c.positions);
});

test("facade keeps the side facing the audience", () => {
  const r = distribute(sphere, { count: 300, facade: true, viewDir: [0, -1, 0], facadeAngle: 60 });
  for (let i = 0; i < r.count; i++) assert.ok(-r.normals[3 * i + 1] > Math.cos(Math.PI / 3) - 1e-9);
  assert.ok(r.area < 0.3 * 4 * Math.PI);
});

test("facade occlusion", () => {
  const mesh = concat([gridPlane({ size: [4, 2], res: 8, y: 0 }), gridPlane({ size: [2, 2], res: 4, y: -1, center: [-1, 0] })]);
  const r = distribute(mesh, { count: 200, facade: true, viewDir: [0, -1, 0] });
  let onBack = 0;
  for (let i = 0; i < r.count; i++) {
    if (Math.abs(r.positions[3 * i + 1]) < 1e-9) {
      onBack++;
      assert.ok(r.positions[3 * i] > -1e-6);
    }
  }
  assert.ok(onBack > 0);
  assert.ok(Math.abs(r.area - 8) < 0.6, `area ${r.area}`);
});

test("an audience spread over 60 degrees sees the sides of a cube", () => {
  const box = cube();
  const narrow = distribute(box, { count: 300, facade: true, viewDir: [0, -1, 0] });
  for (let i = 0; i < narrow.count; i++) assert.ok(narrow.normals[3 * i + 1] < -0.99);
  const wide = distribute(box, { count: 300, facade: true, viewDir: [0, -1, 0], audienceSpread: 60 });
  let side = 0;
  for (let i = 0; i < wide.count; i++) {
    const [nx, ny, nz] = wide.normals.subarray(3 * i, 3 * i + 3);
    assert.ok(ny < 0.5 && Math.abs(nz) < 0.5); // no back, top, bottom
    const isSide = Math.abs(nx) > 0.99;
    side += isSide;
    assert.ok(isSide ? wide.visibility[i] < 1 : wide.visibility[i] === 1);
  }
  assert.ok(side > 60 && side < wide.count - side, `side ${side}`);
  assert.ok(wide.minDistance > 0.75 * ideal(wide), `min ${wide.minDistance / ideal(wide)}`);
});

test("hidden density keeps the back sparse", () => {
  const r = distribute(sphere, { count: 400, facade: true, viewDir: [0, -1, 0], hiddenDensity: 0.25 });
  const { dist } = nearest(r.positions, r.positions, 0.1, true);
  const front = [], back = [];
  for (let i = 0; i < r.count; i++) {
    const y = r.positions[3 * i + 1];
    if (y < -0.2) front.push(i), assert.equal(r.visibility[i], 1);
    if (y > 0.2) back.push(i), assert.equal(r.visibility[i], 0);
  }
  assert.ok(back.length > 40 && front.length > 2.5 * back.length, `${front.length} / ${back.length}`);
  const median = (idx) => idx.map((i) => dist[i]).sort((a, b) => a - b)[idx.length >> 1];
  assert.ok(median(back) > 1.6 * median(front));
  assert.ok(r.minDistance > 0.7 * ideal(r), `min ${r.minDistance / ideal(r)}`);
  const all = distribute(sphere, { count: 300, facade: true, hiddenDensity: 1 });
  assert.ok(Math.abs(all.area / (4 * Math.PI) - 1) < 0.02);
});

test("a camera sees in perspective", () => {
  // a plane 1 m in front of the camera hides twice its width of a plane 2 m away
  const mesh = concat([gridPlane({ size: [6, 2], res: 12, y: 0 }), gridPlane({ size: [1, 2], res: 2, y: -1 })]);
  const r = distribute(mesh, { count: 300, facade: true, viewPoint: [0, -2, 0] });
  let onBack = 0;
  for (let i = 0; i < r.count; i++) {
    assert.equal(r.visibility[i], 1);
    if (Math.abs(r.positions[3 * i + 1]) < 1e-9) {
      onBack++;
      assert.ok(Math.abs(r.positions[3 * i]) > 1 - 1e-6);
    }
  }
  assert.ok(onBack > 100);
  // a far-away audience sees the back plane right next to the front one
  const far = distribute(mesh, { count: 300, facade: true, viewDir: [0, -1, 0] });
  let near = 0;
  for (let i = 0; i < far.count; i++) if (Math.abs(far.positions[3 * i + 1]) < 1e-9 && Math.abs(far.positions[3 * i]) < 0.9) near++;
  assert.ok(near > 0);
});

test("errors", () => {
  assert.throws(() => distribute(sphere, { count: 0 }));
  assert.throws(() => distribute(gridPlane({ facing: 1 }), { count: 10, facade: true, viewDir: [0, -1, 0] }));
});

test("texture orientation", () => {
  const tex = quadrantTexture();
  const cases = [[0.25, 0.75, "top_left"], [0.75, 0.75, "top_right"], [0.25, 0.25, "bottom_left"], [0.75, 0.25, "bottom_right"]];
  for (const [u, v, key] of cases) {
    const c = sampleTexture(tex, u, v).map((x) => Math.round(linearToSrgb(x) * 255));
    assert.deepEqual(c, QUADRANTS[key]);
  }
});

for (const mode of ["point", "area"]) {
  test(`drone colors follow the texture (${mode})`, () => {
    const mesh = { ...gridPlane({ size: [2, 2], res: 8 }), materials: [{ baseColor: [1, 1, 1, 1], texture: quadrantTexture() }] };
    const r = distribute(mesh, { count: 150, colorMode: mode });
    const c8 = colorsSrgb8(r);
    const margin = mode === "area" ? 0.25 : 0.05;
    let checked = 0;
    for (let i = 0; i < r.count; i++) {
      const x = r.positions[3 * i], z = r.positions[3 * i + 2];
      if (Math.min(Math.abs(x), Math.abs(z)) < margin) continue;
      const key = (z > 0 ? "top" : "bottom") + "_" + (x < 0 ? "left" : "right");
      assert.deepEqual(Array.from(c8.subarray(3 * i, 3 * i + 3)), QUADRANTS[key]);
      checked++;
    }
    assert.ok(checked > 50);
  });
}

test("base color and vertex colors multiply", () => {
  const plane = gridPlane({ res: 2 });
  const t = plane.triangles.length / 9;
  const vertexColors = new Float64Array(t * 9);
  for (let i = 0; i < t * 3; i++) vertexColors.set([1, 0.25, 1], 3 * i);
  const r = distribute({ ...plane, vertexColors, materials: [{ baseColor: [0.5, 1, 1, 1] }] }, { count: 20, colorMode: "point" });
  for (let i = 0; i < r.count; i++) {
    assert.ok(Math.abs(r.colorsLinear[3 * i] - 0.5) < 1e-12 && Math.abs(r.colorsLinear[3 * i + 1] - 0.25) < 1e-12);
  }
});

test("srgb round trip", () => {
  for (let i = 0; i <= 100; i++) assert.ok(Math.abs(linearToSrgb(srgbToLinear(i / 100)) - i / 100) < 1e-12);
});

// black, dark brown, brown, gold, white, near-black noise (sRGB 0..255)
const GLOW_CASES = [[0, 0, 0], [60, 35, 18], [120, 70, 35], [240, 200, 60], [255, 255, 255], [8, 3, 6]];
const toLinear = (rows) => Float64Array.from(rows.flat(), (c) => srgbToLinear(c / 255));
const glow8 = (rows, floor) => {
  const out = glow(toLinear(rows), floor);
  return rows.map((_, i) => [0, 1, 2].map((a) => Math.round(linearToSrgb(out[3 * i + a]) * 255)));
};

test("drones never go black", () => {
  const [black, darkBrown, brown, gold, white, noise] = glow8(GLOW_CASES, 0.4);
  assert.deepEqual(black, [102, 102, 102]); // a dim white, not off
  assert.deepEqual(noise, black); // near-black has no hue left to keep, only noise
  assert.ok(darkBrown[0] > darkBrown[1] && darkBrown[1] > darkBrown[2]); // still brown, only brighter
  const v = (c) => Math.max(...c);
  assert.ok(v(black) < v(darkBrown) && v(darkBrown) < v(brown) && v(brown) < v(gold) && v(gold) <= v(white));
  assert.deepEqual(white, [255, 255, 255]);
});

test("glow: off, a dark model, a black model", () => {
  const lin = toLinear(GLOW_CASES);
  assert.equal(glow(lin, 0), lin); // 0: the model's colours as they are
  const [dim, bright] = glow8([[40, 10, 10], [90, 20, 20]], 0.4);
  assert.ok(bright[0] === 255 && dim[0] < 255); // a dark model is stretched up to full brightness
  let seed = 1;
  const rand = () => ((seed = (seed * 16807) % 2147483647) % 20) + 2;
  const tops = glow8(Array.from({ length: 50 }, () => [rand(), rand(), rand()]), 0.4).map((c) => Math.max(...c));
  assert.ok(Math.max(...tops) - Math.min(...tops) <= 10); // glows evenly (within 4 %): its noise does not make drones flicker
});

test("distribute lights a dark model", () => {
  const brown = [0.25, 0.15, 0.08].map(srgbToLinear);
  const mesh = { ...gridPlane({ res: 2 }), materials: [{ baseColor: [...brown, 1] }] };
  const lit = distribute(mesh, { count: 20, colorMode: "point", minBrightness: 0.4 });
  for (let i = 0; i < lit.count; i++) {
    const [r, g, b] = [0, 1, 2].map((a) => linearToSrgb(lit.colorsLinear[3 * i + a]));
    assert.ok(Math.max(r, g, b) > 0.4 && r > g && g > b);
  }
  const plain = distribute(mesh, { count: 20, colorMode: "point" });
  for (let i = 0; i < plain.count; i++) assert.ok(Math.abs(plain.colorsLinear[3 * i] - brown[0]) < 1e-12);
});

test("scale to min distance and height", () => {
  const r = distribute(sphere, { count: 300, scaleMode: "min_distance", targetMinDistance: 2 });
  assert.ok(Math.abs(summary(r).minDistance - 2) < 1e-9);
  const p = scaledPositions(r);
  const { dist } = nearest(p, p, 1, true);
  assert.ok(Math.abs(Math.min(...dist) - 2) < 1e-9);

  const h = distribute(sphere, { count: 300, scaleMode: "height", targetHeight: 30, upAxis: 2 });
  const q = scaledPositions(h);
  const zs = Array.from({ length: h.count }, (_, i) => q[3 * i + 2]);
  assert.ok(Math.abs(Math.max(...zs) - Math.min(...zs) - 30) < 1e-9);
});

test("grid and nearest match brute force", () => {
  let s = 1;
  const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  const pts = Float64Array.from({ length: 1500 }, rnd);
  const q = Float64Array.from({ length: 150 }, rnd);
  const grid = new Grid(pts, 0.2);
  for (let i = 0; i < 50; i++) {
    const got = new Set();
    grid.forEachNear(q[3 * i], q[3 * i + 1], q[3 * i + 2], 0.2, (j) => got.add(j));
    const want = new Set();
    let best = Infinity, bestJ = -1;
    for (let j = 0; j < 500; j++) {
      const d = Math.hypot(pts[3 * j] - q[3 * i], pts[3 * j + 1] - q[3 * i + 1], pts[3 * j + 2] - q[3 * i + 2]);
      if (d < 0.2) want.add(j);
      if (d < best) (best = d), (bestJ = j);
    }
    assert.deepEqual(got, want);
    assert.equal(nearest(q.subarray(3 * i, 3 * i + 3), pts, 0.01).index[0], bestJ);
  }
});

test("closest point on triangle", () => {
  const T = Float64Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const b = [0, 0, 0];
  assert.deepEqual(closestOnTriangle([0.2, 0.2, 5], T, 0, b).map((x) => +x.toFixed(9)), [0.6, 0.2, 0.2]);
  assert.deepEqual(closestOnTriangle([-1, -1, 0], T, 0, b), [1, 0, 0]);
  assert.deepEqual(closestOnTriangle([2, 2, 0], T, 0, b).map((x) => +x.toFixed(9)), [0, 0.5, 0.5]);
});
