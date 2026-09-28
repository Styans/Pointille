// Entry point: distribute(mesh, options) -> result. Mirrors pointille/core/api.py.
//
// mesh = {
//   triangles: Float64Array(T*9)   world-space corners
//   uvs?: Float64Array(T*6)        v-up (OpenGL)
//   materialIds?: Int32Array(T)
//   vertexColors?: Float64Array(T*9)  linear
//   materials?: [{baseColor: [r,g,b,a] linear, texture?: {width, height, data, srgb}}]
// }
import { averageByOwner, surfaceColors, toSrgb8 } from "./color.js";
import { OrthoOccluder, facingMask } from "./facade.js";
import { SurfaceProjector, relax } from "./relax.js";
import { mulberry32 } from "./rng.js";
import { eliminate, poissonRadius, surfaceCandidates, triangleAreasNormals } from "./sampling.js";
import { nearest } from "./spatial.js";
import { scaleFactor, scalePivot, spacing } from "./stats.js";

export const DEFAULTS = {
  count: 500,
  seed: 0,
  oversample: 8,
  relaxIterations: 20,
  facade: false,
  viewDir: [0, -1, 0], // model -> audience
  facadeAngle: 80,
  occlusion: true,
  colorMode: "area", // "area" | "point" | "none"
  scaleMode: "none", // "none" | "min_distance" | "height"
  targetMinDistance: 1.5,
  targetHeight: 20,
  upAxis: 2,
};

export function distribute(mesh, options = {}) {
  const opt = { ...DEFAULTS, ...options };
  const n = Math.floor(opt.count);
  if (!(n >= 1)) throw new Error("count must be at least 1");
  const rand = mulberry32(opt.seed);
  const tris = mesh.triangles;
  const { areas, normals } = triangleAreasNormals(tris);
  const weights = Float64Array.from(areas);
  if (opt.facade) {
    const mask = facingMask(normals, opt.viewDir, opt.facadeAngle);
    for (let i = 0; i < weights.length; i++) weights[i] *= mask[i];
  }
  let totalArea = 0;
  for (const w of weights) totalArea += w;
  if (!(totalArea > 0)) {
    throw new Error(opt.facade ? "no part of the surface faces the audience" : "the model has no surface area");
  }

  const wanted = Math.max(Math.ceil(Math.max(opt.oversample, 2) * n), n + 1);
  const occluder = opt.facade && opt.occlusion ? new OrthoOccluder(tris, opt.viewDir) : null;
  const cand = gatherCandidates(tris, weights, wanted, rand, occluder);
  const m = cand.tri.length;
  if (m < n) {
    throw new Error(`the visible surface only fits about ${m} drones; lower the count or disable facade mode`);
  }
  const area = totalArea * cand.visible;

  const sel = eliminate(cand.pos, n, area);
  const radius = poissonRadius(area, n);
  const ideal = 2 * radius;
  let points = new Float64Array(n * 3), triIds = new Int32Array(n), bary = new Float64Array(n * 3);
  sel.forEach((c, i) => {
    triIds[i] = cand.tri[c];
    for (let a = 0; a < 3; a++) {
      points[3 * i + a] = cand.pos[3 * c + a];
      bary[3 * i + a] = cand.bary[3 * c + a];
    }
  });
  if (opt.relaxIterations > 0 && n > 1) {
    const allowed = [];
    for (let i = 0; i < weights.length; i++) if (weights[i] > 0) allowed.push(i);
    const sub = new Float64Array(allowed.length * 9);
    allowed.forEach((t, k) => sub.set(tris.subarray(9 * t, 9 * t + 9), 9 * k));
    const projector = new SurfaceProjector(sub, Int32Array.from(allowed), 0.5 * ideal, 0.15 * ideal);
    const accept = occluder ? (x, y, z, t) => !occluder.occluded(x, y, z, t) : null;
    ({ points, triIds, bary } = relax(points, triIds, bary, projector, ideal, opt.relaxIterations, 0.2, accept));
  }

  let colors;
  if (opt.colorMode === "area") {
    const candColors = surfaceColors(mesh, cand.tri, cand.bary);
    const { index } = nearest(cand.pos, points, radius);
    colors = averageByOwner(candColors, index, n);
  } else if (opt.colorMode === "point") {
    colors = surfaceColors(mesh, triIds, bary);
  } else {
    colors = new Float64Array(n * 3).fill(1);
  }

  const { min, mean } = spacing(points, radius);
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++)
    for (let a = 0; a < 3; a++) {
      lo[a] = Math.min(lo[a], points[3 * i + a]);
      hi[a] = Math.max(hi[a], points[3 * i + a]);
    }
  const size = [0, 1, 2].map((a) => hi[a] - lo[a]);
  const scale = scaleFactor(opt.scaleMode, min, size, opt.upAxis, opt.targetMinDistance, opt.targetHeight);
  const pnormals = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) pnormals[3 * i + a] = normals[3 * triIds[i] + a];
  return {
    count: n,
    positions: points,
    normals: pnormals,
    triangleIds: triIds,
    barycentrics: bary,
    colorsLinear: colors,
    minDistance: min,
    meanDistance: mean,
    area,
    size,
    scale,
    pivot: scalePivot(lo, hi, opt.upAxis),
    options: opt,
  };
}

export function scaledPositions(result) {
  const { positions: p, pivot, scale } = result;
  const out = new Float64Array(p.length);
  for (let i = 0; i < p.length; i++) out[i] = pivot[i % 3] + (p[i] - pivot[i % 3]) * scale;
  return out;
}

export function colorsSrgb8(result) {
  return toSrgb8(result.colorsLinear);
}

export function summary(result) {
  return {
    count: result.count,
    scale: result.scale,
    minDistance: result.minDistance * result.scale,
    meanDistance: result.meanDistance * result.scale,
    size: result.size.map((v) => v * result.scale),
  };
}

function gatherCandidates(tris, weights, wanted, rand, occluder) {
  if (!occluder) return { ...surfaceCandidates(tris, weights, wanted, rand), visible: 1 };
  const pos = [], tri = [], bary = [];
  let generated = 0, have = 0, batch = wanted;
  for (let round = 0; round < 6; round++) {
    const c = surfaceCandidates(tris, weights, batch, rand);
    generated += batch;
    for (let k = 0; k < batch; k++) {
      if (occluder.occluded(c.pos[3 * k], c.pos[3 * k + 1], c.pos[3 * k + 2], c.tri[k])) continue;
      have++;
      if (tri.length >= wanted) continue;
      tri.push(c.tri[k]);
      for (let a = 0; a < 3; a++) {
        pos.push(c.pos[3 * k + a]);
        bary.push(c.bary[3 * k + a]);
      }
    }
    if (have >= wanted) break;
    batch = Math.ceil(((wanted - have) / Math.max(have / generated, 0.02)) * 1.1);
  }
  return { pos: Float64Array.from(pos), tri: Int32Array.from(tri), bary: Float64Array.from(bary), visible: have / generated };
}
