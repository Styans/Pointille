// Entry point: distribute(mesh, options) -> result. Mirrors pointille/core/api.py.
//
// mesh = {
//   triangles: Float64Array(T*9)   world-space corners
//   uvs?: Float64Array(T*6)        v-up (OpenGL)
//   materialIds?: Int32Array(T)
//   vertexColors?: Float64Array(T*9)  linear
//   materials?: [{baseColor: [r,g,b,a] linear, texture?: {width, height, data, srgb}}]
// }
import { averageByOwner, glow, surfaceColors, toSrgb8 } from "./color.js";
import { Audience } from "./facade.js";
import { SurfaceProjector, relax } from "./relax.js";
import { AMOUNTS, findLines, place, split } from "./outline.js";
import { mulberry32 } from "./rng.js";
import { eliminate, poissonRadius, spacingScale, surfaceCandidates, triangleAreasNormals } from "./sampling.js";
import { nearest } from "./spatial.js";
import { scaleFactor, scalePivot, spacing } from "./stats.js";

export const DEFAULTS = {
  count: 500,
  seed: 0,
  oversample: 8,
  relaxIterations: 20,
  facade: false,
  viewDir: [0, -1, 0], // model -> audience
  viewPoint: null, // a camera instead of a far-away audience: sight lines converge on it
  facadeAngle: 80,
  audienceSpread: 0, // degrees the audience spans around the vertical axis, seen from the model
  hiddenDensity: 0, // drone density where nobody in the audience sees the model, relative to the facade
  occlusion: true,
  colorMode: "area", // "area" | "point" | "none"
  minBrightness: 0, // LEDs cannot be black: the darkest drone glows at this sRGB brightness 0..1 (0: as the model)
  // outline: some drones right on the lines that draw the model, closer together than the fill
  outline: false,
  outlineSilhouette: true, // the contour the middle viewer sees (viewDir / viewPoint)
  outlineCreases: true, // edges sharper than creaseAngle
  outlineBorders: true, // edges of open surfaces: text, logos, planes
  outlineRelief: true, // what a line drawing shows inside the silhouette: eyes, mouth, folds
  outlineColors: true, // borders between colours of the texture
  outlineAmount: "density", // "density": lines outlineDensity times denser than the fill; "share": that share of the drones
  outlineDensity: 1.5,
  outlineShare: 0.3,
  creaseAngle: 40, // degrees between two faces that make a sharp edge
  reliefAngle: 40, // degrees the surface must bend within a short stretch to draw a relief line
  colorContrast: 0.15, // OKLab difference that makes a colour border (black-white is 1)
  scaleMode: "none", // "none" | "min_distance" | "height"
  targetMinDistance: 1.5,
  targetHeight: 20,
  upAxis: 2,
};

export function distribute(mesh, options = {}) {
  const opt = { ...DEFAULTS, ...options };
  const n = Math.floor(opt.count);
  if (!(n >= 1)) throw new Error("count must be at least 1");
  if (!AMOUNTS.includes(opt.outlineAmount)) throw new Error(`unknown outline amount ${opt.outlineAmount}`);
  const rand = mulberry32(opt.seed);
  const tris = mesh.triangles;
  const { areas, normals } = triangleAreasNormals(tris);
  const weights = Float64Array.from(areas);
  let audience = null, density = null;
  if (opt.facade) {
    audience = new Audience(tris, normals, {
      viewDir: opt.viewDir,
      viewPoint: opt.viewPoint,
      spread: opt.audienceSpread,
      facadeAngle: opt.facadeAngle,
      occlusion: opt.occlusion,
      upAxis: opt.upAxis,
    });
    const hidden = Math.min(Math.max(+opt.hiddenDensity || 0, 0), 1);
    if (hidden <= 0) {
      const mask = audience.triangleMask();
      for (let i = 0; i < weights.length; i++) weights[i] *= mask[i];
    }
    // full density where the whole audience sees the surface, `hidden` where nobody does
    if (hidden < 1) density = (x, y, z, t) => hidden + (1 - hidden) * audience.visibility(x, y, z, t);
  }
  let totalArea = 0;
  for (const w of weights) totalArea += w;
  if (!(totalArea > 0)) {
    throw new Error(opt.facade ? "no part of the surface faces the audience" : "the model has no surface area");
  }

  const wanted = Math.max(Math.ceil(Math.max(opt.oversample, 2) * n), n + 1);
  const cand = gatherCandidates(tris, weights, wanted, rand, density);
  const m = cand.tri.length;
  if (m < n) {
    throw new Error(`the visible surface only fits about ${m} drones; lower the count or disable facade mode`);
  }
  // densities relative to the densest candidate: there the spacing is `ideal`
  let top = 0, low = Infinity;
  for (const r of cand.rho) (top = Math.max(top, r)), (low = Math.min(low, r));
  const area = (totalArea * cand.meanDensity) / top;
  const stretch = low === top ? null : cand.rho.map((r) => spacingScale(r / top));
  const relative = density && ((x, y, z, t) => density(x, y, z, t) / top);

  // the outline first: its drones stand still while the fill makes room around them
  let lines = null, nLine = 0, lineSpacing = 0, line = null;
  let fill = cand, fillArea = area, fillStretch = stretch;
  if (opt.outline && n > 3) lines = findLines(mesh, areas, normals, weights, opt, 2 * poissonRadius(area, n), rand, audience);
  if (lines) {
    let free;
    ({ count: nLine, spacing: lineSpacing, free, fillArea } = split(lines, cand.pos, area, n, opt));
    if (nLine) {
      line = place(lines, nLine, lineSpacing, opt.relaxIterations, tris);
      nLine = line.triIds.length;
    }
    const keep = [];
    free.forEach((f, k) => f && keep.push(k));
    if (keep.length < n - nLine) {
      // the lines took fewer drones than planned and left no room for the rest
      fillArea = area;
    } else if (keep.length < free.length) {
      fill = {
        pos: Float64Array.from(keep.flatMap((k) => [cand.pos[3 * k], cand.pos[3 * k + 1], cand.pos[3 * k + 2]])),
        tri: Int32Array.from(keep, (k) => cand.tri[k]),
        bary: Float64Array.from(keep.flatMap((k) => [cand.bary[3 * k], cand.bary[3 * k + 1], cand.bary[3 * k + 2]])),
      };
      fillStretch = stretch && Float64Array.from(keep, (k) => stretch[k]);
    }
  }
  const nFill = n - nLine;

  const sel = eliminate(fill.pos, nFill, fillArea, 8, fillStretch);
  const radius = poissonRadius(fillArea, nFill);
  const ideal = 2 * radius;
  let points = new Float64Array(nFill * 3), triIds = new Int32Array(nFill), bary = new Float64Array(nFill * 3);
  sel.forEach((c, i) => {
    triIds[i] = fill.tri[c];
    for (let a = 0; a < 3; a++) {
      points[3 * i + a] = fill.pos[3 * c + a];
      bary[3 * i + a] = fill.bary[3 * c + a];
    }
  });
  if (opt.relaxIterations > 0 && n > 1) {
    const allowed = [];
    for (let i = 0; i < weights.length; i++) if (weights[i] > 0) allowed.push(i);
    const sub = new Float64Array(allowed.length * 9);
    allowed.forEach((t, k) => sub.set(tris.subarray(9 * t, 9 * t + 9), 9 * k));
    const widest = fillStretch ? fillStretch.reduce((a, b) => Math.max(a, b), 1) : 1;
    const projector = new SurfaceProjector(sub, Int32Array.from(allowed), 0.5 * ideal, 0.15 * ideal * widest);
    ({ points, triIds, bary } = relax(points, triIds, bary, projector, ideal, opt.relaxIterations, 0.2, relative, widest,
      nLine ? line.points : null));
  }
  if (nLine) {
    const join = (a, b, T) => {
      const out = new T(a.length + b.length);
      out.set(a);
      out.set(b, a.length);
      return out;
    };
    points = join(line.points, points, Float64Array);
    triIds = join(line.triIds, triIds, Int32Array);
    bary = join(line.bary, bary, Float64Array);
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
  if (nLine && opt.colorMode !== "none") {
    // an outline drone shows the colour right under it, or the side of a colour border it marks
    const under = surfaceColors(mesh, triIds.subarray(0, nLine), bary.subarray(0, 3 * nLine));
    const marked = lines.colorsAt(points.subarray(0, 3 * nLine));
    for (let k = 0; k < 3 * nLine; k++) colors[k] = Number.isNaN(marked[k]) ? under[k] : marked[k];
  }
  colors = glow(colors, opt.minBrightness);

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
    // share of the audience that sees each drone
    visibility: Float64Array.from({ length: n }, (_, i) =>
      audience ? audience.visibility(points[3 * i], points[3 * i + 1], points[3 * i + 2], triIds[i]) : 1,
    ),
    outline: Uint8Array.from({ length: n }, (_, i) => (i < nLine ? 1 : 0)), // the outline drones come first
    outlineCount: nLine,
    outlineSpacing: nLine ? lineSpacing : 0, // distance between neighbouring outline drones, unscaled
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
    outlineCount: result.outlineCount,
    outlineSpacing: result.outlineSpacing * result.scale,
  };
}

// Candidates over the surface, as many per area as density says (uniform
// without it): each is kept with probability density. meanDensity is the
// average over the area the candidates were drawn from.
function gatherCandidates(tris, weights, wanted, rand, density) {
  if (!density) {
    const c = surfaceCandidates(tris, weights, wanted, rand);
    return { ...c, rho: new Float64Array(wanted).fill(1), meanDensity: 1 };
  }
  const pos = [], tri = [], bary = [], rho = [];
  let generated = 0, have = 0, total = 0, batch = wanted;
  for (let round = 0; round < 6; round++) {
    const c = surfaceCandidates(tris, weights, batch, rand);
    generated += batch;
    for (let k = 0; k < batch; k++) {
      const r = density(c.pos[3 * k], c.pos[3 * k + 1], c.pos[3 * k + 2], c.tri[k]);
      total += r;
      if (!(rand() < r)) continue;
      have++;
      if (tri.length >= wanted) continue;
      tri.push(c.tri[k]);
      rho.push(r);
      for (let a = 0; a < 3; a++) {
        pos.push(c.pos[3 * k + a]);
        bary.push(c.bary[3 * k + a]);
      }
    }
    if (have >= wanted) break;
    batch = Math.ceil(((wanted - have) / Math.max(total / generated, 0.02)) * 1.1);
  }
  return {
    pos: Float64Array.from(pos),
    tri: Int32Array.from(tri),
    bary: Float64Array.from(bary),
    rho: Float64Array.from(rho),
    meanDensity: total / generated,
  };
}
