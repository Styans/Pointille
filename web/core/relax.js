// Relaxation: drones push each other apart while staying on the surface,
// snapped back with an exact closest-point projection.
import { Grid } from "./spatial.js";

// Barycentrics of the closest point on triangle (a, b, c) to p.
// Ericson, Real-Time Collision Detection, 5.1.5.
export function closestOnTriangle(p, T, o, out) {
  const ax = T[o], ay = T[o + 1], az = T[o + 2];
  const abx = T[o + 3] - ax, aby = T[o + 4] - ay, abz = T[o + 5] - az;
  const acx = T[o + 6] - ax, acy = T[o + 7] - ay, acz = T[o + 8] - az;
  const apx = p[0] - ax, apy = p[1] - ay, apz = p[2] - az;
  const d1 = abx * apx + aby * apy + abz * apz;
  const d2 = acx * apx + acy * apy + acz * apz;
  if (d1 <= 0 && d2 <= 0) return set(out, 1, 0, 0);
  const bpx = p[0] - T[o + 3], bpy = p[1] - T[o + 4], bpz = p[2] - T[o + 5];
  const d3 = abx * bpx + aby * bpy + abz * bpz;
  const d4 = acx * bpx + acy * bpy + acz * bpz;
  if (d3 >= 0 && d4 <= d3) return set(out, 0, 1, 0);
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    return set(out, 1 - v, v, 0);
  }
  const cpx = p[0] - T[o + 6], cpy = p[1] - T[o + 7], cpz = p[2] - T[o + 8];
  const d5 = abx * cpx + aby * cpy + abz * cpz;
  const d6 = acx * cpx + acy * cpy + acz * cpz;
  if (d6 >= 0 && d5 <= d6) return set(out, 0, 0, 1);
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    return set(out, 1 - w, 0, w);
  }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / (d4 - d3 + (d5 - d6));
    return set(out, 0, 1 - w, w);
  }
  const denom = 1 / (va + vb + vc);
  const v = vb * denom, w = vc * denom;
  return set(out, 1 - v - w, v, w);
}

function set(out, a, b, c) {
  out[0] = a;
  out[1] = b;
  out[2] = c;
  return out;
}

function lattice(k) {
  const out = [];
  for (let i = 0; i < k; i++)
    for (let j = 0; j < k - i; j++) {
      out.push([k - i - j, i, j], [k - i - j - 1, i + 1, j], [k - i - j - 1, i, j + 1]);
      if (i + j <= k - 2) out.push([k - i - j - 1, i + 1, j], [k - i - j - 2, i + 1, j + 1], [k - i - j - 1, i, j + 1]);
    }
  return out.map((b) => b.map((x) => x / k));
}

export class SurfaceProjector {
  // tris: flat triangles; ids: original triangle id of each; long triangles are
  // split (for indexing only) so a query scans a bounded neighbourhood.
  constructor(tris, ids, maxEdge, maxMove) {
    const t = ids.length;
    const lattices = new Map();
    let total = 0;
    const ks = new Int32Array(t);
    for (let i = 0; i < t; i++) {
      const o = 9 * i;
      let e = 0;
      for (let a = 0; a < 3; a++) {
        const b = (a + 1) % 3;
        e = Math.max(e, Math.hypot(tris[o + 3 * a] - tris[o + 3 * b], tris[o + 3 * a + 1] - tris[o + 3 * b + 1], tris[o + 3 * a + 2] - tris[o + 3 * b + 2]));
      }
      ks[i] = Math.min(Math.max(Math.ceil(e / maxEdge), 1), 32);
      total += ks[i] * ks[i];
    }
    this.sub = new Float64Array(total * 9);
    this.parent = new Int32Array(total);
    this.subBary = new Float64Array(total * 9);
    const cent = new Float64Array(total * 3);
    let s = 0, subEdge = 0;
    for (let i = 0; i < t; i++) {
      const k = ks[i];
      if (!lattices.has(k)) lattices.set(k, lattice(k));
      const lat = lattices.get(k);
      const o = 9 * i;
      for (let q = 0; q < lat.length; q += 3, s++) {
        for (let c = 0; c < 3; c++) {
          const b = lat[q + c];
          for (let a = 0; a < 3; a++) {
            const v = b[0] * tris[o + a] + b[1] * tris[o + 3 + a] + b[2] * tris[o + 6 + a];
            this.sub[9 * s + 3 * c + a] = v;
            this.subBary[9 * s + 3 * c + a] = b[a];
            cent[3 * s + a] += v / 3;
          }
        }
        this.parent[s] = ids[i];
        for (let a = 0; a < 3; a++) {
          const b = (a + 1) % 3, S = this.sub, u = 9 * s;
          subEdge = Math.max(subEdge, Math.hypot(S[u + 3 * a] - S[u + 3 * b], S[u + 3 * a + 1] - S[u + 3 * b + 1], S[u + 3 * a + 2] - S[u + 3 * b + 2]));
        }
      }
    }
    this.maxMove = maxMove;
    this.reach = subEdge + maxMove;
    this.grid = new Grid(cent, this.reach);
  }

  // closest surface point to p: {found, pos, tri, bary}
  project(p) {
    let best = Infinity, bestS = -1;
    const b = [0, 0, 0], bestB = [0, 0, 0];
    this.grid.forEachNear(p[0], p[1], p[2], this.reach, (s) => {
      closestOnTriangle(p, this.sub, 9 * s, b);
      const o = 9 * s, S = this.sub;
      let d2 = 0;
      for (let a = 0; a < 3; a++) {
        const q = b[0] * S[o + a] + b[1] * S[o + 3 + a] + b[2] * S[o + 6 + a];
        d2 += (q - p[a]) * (q - p[a]);
      }
      if (d2 < best) {
        best = d2;
        bestS = s;
        bestB[0] = b[0]; bestB[1] = b[1]; bestB[2] = b[2];
      }
    });
    if (bestS < 0) return null;
    const o = 9 * bestS, S = this.sub, B = this.subBary;
    const pos = [0, 1, 2].map((a) => bestB[0] * S[o + a] + bestB[1] * S[o + 3 + a] + bestB[2] * S[o + 6 + a]);
    const bary = [0, 1, 2].map((a) => bestB[0] * B[o + a] + bestB[1] * B[o + 3 + a] + bestB[2] * B[o + 6 + a]);
    return { pos, tri: this.parent[bestS], bary };
  }
}

// Repulsion relaxation; returns the state with the best minimum spacing seen.
// accept(x, y, z, tri) can veto a move (facade mode keeps drones visible).
export function relax(points, triIds, bary, projector, spacing, iterations, step = 0.2, accept = null) {
  let p = Float64Array.from(points), t = Int32Array.from(triIds), b = Float64Array.from(bary);
  const n = p.length / 3;
  if (iterations <= 0 || n < 2) return { points: p, triIds: t, bary: b };
  let bestMin = -1, best = null;
  const disp = new Float64Array(n * 3);
  const maxMove = projector.maxMove;
  for (let it = 0; it <= iterations; it++) {
    const grid = new Grid(p, spacing);
    disp.fill(0);
    let curMin = Infinity;
    for (let i = 0; i < n; i++) {
      const x = p[3 * i], y = p[3 * i + 1], z = p[3 * i + 2];
      grid.forEachNear(x, y, z, spacing, (j, d) => {
        if (j === i) return;
        if (d < curMin) curMin = d;
        // soft 1/d repulsion: strong for close pairs, zero at the ideal spacing
        const mag = spacing / Math.max(d, spacing * 1e-3) - 1;
        const f = mag / Math.max(d, 1e-300);
        disp[3 * i] += (x - p[3 * j]) * f;
        disp[3 * i + 1] += (y - p[3 * j + 1]) * f;
        disp[3 * i + 2] += (z - p[3 * j + 2]) * f;
      });
    }
    if (curMin === Infinity) curMin = spacing;
    if (curMin > bestMin) {
      bestMin = curMin;
      best = { points: p.slice(), triIds: t.slice(), bary: b.slice() };
    }
    if (it === iterations) break;
    for (let i = 0; i < n; i++) {
      let dx = disp[3 * i] * step * spacing, dy = disp[3 * i + 1] * step * spacing, dz = disp[3 * i + 2] * step * spacing;
      const len = Math.hypot(dx, dy, dz);
      if (len > maxMove) {
        const s = maxMove / len;
        dx *= s; dy *= s; dz *= s;
      }
      const hit = projector.project([p[3 * i] + dx, p[3 * i + 1] + dy, p[3 * i + 2] + dz]);
      if (!hit) continue;
      if (accept && !accept(hit.pos[0], hit.pos[1], hit.pos[2], hit.tri)) continue;
      for (let a = 0; a < 3; a++) {
        p[3 * i + a] = hit.pos[a];
        b[3 * i + a] = hit.bary[a];
      }
      t[i] = hit.tri;
    }
  }
  return best;
}
