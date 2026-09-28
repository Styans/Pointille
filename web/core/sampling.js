// Blue-noise selection of exactly N points on a surface:
// area-uniform candidates, then Weighted Sample Elimination (Yuksel 2015).
import { Grid } from "./spatial.js";

export function triangleAreasNormals(tris) {
  const t = tris.length / 9;
  const areas = new Float64Array(t);
  const normals = new Float64Array(t * 3);
  for (let i = 0; i < t; i++) {
    const o = 9 * i;
    const ux = tris[o + 3] - tris[o], uy = tris[o + 4] - tris[o + 1], uz = tris[o + 5] - tris[o + 2];
    const vx = tris[o + 6] - tris[o], vy = tris[o + 7] - tris[o + 1], vz = tris[o + 8] - tris[o + 2];
    const cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
    const len = Math.hypot(cx, cy, cz);
    areas[i] = 0.5 * len;
    if (len > 0) {
      normals[3 * i] = cx / len;
      normals[3 * i + 1] = cy / len;
      normals[3 * i + 2] = cz / len;
    }
  }
  return { areas, normals };
}

// count points uniform over the surface, triangles picked proportional to weights
export function surfaceCandidates(tris, weights, count, rand) {
  const t = weights.length;
  const cdf = new Float64Array(t);
  let acc = 0;
  for (let i = 0; i < t; i++) cdf[i] = acc += weights[i];
  if (!(acc > 0)) throw new Error("surface has zero area");
  const pos = new Float64Array(count * 3);
  const tri = new Int32Array(count);
  const bary = new Float64Array(count * 3);
  for (let k = 0; k < count; k++) {
    const target = rand() * acc;
    let lo = 0, hi = t - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cdf[mid] > target) hi = mid;
      else lo = mid + 1;
    }
    const r1 = Math.sqrt(rand()), r2 = rand();
    const b0 = 1 - r1, b1 = r1 * (1 - r2), b2 = r1 * r2;
    const o = 9 * lo;
    tri[k] = lo;
    bary[3 * k] = b0;
    bary[3 * k + 1] = b1;
    bary[3 * k + 2] = b2;
    for (let a = 0; a < 3; a++) pos[3 * k + a] = b0 * tris[o + a] + b1 * tris[o + 3 + a] + b2 * tris[o + 6 + a];
  }
  return { pos, tri, bary };
}

// max possible spacing radius of count points on area (hexagonal packing)
export function poissonRadius(area, count) {
  return Math.sqrt(area / (2 * Math.sqrt(3) * Math.max(count, 1)));
}

// Weighted Sample Elimination: indices of the count kept points.
export function eliminate(points, count, area, alpha = 8) {
  const m = points.length / 3;
  if (count >= m) return Int32Array.from({ length: m }, (_, i) => i);
  const dMax = 2 * poissonRadius(area, count);
  const dMin = dMax * (1 - Math.pow(count / m, 1.5)) * 0.65; // weight limiting
  const grid = new Grid(points, dMax);
  const offsets = new Int32Array(m + 1);
  const nbr = [];
  const wts = [];
  const w = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    grid.forEachNear(points[3 * i], points[3 * i + 1], points[3 * i + 2], dMax, (j, d) => {
      if (j === i) return;
      const wij = Math.pow(1 - Math.max(d, dMin) / dMax, alpha);
      nbr.push(j);
      wts.push(wij);
      w[i] += wij;
    });
    offsets[i + 1] = nbr.length;
  }

  // indexed max-heap; weights only ever decrease, so updates sift down
  const heap = new Int32Array(m);
  const pos = new Int32Array(m);
  let size = m;
  for (let i = 0; i < m; i++) heap[i] = pos[i] = i;
  const above = (a, b) => w[a] > w[b] || (w[a] === w[b] && a < b);
  const siftDown = (k) => {
    const item = heap[k];
    for (;;) {
      let c = 2 * k + 1;
      if (c >= size) break;
      if (c + 1 < size && above(heap[c + 1], heap[c])) c++;
      if (!above(heap[c], item)) break;
      heap[k] = heap[c];
      pos[heap[k]] = k;
      k = c;
    }
    heap[k] = item;
    pos[item] = k;
  };
  for (let k = (size >> 1) - 1; k >= 0; k--) siftDown(k);

  const removed = new Uint8Array(m);
  while (size > count) {
    const i = heap[0];
    heap[0] = heap[--size];
    pos[heap[0]] = 0;
    siftDown(0);
    removed[i] = 1;
    for (let k = offsets[i]; k < offsets[i + 1]; k++) {
      const j = nbr[k];
      if (removed[j]) continue;
      w[j] -= wts[k];
      siftDown(pos[j]);
    }
  }
  const out = new Int32Array(count);
  for (let i = 0, c = 0; i < m; i++) if (!removed[i]) out[c++] = i;
  return out;
}
