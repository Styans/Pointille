// Facade mode: keep only the surface the audience can see. Sight lines are
// treated as parallel to viewDir (model -> audience), so occlusion uses a
// 2D grid of projected triangles.

const MAX_PAIRS = 8_000_000;

export function normalize(v) {
  const n = Math.hypot(v[0], v[1], v[2]);
  if (!n) throw new Error("view direction must be non-zero");
  return [v[0] / n, v[1] / n, v[2] / n];
}

export function viewBasis(viewDir) {
  const d = normalize(viewDir);
  const h = Math.abs(d[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  const e1 = normalize([h[1] * d[2] - h[2] * d[1], h[2] * d[0] - h[0] * d[2], h[0] * d[1] - h[1] * d[0]]);
  const e2 = [d[1] * e1[2] - d[2] * e1[1], d[2] * e1[0] - d[0] * e1[2], d[0] * e1[1] - d[1] * e1[0]];
  return { d, e1, e2 };
}

export function facingMask(normals, viewDir, maxAngleDeg) {
  const d = normalize(viewDir);
  const c = Math.cos((maxAngleDeg * Math.PI) / 180);
  const t = normals.length / 3;
  const out = new Uint8Array(t);
  for (let i = 0; i < t; i++) out[i] = normals[3 * i] * d[0] + normals[3 * i + 1] * d[1] + normals[3 * i + 2] * d[2] > c ? 1 : 0;
  return out;
}

const dot3 = (a, i, b) => a[i] * b[0] + a[i + 1] * b[1] + a[i + 2] * b[2];

export class OrthoOccluder {
  constructor(tris, viewDir) {
    const { d, e1, e2 } = viewBasis(viewDir);
    this.d = d;
    this.e1 = e1;
    this.e2 = e2;
    this.tris = tris;
    const t = tris.length / 9;
    const px = new Float64Array(t * 3), py = new Float64Array(t * 3);
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let k = 0; k < t * 3; k++) {
      px[k] = dot3(tris, 3 * k, e1);
      py[k] = dot3(tris, 3 * k, e2);
      minX = Math.min(minX, px[k]); maxX = Math.max(maxX, px[k]);
      minY = Math.min(minY, py[k]); maxY = Math.max(maxY, py[k]);
      for (let a = 0; a < 3; a++) {
        lo[a] = Math.min(lo[a], tris[3 * k + a]);
        hi[a] = Math.max(hi[a], tris[3 * k + a]);
      }
    }
    this.eps = Math.max(Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]), 1e-12) * 1e-6;
    this.ox = minX;
    this.oy = minY;
    const extent = Math.max(maxX - minX, maxY - minY, 1e-12);
    let res = Math.min(Math.max(Math.floor(Math.sqrt(t) * 1.5), 8), 2048);
    let spans;
    const x0 = new Int32Array(t), x1 = new Int32Array(t), y0 = new Int32Array(t), y1 = new Int32Array(t);
    for (;;) {
      this.cell = extent / res;
      spans = 0;
      for (let i = 0; i < t; i++) {
        const a = 3 * i;
        x0[i] = this.cellX(Math.min(px[a], px[a + 1], px[a + 2]));
        x1[i] = this.cellX(Math.max(px[a], px[a + 1], px[a + 2]));
        y0[i] = this.cellY(Math.min(py[a], py[a + 1], py[a + 2]));
        y1[i] = this.cellY(Math.max(py[a], py[a + 1], py[a + 2]));
        spans += (x1[i] - x0[i] + 1) * (y1[i] - y0[i] + 1);
      }
      if (spans <= MAX_PAIRS || res <= 8) break;
      res >>= 1;
    }
    this.res = res + 1;
    const cells = this.res * this.res;
    const counts = new Int32Array(cells + 1);
    for (let i = 0; i < t; i++)
      for (let y = y0[i]; y <= y1[i]; y++) for (let x = x0[i]; x <= x1[i]; x++) counts[x + this.res * y + 1]++;
    for (let c = 0; c < cells; c++) counts[c + 1] += counts[c];
    this.offsets = counts;
    this.list = new Int32Array(spans);
    const fill = counts.slice(0, cells);
    for (let i = 0; i < t; i++)
      for (let y = y0[i]; y <= y1[i]; y++) for (let x = x0[i]; x <= x1[i]; x++) this.list[fill[x + this.res * y]++] = i;
  }

  cellX(v) {
    return Math.max(0, Math.floor((v - this.ox) / this.cell));
  }

  cellY(v) {
    return Math.max(0, Math.floor((v - this.oy) / this.cell));
  }

  // true when the straight line from (x, y, z) towards the audience hits the model
  occluded(x, y, z, ownTri = -1) {
    const cx = Math.min(this.cellX(x * this.e1[0] + y * this.e1[1] + z * this.e1[2]), this.res - 1);
    const cy = Math.min(this.cellY(x * this.e2[0] + y * this.e2[1] + z * this.e2[2]), this.res - 1);
    const c = cx + this.res * cy;
    const d = this.d, T = this.tris, eps = this.eps;
    const ox = x + d[0] * eps, oy = y + d[1] * eps, oz = z + d[2] * eps;
    for (let k = this.offsets[c]; k < this.offsets[c + 1]; k++) {
      const i = this.list[k];
      if (i === ownTri) continue;
      const o = 9 * i;
      const ax = T[o], ay = T[o + 1], az = T[o + 2];
      const e1x = T[o + 3] - ax, e1y = T[o + 4] - ay, e1z = T[o + 5] - az;
      const e2x = T[o + 6] - ax, e2y = T[o + 7] - ay, e2z = T[o + 8] - az;
      const px = d[1] * e2z - d[2] * e2y, py = d[2] * e2x - d[0] * e2z, pz = d[0] * e2y - d[1] * e2x;
      const det = e1x * px + e1y * py + e1z * pz;
      if (Math.abs(det) < 1e-18) continue;
      const inv = 1 / det;
      const tx = ox - ax, ty = oy - ay, tz = oz - az;
      const u = (tx * px + ty * py + tz * pz) * inv;
      if (u < 0 || u > 1) continue;
      const qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x;
      const v = (d[0] * qx + d[1] * qy + d[2] * qz) * inv;
      if (v < 0 || u + v > 1) continue;
      if ((e2x * qx + e2y * qy + e2z * qz) * inv > eps) return true;
    }
    return false;
  }
}
