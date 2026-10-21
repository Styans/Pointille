// Facade mode: spend the drones where the audience can see them. The audience
// is a row of viewers across its width; each is far away (sight lines parallel
// to viewDir, model -> audience) or a point such as a camera, where the sight
// lines converge. A point's visibility is the share of viewers that see it: it
// faces them and no part of the model is in the way. Occlusion uses a 2D grid
// of the triangles projected along the sight lines.

const MAX_PAIRS = 8_000_000;
const MAX_VIEWS = 9;

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

// v turned by angle radians around the unit axis k (Rodrigues)
export function rotate(v, k, angle) {
  const c = Math.cos(angle), s = Math.sin(angle);
  const kv = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];
  const cr = [k[1] * v[2] - k[2] * v[1], k[2] * v[0] - k[0] * v[2], k[0] * v[1] - k[1] * v[0]];
  return [0, 1, 2].map((a) => v[a] * c + cr[a] * s + k[a] * kv * (1 - c));
}

const dot3 = (a, i, b) => a[i] * b[0] + a[i + 1] * b[1] + a[i + 2] * b[2];

function diagonalEps(tris) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let k = 0; k < tris.length; k += 3)
    for (let a = 0; a < 3; a++) {
      lo[a] = Math.min(lo[a], tris[k + a]);
      hi[a] = Math.max(hi[a], tris[k + a]);
    }
  const diag = tris.length ? Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) : 0;
  return Math.max(diag, 1e-12) * 1e-6;
}

// Triangles ids[i] bucketed into square cells by their 2D projection
// (px, py: three corners per triangle).
class ProjectedGrid {
  constructor(px, py, ids) {
    const t = ids.length;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (let k = 0; k < t * 3; k++) {
      minX = Math.min(minX, px[k]); maxX = Math.max(maxX, px[k]);
      minY = Math.min(minY, py[k]); maxY = Math.max(maxY, py[k]);
    }
    if (!t) minX = minY = maxX = maxY = 0;
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
      for (let y = y0[i]; y <= y1[i]; y++) for (let x = x0[i]; x <= x1[i]; x++) this.list[fill[x + this.res * y]++] = ids[i];
  }

  cellX(v) {
    return Math.max(0, Math.floor((v - this.ox) / this.cell));
  }

  cellY(v) {
    return Math.max(0, Math.floor((v - this.oy) / this.cell));
  }

  // the cell holding (x, y); its triangles are list[offsets[c] .. offsets[c + 1])
  cellAt(x, y) {
    return Math.min(this.cellX(x), this.res - 1) + this.res * Math.min(this.cellY(y), this.res - 1);
  }
}

// Möller–Trumbore: does the ray o + t·d with eps < t < reach cross triangle i?
function crosses(T, i, ox, oy, oz, dx, dy, dz, reach, eps) {
  const o = 9 * i;
  const ax = T[o], ay = T[o + 1], az = T[o + 2];
  const e1x = T[o + 3] - ax, e1y = T[o + 4] - ay, e1z = T[o + 5] - az;
  const e2x = T[o + 6] - ax, e2y = T[o + 7] - ay, e2z = T[o + 8] - az;
  const px = dy * e2z - dz * e2y, py = dz * e2x - dx * e2z, pz = dx * e2y - dy * e2x;
  const det = e1x * px + e1y * py + e1z * pz;
  if (Math.abs(det) < 1e-18) return false;
  const inv = 1 / det;
  const tx = ox - ax, ty = oy - ay, tz = oz - az;
  const u = (tx * px + ty * py + tz * pz) * inv;
  if (u < 0 || u > 1) return false;
  const qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x;
  const v = (dx * qx + dy * qy + dz * qz) * inv;
  if (v < 0 || u + v > 1) return false;
  const t = (e2x * qx + e2y * qy + e2z * qz) * inv;
  return t > eps && t < reach;
}

// A far-away viewer: every sight line is parallel to viewDir.
export class OrthoOccluder {
  constructor(tris, viewDir) {
    const { d, e1, e2 } = viewBasis(viewDir);
    this.d = d;
    this.e1 = e1;
    this.e2 = e2;
    this.tris = tris;
    this.eps = diagonalEps(tris);
    const t = tris.length / 9;
    const px = new Float64Array(t * 3), py = new Float64Array(t * 3);
    for (let k = 0; k < t * 3; k++) {
      px[k] = dot3(tris, 3 * k, e1);
      py[k] = dot3(tris, 3 * k, e2);
    }
    this.grid = new ProjectedGrid(px, py, Int32Array.from({ length: t }, (_, i) => i));
  }

  // true when the straight line from (x, y, z) towards the audience hits the model
  occluded(x, y, z, ownTri = -1) {
    const g = this.grid, d = this.d, eps = this.eps;
    const c = g.cellAt(x * this.e1[0] + y * this.e1[1] + z * this.e1[2], x * this.e2[0] + y * this.e2[1] + z * this.e2[2]);
    const ox = x + d[0] * eps, oy = y + d[1] * eps, oz = z + d[2] * eps;
    for (let k = g.offsets[c]; k < g.offsets[c + 1]; k++) {
      const i = g.list[k];
      if (i !== ownTri && crosses(this.tris, i, ox, oy, oz, d[0], d[1], d[2], Infinity, eps)) return true;
    }
    return false;
  }
}

// A viewer at eye looking at target: sight lines converge on the eye.
// Triangles are projected through the eye onto a plane facing target; a
// segment from a point to the eye then projects to a single spot. Triangles
// reaching behind the eye cannot be projected and are tested against every
// point; points behind the eye are not seen at all.
export class PointOccluder {
  constructor(tris, eye, target) {
    const { d, e1, e2 } = viewBasis([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
    this.axis = d;
    this.e1 = e1;
    this.e2 = e2;
    this.eye = eye;
    this.tris = tris;
    this.eps = diagonalEps(tris);
    const near = Math.max(Math.hypot(target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]), this.eps) * 0.02;
    const t = tris.length / 9;
    const px = new Float64Array(t * 3), py = new Float64Array(t * 3), ids = [], always = [];
    for (let i = 0; i < t; i++) {
      let ahead = true;
      const k = ids.length;
      for (let c = 0; c < 3 && ahead; c++) {
        const o = 9 * i + 3 * c;
        const r = [tris[o] - eye[0], tris[o + 1] - eye[1], tris[o + 2] - eye[2]];
        const depth = dot3(r, 0, d);
        if (depth <= near) ahead = false;
        px[3 * k + c] = dot3(r, 0, e1) / depth;
        py[3 * k + c] = dot3(r, 0, e2) / depth;
      }
      (ahead ? ids : always).push(i);
    }
    this.grid = new ProjectedGrid(px.subarray(0, 3 * ids.length), py.subarray(0, 3 * ids.length), Int32Array.from(ids));
    this.always = Int32Array.from(always);
  }

  occluded(x, y, z, ownTri = -1) {
    const eps = this.eps, r = [x - this.eye[0], y - this.eye[1], z - this.eye[2]];
    const depth = dot3(r, 0, this.axis);
    if (depth <= eps) return true; // behind the eye
    const dist = Math.hypot(r[0], r[1], r[2]);
    const dx = -r[0] / dist, dy = -r[1] / dist, dz = -r[2] / dist;
    const ox = x + dx * eps, oy = y + dy * eps, oz = z + dz * eps;
    const reach = dist - eps;
    const g = this.grid;
    const c = g.cellAt(dot3(r, 0, this.e1) / depth, dot3(r, 0, this.e2) / depth);
    for (let k = g.offsets[c]; k < g.offsets[c + 1]; k++) {
      const i = g.list[k];
      if (i !== ownTri && crosses(this.tris, i, ox, oy, oz, dx, dy, dz, reach, eps)) return true;
    }
    for (const i of this.always) if (i !== ownTri && crosses(this.tris, i, ox, oy, oz, dx, dy, dz, reach, eps)) return true;
    return false;
  }
}

// Viewers spread over `spread` degrees around the vertical axis. viewPoint
// puts the middle viewer at a point (a camera) and turns the others around
// the model's centre; otherwise the viewers are far away along viewDir.
export class Audience {
  constructor(tris, normals, { viewDir = [0, -1, 0], viewPoint = null, spread = 0, facadeAngle = 80, occlusion = true, upAxis = 2 } = {}) {
    this.tris = tris;
    this.normals = normals;
    this.cos = Math.cos((facadeAngle * Math.PI) / 180);
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let k = 0; k < tris.length; k += 3)
      for (let a = 0; a < 3; a++) {
        lo[a] = Math.min(lo[a], tris[k + a]);
        hi[a] = Math.max(hi[a], tris[k + a]);
      }
    const center = [0, 1, 2].map((a) => (lo[a] + hi[a]) / 2);
    const up = [0, 0, 0];
    up[upAxis] = 1;
    const width = (Math.min(Math.max(+spread || 0, 0), 360) * Math.PI) / 180;
    const k = width <= 0 ? 1 : Math.min(Math.max(Math.ceil((width * 180) / Math.PI / 10) + 1, 3), MAX_VIEWS);
    const full = width >= 2 * Math.PI - 1e-9; // the first and the last viewer would stand at the same spot
    const angles = k === 1 ? [0] : Array.from({ length: k }, (_, i) => -width / 2 + (width * i) / (full ? k : k - 1));

    this.points = viewPoint != null;
    const middle = this.points ? [0, 1, 2].map((a) => viewPoint[a] - center[a]) : normalize(viewDir);
    if (!Math.hypot(middle[0], middle[1], middle[2])) throw new Error("the camera is at the centre of the model");
    const m = normalize(middle);
    const axis = Math.abs(m[0] * up[0] + m[1] * up[1] + m[2] * up[2]) < 0.9 ? up : viewBasis(middle).e1;
    this.views = angles.map((a) => rotate(middle, axis, a));
    if (this.points) this.views = this.views.map((v) => [0, 1, 2].map((a) => center[a] + v[a]));
    this.occluders = this.views.map((v) =>
      !occlusion ? null : this.points ? new PointOccluder(tris, v, center) : new OrthoOccluder(tris, v),
    );
  }

  // triangles at least one viewer may see (the exact test is per point)
  triangleMask() {
    const t = this.normals.length / 3, T = this.tris, N = this.normals;
    const out = new Uint8Array(t);
    for (let i = 0; i < t; i++)
      for (const v of this.views) {
        const f = this.points
          ? N[3 * i] * (v[0] - T[9 * i]) + N[3 * i + 1] * (v[1] - T[9 * i + 1]) + N[3 * i + 2] * (v[2] - T[9 * i + 2]) > 0
          : dot3(N, 3 * i, v) > this.cos;
        if (f) {
          out[i] = 1;
          break;
        }
      }
    return out;
  }

  // share of the viewers that see the point, in [0, 1]
  visibility(x, y, z, tri) {
    const N = this.normals, o = 3 * tri;
    let seen = 0;
    this.views.forEach((v, k) => {
      let facing;
      if (this.points) {
        const tx = v[0] - x, ty = v[1] - y, tz = v[2] - z;
        facing = N[o] * tx + N[o + 1] * ty + N[o + 2] * tz > this.cos * Math.hypot(tx, ty, tz);
      } else {
        facing = dot3(N, o, v) > this.cos;
      }
      const occ = this.occluders[k];
      if (facing && !(occ && occ.occluded(x, y, z, tri))) seen++;
    });
    return seen / this.views.length;
  }
}
