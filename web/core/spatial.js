// Uniform-grid neighbour queries over flat xyz arrays.

export class Grid {
  constructor(points, cell) {
    const n = points.length / 3;
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++) {
      for (let a = 0; a < 3; a++) {
        const v = points[3 * i + a];
        if (v < lo[a]) lo[a] = v;
        if (v > hi[a]) hi[a] = v;
      }
    }
    if (!n) lo.fill(0), hi.fill(0);
    const extent = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
    // keep linear cell keys exact in a double; a coarser grid is still correct
    this.cell = Math.max(cell, extent / 1e5, 1e-300);
    this.origin = lo;
    this.dims = [0, 1, 2].map((a) => Math.floor((hi[a] - lo[a]) / this.cell) + 3);
    this.points = points;
    const keys = new Float64Array(n);
    for (let i = 0; i < n; i++) keys[i] = this.keyOf(points[3 * i], points[3 * i + 1], points[3 * i + 2]);
    const order = new Uint32Array(n);
    for (let i = 0; i < n; i++) order[i] = i;
    order.sort((x, y) => keys[x] - keys[y] || x - y);
    this.order = order;
    this.cells = new Map();
    this.starts = [];
    this.counts = [];
    for (let k = 0; k < n; k++) {
      const key = keys[order[k]];
      if (k === 0 || key !== keys[order[k - 1]]) {
        this.cells.set(key, this.starts.length);
        this.starts.push(k);
        this.counts.push(0);
      }
      this.counts[this.counts.length - 1]++;
    }
  }

  coord(v, a) {
    return Math.floor((v - this.origin[a]) / this.cell) + 1;
  }

  keyOf(x, y, z) {
    const [nx, ny] = this.dims;
    return this.coord(x, 0) + nx * (this.coord(y, 1) + ny * this.coord(z, 2));
  }

  // Calls fn(j, distance) for every grid point closer than r (r <= cell).
  forEachNear(x, y, z, r, fn) {
    const [nx, ny, nz] = this.dims;
    const cx = this.coord(x, 0), cy = this.coord(y, 1), cz = this.coord(z, 2);
    const r2 = r * r;
    const p = this.points;
    for (let dz = -1; dz <= 1; dz++) {
      const iz = cz + dz;
      if (iz < 0 || iz >= nz) continue;
      for (let dy = -1; dy <= 1; dy++) {
        const iy = cy + dy;
        if (iy < 0 || iy >= ny) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const ix = cx + dx;
          if (ix < 0 || ix >= nx) continue;
          const c = this.cells.get(ix + nx * (iy + ny * iz));
          if (c === undefined) continue;
          const end = this.starts[c] + this.counts[c];
          for (let k = this.starts[c]; k < end; k++) {
            const j = this.order[k];
            const ddx = p[3 * j] - x, ddy = p[3 * j + 1] - y, ddz = p[3 * j + 2] - z;
            const d2 = ddx * ddx + ddy * ddy + ddz * ddz;
            if (d2 < r2) fn(j, Math.sqrt(d2));
          }
        }
      }
    }
  }
}

function extentOf(a) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < a.length; i++) {
    if (a[i] < lo) lo = a[i];
    if (a[i] > hi) hi = a[i];
  }
  return a.length ? hi - lo : 0;
}

// Nearest point for every query: {dist: Float64Array, index: Int32Array}.
export function nearest(queries, points, guess, excludeSelf = false) {
  const n = queries.length / 3;
  const dist = new Float64Array(n).fill(Infinity);
  const index = new Int32Array(n).fill(-1);
  if (!n || !points.length || (excludeSelf && points.length < 6)) return { dist, index };
  const extent = Math.max(extentOf(points), extentOf(queries)) || 1;
  let radius = Math.max(guess, extent * 1e-6);
  let todo = Array.from({ length: n }, (_, i) => i);
  while (todo.length) {
    const grid = new Grid(points, radius);
    const r = Math.min(radius, grid.cell);
    for (const q of todo) {
      grid.forEachNear(queries[3 * q], queries[3 * q + 1], queries[3 * q + 2], r, (j, d) => {
        if (excludeSelf && j === q) return;
        if (d < dist[q] || (d === dist[q] && j < index[q])) {
          dist[q] = d;
          index[q] = j;
        }
      });
    }
    todo = todo.filter((q) => index[q] < 0);
    if (radius > extent * 4) break;
    radius *= 2;
  }
  return { dist, index };
}

export function nnDistances(points, guess) {
  return nearest(points, points, guess, true).dist;
}
