// Outline: drones on the lines that draw the model, closer together than the fill.
// Mirrors pointille/core/outline.py, where the method is described: borders and
// sharp creases from the mesh; the silhouette, the relief (eyes, mouth, folds)
// and colour borders read from rendered views of the model by an edge detector;
// all chained into lines between corners, a drone in every corner, evenly
// spaced drones between, then a relaxation along the lines.
import { linearToOklab, surfaceColors } from "./color.js";
import { normalize, rotate, viewBasis } from "./facade.js";
import { SurfaceProjector, relax } from "./relax.js";
import { poissonRadius } from "./sampling.js";
import { Grid, nearest } from "./spatial.js";

export const AMOUNTS = ["density", "share"];
const STEP = 1 / 16; // line points per fill spacing: fine enough for drones to slide along
const PIECE = 4; // segments are cut into pieces of at most this many steps for the sight tests
const MIN_LENGTH = 1; // lines that all fit in less than one fill spacing are noise
const MAX_SHARE = 0.5; // "density" puts at most half of the drones on lines
const MIN_SPACING = 0.4; // "share" never packs a line closer than this many fill spacings
const GAP = 0.75; // the fill keeps this many fill spacings clear of the lines
const CORNER_TURN = 45; // a line bending sharper than this (degrees) keeps a drone in the corner
const CORNER_GAP = 0.8; // corner drones closer than this many line spacings to a stronger one are dropped
const CORNER_WINDOW = 0.25; // bends and lengths are judged over this many fill spacings of line
const LINE_GAP = 0.7; // where two lines run close, drones nearer than this many line spacings to another go
const VIEW_PIXEL = 1 / 12; // pixel of the views silhouette, relief and colours are read from, in fill spacings
const VIEW_SIGMA = 2; // the views are smoothed over this many pixels: finer detail draws no line
const INSET = 1e-3; // points on a mesh edge sit a hair inside their triangle, so sight tests do not graze it
const VIEW_NODES = 2 ** 40; // node names of the view lines, apart from the segments' ones
const MAX_PIXELS = 4000000;

const dot3 = (A, i, B, j) => A[i] * B[j] + A[i + 1] * B[j + 1] + A[i + 2] * B[j + 2];

// Straight segments inside triangles, barycentrics b0 -> b1, end nodes k0 / k1.
function makeSegs(tri, b0, b1, k0, k1) {
  return { tri: Int32Array.from(tri), b0: Float64Array.from(b0), b1: Float64Array.from(b1), k0: Float64Array.from(k0), k1: Float64Array.from(k1) };
}

function takeSegs(s, keep) {
  const tri = [], b0 = [], b1 = [], k0 = [], k1 = [];
  for (let i = 0; i < s.tri.length; i++) {
    if (!keep[i]) continue;
    tri.push(s.tri[i]);
    k0.push(s.k0[i]);
    k1.push(s.k1[i]);
    for (let a = 0; a < 3; a++) b0.push(s.b0[3 * i + a]), b1.push(s.b1[3 * i + a]);
  }
  return makeSegs(tri, b0, b1, k0, k1);
}

function concatSegs(parts) {
  const out = { tri: [], b0: [], b1: [], k0: [], k1: [] };
  for (const p of parts) for (const f of Object.keys(out)) for (const v of p[f]) out[f].push(v);
  return makeSegs(out.tri, out.b0, out.b1, out.k0, out.k1);
}

function baryPoint(tris, t, B, i, out = [0, 0, 0], o = 0) {
  const T = 9 * t;
  for (let a = 0; a < 3; a++) out[o + a] = B[i] * tris[T + a] + B[i + 1] * tris[T + 3 + a] + B[i + 2] * tris[T + 6 + a];
  return out;
}

function segEnds(s, tris) {
  const n = s.tri.length, p0 = new Float64Array(3 * n), p1 = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) {
    baryPoint(tris, s.tri[i], s.b0, 3 * i, p0, 3 * i);
    baryPoint(tris, s.tri[i], s.b1, 3 * i, p1, 3 * i);
  }
  return { p0, p1 };
}

function segMids(s, tris) {
  const { p0, p1 } = segEnds(s, tris);
  return p0.map((v, i) => 0.5 * (v + p1[i]));
}

function onehot(k) {
  const b = [INSET / 3, INSET / 3, INSET / 3];
  b[k] = 1 - (2 * INSET) / 3;
  return b;
}

// segments cut into pieces no longer than maxLen; the cuts get new node names from nextKey on
function cut(s, tris, maxLen, nextKey) {
  const { p0, p1 } = segEnds(s, tris);
  const tri = [], b0 = [], b1 = [], k0 = [], k1 = [];
  for (let i = 0; i < s.tri.length; i++) {
    const len = Math.hypot(p1[3 * i] - p0[3 * i], p1[3 * i + 1] - p0[3 * i + 1], p1[3 * i + 2] - p0[3 * i + 2]);
    const m = Math.max(1, Math.ceil(len / maxLen));
    for (let j = 0; j < m; j++) {
      const f0 = j / m, f1 = (j + 1) / m;
      tri.push(s.tri[i]);
      for (let a = 0; a < 3; a++) {
        b0.push(s.b0[3 * i + a] * (1 - f0) + s.b1[3 * i + a] * f0);
        b1.push(s.b0[3 * i + a] * (1 - f1) + s.b1[3 * i + a] * f1);
      }
      k0.push(j === 0 ? s.k0[i] : nextKey + j - 1);
      k1.push(j === m - 1 ? s.k1[i] : nextKey + j);
    }
    nextKey += m - 1;
  }
  return { segs: makeSegs(tri, b0, b1, k0, k1), nextKey };
}

// points every `step` or closer along the segments, their ends left out
function segPoints(s, tris, step) {
  const { p0, p1 } = segEnds(s, tris);
  const pos = [], tri = [], bary = [];
  const p = [0, 0, 0], b = [0, 0, 0];
  for (let i = 0; i < s.tri.length; i++) {
    const len = Math.hypot(p1[3 * i] - p0[3 * i], p1[3 * i + 1] - p0[3 * i + 1], p1[3 * i + 2] - p0[3 * i + 2]);
    const m = Math.max(1, Math.ceil(len / step));
    for (let j = 0; j < m; j++) {
      const f = (j + 0.5) / m;
      for (let a = 0; a < 3; a++) b[a] = s.b0[3 * i + a] * (1 - f) + s.b1[3 * i + a] * f;
      baryPoint(tris, s.tri[i], b, 0, p);
      pos.push(...p);
      tri.push(s.tri[i]);
      bary.push(...b);
    }
  }
  return { pos, tri, bary };
}

// unit vectors from points towards the middle viewer
function sightOf(opt) {
  if (opt.viewPoint != null) {
    const e = opt.viewPoint;
    return (x, y, z) => {
      const dx = e[0] - x, dy = e[1] - y, dz = e[2] - z, l = Math.hypot(dx, dy, dz) || 1e-300;
      return [dx / l, dy / l, dz / l];
    };
  }
  const v = normalize(opt.viewDir);
  return () => v;
}

// vertex ids of the corners merged by position, and the vertices
function weld(tris) {
  const n = tris.length / 3;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++)
    for (let a = 0; a < 3; a++) (lo[a] = Math.min(lo[a], tris[3 * i + a])), (hi[a] = Math.max(hi[a], tris[3 * i + a]));
  const q = (Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1) * 1e-5; // < 2**17 per axis
  const ids = new Map(), vid = new Int32Array(n), verts = [];
  for (let i = 0; i < n; i++) {
    const c = [0, 1, 2].map((a) => Math.floor((tris[3 * i + a] - lo[a]) / q + 0.5));
    const key = c[0] + 131072 * (c[1] + 131072 * c[2]);
    let id = ids.get(key);
    if (id === undefined) {
      id = verts.length / 3;
      ids.set(key, id);
      verts.push(tris[3 * i], tris[3 * i + 1], tris[3 * i + 2]);
    }
    vid[i] = id;
  }
  return { vid, verts: Float64Array.from(verts) };
}

// half-edge 3t + k runs from corner k of triangle t to corner k + 1: its mesh edge, start and end vertex
function edgesOf(vid) {
  const h = vid.length, nv = vid.reduce((m, v) => Math.max(m, v), 0) + 1;
  const edge = new Int32Array(h), va = new Int32Array(h), vb = new Int32Array(h), ids = new Map();
  for (let i = 0; i < h; i++) {
    const t = (i / 3) | 0, k = i % 3;
    va[i] = vid[i];
    vb[i] = vid[3 * t + ((k + 1) % 3)];
    const key = Math.min(va[i], vb[i]) * nv + Math.max(va[i], vb[i]);
    let e = ids.get(key);
    if (e === undefined) ids.set(key, (e = ids.size));
    edge[i] = e;
  }
  return { edge, va, vb, count: ids.size };
}

// half-edges on the borders and on the sharp creases, one per mesh edge
// (on a crease the one whose triangle faces the viewer more)
function featureEdges(tris, normals, live, E, creaseCos, sight) {
  const first = new Int32Array(E.count).fill(-1), second = new Int32Array(E.count).fill(-1), sides = new Int32Array(E.count);
  for (let he = 0; he < E.edge.length; he++) {
    if (!live[(he / 3) | 0] || E.va[he] === E.vb[he]) continue;
    const e = E.edge[he];
    if (first[e] < 0) first[e] = he;
    else if (second[e] < 0) second[e] = he;
    sides[e]++;
  }
  const border = [], crease = [];
  for (let e = 0; e < E.count; e++) {
    if (sides[e] === 1) border.push(first[e]);
    if (sides[e] < 2) continue;
    const a = first[e], b = second[e], ta = (a / 3) | 0, tb = (b / 3) | 0;
    let cos = dot3(normals, 3 * ta, normals, 3 * tb);
    if (E.va[a] === E.va[b]) cos = -cos; // both run the same way: the faces wind oppositely
    if (!(sides[e] > 2 || cos < creaseCos)) continue;
    const o = 9 * ta + 3 * (a % 3), o2 = 9 * ta + 3 * ((a % 3 + 1) % 3);
    const s = sight(0.5 * (tris[o] + tris[o2]), 0.5 * (tris[o + 1] + tris[o2 + 1]), 0.5 * (tris[o + 2] + tris[o2 + 2]));
    crease.push(dot3(normals, 3 * tb, s, 0) > dot3(normals, 3 * ta, s, 0) ? b : a);
  }
  return { border, crease };
}

function edgeSegs(he, E) {
  const tri = [], b0 = [], b1 = [], k0 = [], k1 = [];
  for (const h of he) {
    tri.push((h / 3) | 0);
    b0.push(...onehot(h % 3));
    b1.push(...onehot((h + 1) % 3));
    k0.push(E.va[h]);
    k1.push(E.vb[h]);
  }
  return makeSegs(tri, b0, b1, k0, k1);
}

// normal at each triangle corner: the area-weighted mean of the triangles around the
// vertex that meet this one smoothly (auto smooth)
function cornerNormals(vid, normals, areas, smoothCos) {
  const c = vid.length, nv = vid.reduce((m, v) => Math.max(m, v), 0) + 1;
  const start = new Int32Array(nv + 1);
  for (const v of vid) start[v + 1]++;
  for (let v = 0; v < nv; v++) start[v + 1] += start[v];
  const at = start.slice(0, nv), list = new Int32Array(c);
  for (let i = 0; i < c; i++) list[at[vid[i]]++] = i;
  const out = new Float64Array(3 * c);
  for (let i = 0; i < c; i++) {
    const t = (i / 3) | 0, v = vid[i];
    for (let k = start[v]; k < start[v + 1]; k++) {
      const u = (list[k] / 3) | 0;
      if (dot3(normals, 3 * t, normals, 3 * u) < smoothCos) continue;
      for (let a = 0; a < 3; a++) out[3 * i + a] += areas[u] * normals[3 * u + a];
    }
    const l = Math.hypot(out[3 * i], out[3 * i + 1], out[3 * i + 2]);
    for (let a = 0; a < 3; a++) out[3 * i + a] = l > 0 ? out[3 * i + a] / l : normals[3 * t + a];
  }
  return out;
}

// labels of the connected components of n nodes joined by the pairs (i[k], j[k])
function components(n, i, j) {
  const parent = Int32Array.from({ length: n }, (_, k) => k);
  const root = (x) => {
    while (parent[x] !== x) x = parent[x] = parent[parent[x]];
    return x;
  };
  for (let k = 0; k < i.length; k++) {
    const a = root(i[k]), b = root(j[k]);
    if (a !== b) parent[Math.max(a, b)] = Math.min(a, b);
  }
  for (let k = 0; k < n; k++) parent[k] = root(k);
  return parent;
}

// for each edge (node k0 at p0 to node k1 at p1): is its connected line at least `least` across?
function wide(k0, k1, p0, p1, least) {
  const s = k0.length, ids = new Map();
  const idOf = (key) => {
    let id = ids.get(key);
    if (id === undefined) ids.set(key, (id = ids.size));
    return id;
  };
  const a = Array.from(k0, idOf), b = Array.from(k1, idOf);
  const label = components(ids.size, a, b);
  const lo = new Float64Array(3 * ids.size).fill(Infinity), hi = new Float64Array(3 * ids.size).fill(-Infinity);
  for (let e = 0; e < s; e++)
    for (const [P, l] of [[p0, label[a[e]]], [p1, label[b[e]]]])
      for (let c = 0; c < 3; c++) (lo[3 * l + c] = Math.min(lo[3 * l + c], P[3 * e + c])), (hi[3 * l + c] = Math.max(hi[3 * l + c], P[3 * e + c]));
  return Uint8Array.from(a, (_, e) => {
    const l = label[a[e]];
    return Math.hypot(hi[3 * l] - lo[3 * l], hi[3 * l + 1] - lo[3 * l + 1], hi[3 * l + 2] - lo[3 * l + 2]) >= least ? 1 : 0;
  });
}

// for each query: is some point closer than radius?
function near(queries, points, radius) {
  const n = queries.length / 3, out = new Uint8Array(n);
  if (!n || !points.length || !(radius > 0)) return out;
  const grid = new Grid(points, radius);
  for (let q = 0; q < n; q++)
    grid.forEachNear(queries[3 * q], queries[3 * q + 1], queries[3 * q + 2], radius, () => {
      out[q] = 1;
    });
  return out;
}

// indices of the points (taken in order) that keep `radius` from every one kept before
function park(points, radius) {
  const cells = new Map(), kept = [], r2 = radius * radius;
  for (let i = 0; i < points.length / 3; i++) {
    const x = points[3 * i], y = points[3 * i + 1], z = points[3 * i + 2];
    const cx = Math.floor(x / radius), cy = Math.floor(y / radius), cz = Math.floor(z / radius);
    let clear = true;
    for (let dx = -1; dx <= 1 && clear; dx++)
      for (let dy = -1; dy <= 1 && clear; dy++)
        for (let dz = -1; dz <= 1 && clear; dz++)
          for (const q of cells.get(`${cx + dx},${cy + dy},${cz + dz}`) || [])
            if ((x - q[0]) ** 2 + (y - q[1]) ** 2 + (z - q[2]) ** 2 < r2) {
              clear = false;
              break;
            }
    if (!clear) continue;
    const key = `${cx},${cy},${cz}`;
    if (!cells.has(key)) cells.set(key, []);
    cells.get(key).push([x, y, z]);
    kept.push(i);
  }
  return kept;
}

// A line from corner `start` to corner `end` (or round a loop): its nodes p (flat) in walking order,
// a loop's last node back on its first. Segment pieces lie in triangle tri[i] (barycentrics ba -> bb);
// pieces read from a view have no triangle, their points are put back onto the surface. Distances
// along the line are chords between points `res` apart, so wiggles smaller than that do not count.
// A line read from a view is measured in the picture, p its nodes there (drones evenly spaced in the
// picture are no closer in space), and q holds the same nodes on the model.
class Chain {
  constructor(p, tri, ba, bb, nodeTri, nodeBary, closed, start, end, res, q = null) {
    Object.assign(this, { p, tri, ba, bb, nodeTri, nodeBary, closed, start, end, res, q });
    const n = p.length / 3;
    this.t = new Float64Array(n);
    for (let i = 1; i < n; i++)
      this.t[i] = this.t[i - 1] + Math.hypot(p[3 * i] - p[3 * i - 3], p[3 * i + 1] - p[3 * i - 2], p[3 * i + 2] - p[3 * i - 1]);
    const total = this.t[n - 1], k = Math.max(2, Math.ceil(total / res) + 1);
    this.tk = Float64Array.from({ length: k }, (_, i) => (total * i) / (k - 1));
    this.ck = new Float64Array(k);
    let prev = this.point(0);
    for (let i = 1; i < k; i++) {
      const q = this.point(this.tk[i]);
      this.ck[i] = this.ck[i - 1] + Math.hypot(q[0] - prev[0], q[1] - prev[1], q[2] - prev[2]);
      prev = q;
    }
  }

  get length() {
    return this.ck[this.ck.length - 1];
  }

  // index of the piece holding arc length `arc` and the fraction along it
  locate(arc) {
    const t = this.t;
    let lo = 0, hi = t.length - 2;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (t[mid] <= arc) lo = mid;
      else hi = mid - 1;
    }
    const f = Math.min(Math.max((arc - t[lo]) / Math.max(t[lo + 1] - t[lo], 1e-300), 0), 1);
    return [lo, f];
  }

  // position at arc length `arc` (round and round a loop, clamped to the ends otherwise)
  point(arc) {
    const total = this.t[this.t.length - 1];
    arc = this.closed && total > 0 ? ((arc % total) + total) % total : Math.min(Math.max(arc, 0), total);
    const [i, f] = this.locate(arc), p = this.p;
    return [0, 1, 2].map((a) => p[3 * i + a] * (1 - f) + p[3 * i + 3 + a] * f);
  }

  // points at distances xs along the line, pushed onto out = {pos, tri, bary}
  at(xs, tris, surface, out) {
    for (const x of xs) {
      const arc = interp(x, this.ck, this.tk);
      const [i, f] = this.locate(arc);
      if (this.tri) {
        const b = [0, 1, 2].map((a) => this.ba[3 * i + a] * (1 - f) + this.bb[3 * i + a] * f);
        out.pos.push(...baryPoint(tris, this.tri[i], b, 0));
        out.tri.push(this.tri[i]);
        out.bary.push(...b);
      } else {
        const Q = this.q || this.p;
        const hit = surface.project([0, 1, 2].map((a) => Q[3 * i + a] * (1 - f) + Q[3 * i + 3 + a] * f));
        if (!hit) continue;
        out.pos.push(...hit.pos);
        out.tri.push(hit.tri);
        out.bary.push(...hit.bary);
      }
    }
  }

  // node i on the model
  node(i) {
    const Q = this.q || this.p;
    return [Q[3 * i], Q[3 * i + 1], Q[3 * i + 2]];
  }

  // the open line from node a to node b; on a loop b may run past its last node
  cut(a, b, start, end) {
    const n = this.p.length / 3 - 1;
    const nodes = Array.from({ length: b - a + 1 }, (_, i) => (this.closed ? (a + i) % n : a + i));
    const pick = (A, idx, w) => (A ? Float64Array.from(idx.flatMap((i) => Array.from(A.subarray(w * i, w * i + w)))) : null);
    const pieces = nodes.slice(0, -1);
    return new Chain(pick(this.p, nodes, 3), this.tri && Int32Array.from(pieces, (i) => this.tri[i]), pick(this.ba, pieces, 3),
      pick(this.bb, pieces, 3), Int32Array.from(nodes, (i) => this.nodeTri[i]), pick(this.nodeBary, nodes, 3), false, start, end, this.res,
      pick(this.q, nodes, 3));
  }
}

function interp(x, xs, ys) {
  if (x <= xs[0]) return ys[0];
  const n = xs.length;
  if (x >= xs[n - 1]) return ys[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid;
    else hi = mid;
  }
  const f = (x - xs[lo]) / Math.max(xs[hi] - xs[lo], 1e-300);
  return ys[lo] + (ys[hi] - ys[lo]) * f;
}

// Edges (node k0[e] at p0 to node k1[e] at p1) chained into lines between the nodes where lines end or
// meet, or round a loop. Chains are {edges, forward, closed, start, end}; corners are those nodes,
// {node, end, strength}, `end` indexing [p0, p1] (e or s + e) at the edge end `prefer` scores highest.
function chainsOf(k0, k1, prefer) {
  const s = k0.length;
  const keys = Array.from(new Set([...k0, ...k1])).sort((x, y) => x - y);
  const id = new Map(keys.map((k, i) => [k, i]));
  const ends = Int32Array.from([...k0, ...k1], (k) => id.get(k));
  const order = Array.from({ length: 2 * s }, (_, h) => h).sort((x, y) => ends[x] - ends[y] || prefer[y] - prefer[x] || x - y);
  const start = new Int32Array(keys.length + 1);
  for (const h of order) start[ends[h] + 1]++;
  for (let v = 0; v < keys.length; v++) start[v + 1] += start[v];
  const degree = (v) => start[v + 1] - start[v];
  const pick = Array.from({ length: keys.length }, (_, v) => v).filter((v) => degree(v) !== 2)
    .sort((x, y) => degree(y) - degree(x) || x - y);
  const visited = new Uint8Array(s), chains = [];
  const walk = (node, h) => {
    const edges = [], forward = [], from = node;
    for (;;) {
      const e = h % s, back = h >= s;
      visited[e] = 1;
      edges.push(e);
      forward.push(!back);
      node = ends[back ? e : s + e];
      if (degree(node) !== 2) return { edges, forward, closed: false, start: keys[from], end: keys[node] };
      const a = order[start[node]], b = order[start[node] + 1];
      h = a % s === e ? b : a;
      if (visited[h % s]) return { edges, forward, closed: true, start: keys[from], end: keys[node] };
    }
  };
  for (const v of pick) for (let k = start[v]; k < start[v + 1]; k++) if (!visited[order[k] % s]) chains.push(walk(v, order[k]));
  for (let e = 0; e < s; e++) if (!visited[e]) chains.push(walk(ends[e], e)); // a loop with no end
  return { chains, corners: pick.map((v) => ({ node: keys[v], end: order[start[v]], strength: 2 + degree(v) })) };
}

// the chains cut where they bend sharper than CORNER_TURN over `window` of line on either side, at
// the sharpest node of each bend; the new corners get node names below 0
function bends(chains, window) {
  const out = [], corners = [], limit = Math.cos((CORNER_TURN * Math.PI) / 180);
  let name = -1;
  for (const c of chains) {
    const t = c.t, total = t[t.length - 1], n = t.length - 1;
    const inner = [];
    for (let i = 0; i < (c.closed ? n : n + 1); i++) if (c.closed || (t[i] >= window && t[i] <= total - window)) inner.push(i);
    if (total <= 2 * window || !inner.length) {
      out.push(c);
      continue;
    }
    const bend = inner.map((i) => {
      const here = [c.p[3 * i], c.p[3 * i + 1], c.p[3 * i + 2]];
      const back = c.point(t[i] - window), fwd = c.point(t[i] + window);
      const d1 = here.map((v, a) => v - back[a]), d2 = fwd.map((v, a) => v - here[a]);
      return (d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]) / Math.max(Math.hypot(...d1) * Math.hypot(...d2), 1e-300);
    });
    const cuts = [], names = new Map();
    for (const k of inner.map((_, k) => k).sort((x, y) => bend[x] - bend[y] || x - y)) {
      if (bend[k] >= limit) break;
      const i = inner[k];
      if (cuts.some((j) => {
        let gap = Math.abs(t[j] - t[i]);
        if (c.closed) gap = Math.min(gap, total - gap);
        return gap <= window;
      })) continue;
      cuts.push(i);
      names.set(i, name);
      corners.push({ node: name, pos: c.node(i), tri: c.nodeTri[i], bary: Array.from(c.nodeBary.subarray(3 * i, 3 * i + 3)), strength: 1 - bend[k] });
      name--;
    }
    if (!cuts.length) {
      out.push(c);
      continue;
    }
    cuts.sort((x, y) => x - y);
    if (c.closed) {
      // a loop opens at its first cut and ends there again
      cuts.forEach((a, k) => {
        const b = k + 1 < cuts.length ? cuts[k + 1] : cuts[0] + n;
        out.push(c.cut(a, b, names.get(a), names.get(b % n)));
      });
    } else {
      const at = [0, ...cuts, n], keys = [c.start, ...cuts.map((i) => names.get(i)), c.end];
      for (let k = 0; k + 1 < at.length; k++) out.push(c.cut(at[k], at[k + 1], keys[k], keys[k + 1]));
    }
  }
  return { chains: out, corners };
}

// for each edge (node k0 at p0 to node k1 at p1): is it off the spurs, branches from a loose end to a
// fork shorter than `least` (the hairs a noisy line or a thinned band of pixels grows)?
function unspurred(k0, k1, p0, p1, least) {
  const keep = new Uint8Array(k0.length).fill(1);
  for (let round = 0; round < 3; round++) {
    // a fork left with two branches may have been a spur's root
    const live = [];
    keep.forEach((k, e) => k && live.push(e));
    const ids = new Map(), touch = [];
    const idOf = (key) => {
      let id = ids.get(key);
      if (id === undefined) ids.set(key, (id = ids.size)), touch.push([]);
      return id;
    };
    const a = live.map((e) => idOf(k0[e])), b = live.map((e) => idOf(k1[e]));
    const length = live.map((e) => Math.hypot(p1[3 * e] - p0[3 * e], p1[3 * e + 1] - p0[3 * e + 1], p1[3 * e + 2] - p0[3 * e + 2]));
    a.forEach((v, e) => touch[v].push(e));
    b.forEach((v, e) => touch[v].push(e));
    const degree = (v) => touch[v].length;
    const cuts = [];
    for (let leaf = 0; leaf < touch.length; leaf++) {
      if (degree(leaf) !== 1) continue;
      const path = [];
      let node = leaf, came = -1, total = 0;
      for (;;) {
        const next = touch[node].filter((e) => e !== came);
        if (degree(node) >= 3 || !next.length || (node !== leaf && degree(node) !== 2)) break;
        came = next[0];
        path.push(came);
        total += length[came];
        node = a[came] === node ? b[came] : a[came];
        if (total >= least) break;
      }
      if (path.length && degree(node) >= 3 && total < least) cuts.push(...path);
    }
    if (!cuts.length) break;
    for (const e of cuts) keep[live[e]] = 0;
  }
  return keep;
}

// (eye, toward) of the views silhouette, relief and colours are read from: the middle viewer in
// facade mode; otherwise it, the three views round the vertical axis and the one from above, for a
// show seen from every side. eye null is a viewer far away in the direction toward.
function viewsOf(opt, tris) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let k = 0; k < tris.length; k += 3)
    for (let a = 0; a < 3; a++) (lo[a] = Math.min(lo[a], tris[k + a])), (hi[a] = Math.max(hi[a], tris[k + a]));
  const center = [0, 1, 2].map((a) => (lo[a] + hi[a]) / 2);
  const middle = opt.viewPoint != null ? [0, 1, 2].map((a) => opt.viewPoint[a] - center[a]) : normalize(opt.viewDir);
  let out = [middle];
  if (!opt.facade) {
    const up = [0, 0, 0];
    up[opt.upAxis] = 1;
    out = [0, 0.5, 1, 1.5].map((k) => rotate(middle, up, k * Math.PI));
    out.push(up.map((v) => v * Math.hypot(...middle)));
  }
  return out.map((v) => ({ eye: opt.viewPoint != null ? [0, 1, 2].map((a) => center[a] + v[a]) : null, toward: v }));
}

// What a view of the model shows: a visibility buffer, the triangle and (perspective-correct)
// barycentrics each pixel sees, or null when it shows nothing. eye null is a viewer far away in the
// direction toward; otherwise h is the pixel size at the model's centre.
function render(tris, eye, toward, h) {
  const { d, e1, e2 } = viewBasis(toward);
  const T = tris.length / 9, far = Math.hypot(...toward);
  const ids = [], X = [], Y = [], near = [], W8 = [];
  for (let t = 0; t < T; t++) {
    const xs = [], ys = [], ns = [], ws = [];
    let ok = true;
    for (let k = 0; k < 3; k++) {
      const o = 9 * t + 3 * k;
      if (!eye) {
        xs.push(tris[o] * e1[0] + tris[o + 1] * e1[1] + tris[o + 2] * e1[2]);
        ys.push(tris[o] * e2[0] + tris[o + 1] * e2[1] + tris[o + 2] * e2[2]);
        ns.push(tris[o] * d[0] + tris[o + 1] * d[1] + tris[o + 2] * d[2]); // nearer is larger
      } else {
        const r = [tris[o] - eye[0], tris[o + 1] - eye[1], tris[o + 2] - eye[2]];
        const z = -(r[0] * d[0] + r[1] * d[1] + r[2] * d[2]); // distance in front of the eye
        if (!(z > 0.02 * far)) ok = false;
        xs.push(((r[0] * e1[0] + r[1] * e1[1] + r[2] * e1[2]) * far) / z);
        ys.push(((r[0] * e2[0] + r[1] * e2[1] + r[2] * e2[2]) * far) / z);
        ns.push(1 / z); // 1/z is affine in the picture
        ws.push(z);
      }
    }
    if (!ok) continue;
    ids.push(t);
    X.push(...xs);
    Y.push(...ys);
    near.push(...ns);
    if (eye) W8.push(...ws);
  }
  if (!ids.length) return null;
  let lx = Infinity, ly = Infinity, hx = -Infinity, hy = -Infinity;
  for (let k = 0; k < X.length; k++) (lx = Math.min(lx, X[k])), (hx = Math.max(hx, X[k])), (ly = Math.min(ly, Y[k])), (hy = Math.max(hy, Y[k]));
  h = Math.max(h, Math.sqrt(((hx - lx + 4 * h) * (hy - ly + 4 * h)) / MAX_PIXELS));
  lx -= 2 * h; ly -= 2 * h; hx += 2 * h; hy += 2 * h;
  const W = Math.ceil((hx - lx) / h) + 1, H = Math.ceil((hy - ly) / h) + 1;
  const best = new Float64Array(W * H).fill(-Infinity), tri = new Int32Array(W * H).fill(-1), bary = new Float64Array(3 * W * H);
  for (let n = 0; n < ids.length; n++) {
    // pixel centres are at whole coordinates, row 0 on top
    const ax = (X[3 * n] - lx) / h - 0.5, bx = (X[3 * n + 1] - lx) / h - 0.5, qx = (X[3 * n + 2] - lx) / h - 0.5;
    const ay = (hy - Y[3 * n]) / h - 0.5, by = (hy - Y[3 * n + 1]) / h - 0.5, qy = (hy - Y[3 * n + 2]) / h - 0.5;
    const den = (by - qy) * (ax - qx) + (qx - bx) * (ay - qy);
    if (!(Math.abs(den) > 1e-18)) continue;
    const x0 = Math.max(Math.ceil(Math.min(ax, bx, qx)), 0), x1 = Math.min(Math.floor(Math.max(ax, bx, qx)), W - 1);
    const y0 = Math.max(Math.ceil(Math.min(ay, by, qy)), 0), y1 = Math.min(Math.floor(Math.max(ay, by, qy)), H - 1);
    for (let cy = y0; cy <= y1; cy++)
      for (let cx = x0; cx <= x1; cx++) {
        const l0 = ((by - qy) * (cx - qx) + (qx - bx) * (cy - qy)) / den;
        const l1 = ((qy - ay) * (cx - qx) + (ax - qx) * (cy - qy)) / den;
        const l2 = 1 - l0 - l1;
        if (l0 < -1e-9 || l1 < -1e-9 || l2 < -1e-9) continue;
        const depth = l0 * near[3 * n] + l1 * near[3 * n + 1] + l2 * near[3 * n + 2];
        const pix = cy * W + cx;
        if (!(depth > best[pix])) continue;
        best[pix] = depth;
        tri[pix] = n;
        let b0 = l0, b1 = l1, b2 = l2;
        if (eye) {
          // back from the picture to the triangle
          b0 /= W8[3 * n]; b1 /= W8[3 * n + 1]; b2 /= W8[3 * n + 2];
          const sum = b0 + b1 + b2;
          b0 /= sum; b1 /= sum; b2 /= sum;
        }
        bary[3 * pix] = b0; bary[3 * pix + 1] = b1; bary[3 * pix + 2] = b2;
      }
  }
  let any = false;
  for (let k = 0; k < tri.length; k++) if (tri[k] >= 0) (tri[k] = ids[tri[k]]), (any = true);
  return any ? { tri, bary, W, H, h } : null;
}

// separable Gaussian blur of an H x W image with C channels (flat), edges repeated
function blur(img, H, W, C, sigma) {
  const r = Math.max(1, Math.ceil(3 * sigma));
  const kernel = Array.from({ length: 2 * r + 1 }, (_, i) => Math.exp(-0.5 * ((i - r) / sigma) ** 2));
  const sum = kernel.reduce((a, b) => a + b, 0);
  for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;
  let out = img;
  for (const along of [0, 1]) {
    const next = new Float64Array(out.length);
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++)
        for (let k = 0; k < kernel.length; k++) {
          const yy = along === 0 ? Math.min(Math.max(y + k - r, 0), H - 1) : y;
          const xx = along === 1 ? Math.min(Math.max(x + k - r, 0), W - 1) : x;
          const from = (yy * W + xx) * C, to = (y * W + x) * C;
          for (let c = 0; c < C; c++) next[to + c] += kernel[k] * out[from + c];
        }
    out = next;
  }
  return out;
}

const NEIGHBOURS = [[0, 1], [1, 0], [1, 1], [1, -1]];

// calls fn(a, b) for every pixel a and its neighbour b = a + (dy, dx) inside the picture
function forPairs(H, W, dy, dx, fn) {
  for (let y = 0; y + dy < H; y++)
    for (let x = Math.max(-dx, 0); x < W && x + dx < W; x++) fn(y * W + x, (y + dy) * W + x + dx);
}

// do neighbouring pixels a and b show two parts of the model, one covering the other? A step far
// longer than the slope of the surface explains (a grazed surface has long steps too).
function covers(P, facing, h, a, b, dy, dx) {
  const apart = Math.hypot(P[3 * a] - P[3 * b], P[3 * a + 1] - P[3 * b + 1], P[3 * a + 2] - P[3 * b + 2]);
  return [apart, apart > (3 * h * Math.hypot(dx, dy)) / Math.max(Math.min(facing[a], facing[b]), 0.1)];
}

// how fast a field (C channels) changes per length of surface: the largest difference to a
// neighbouring pixel over their distance, where both are in mask and neither covers the other
function change(field, C, P, facing, mask, H, W, h) {
  const out = new Float64Array(H * W);
  for (const [dy, dx] of NEIGHBOURS)
    forPairs(H, W, dy, dx, (a, b) => {
      if (!mask[a] || !mask[b]) return;
      const [apart, cover] = covers(P, facing, h, a, b, dy, dx);
      if (cover) return;
      let d2 = 0;
      for (let c = 0; c < C; c++) d2 += (field[C * a + c] - field[C * b + c]) ** 2;
      const rate = Math.sqrt(d2) / Math.max(apart, 1e-300);
      out[a] = Math.max(out[a], rate);
      out[b] = Math.max(out[b], rate);
    });
  return out;
}

// pixels on the crests of strength (at least lo, connected to a crest of at least hi) and, for each
// pixel, the whole-pixel direction across its crest
function crest(strength, mask, H, W, lo, hi) {
  const s = blur(strength, H, W, 1, 1);
  const at = (y, x, fill) => (y < 0 || y >= H || x < 0 || x >= W ? fill : s[y * W + x]);
  const ox = new Int8Array(H * W), oy = new Int8Array(H * W), weak = new Uint8Array(H * W), strong = new Uint8Array(H * W);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const k = y * W + x, v = s[k];
      const sxx = at(y, x + 1, 0) + at(y, x - 1, 0) - 2 * v;
      const syy = at(y + 1, x, 0) + at(y - 1, x, 0) - 2 * v;
      const sxy = 0.25 * (at(y + 1, x + 1, 0) + at(y - 1, x - 1, 0) - at(y + 1, x - 1, 0) - at(y - 1, x + 1, 0));
      const tr = sxx + syy, det = sxx * syy - sxy * sxy;
      const lam = 0.5 * tr - Math.sqrt(Math.max(0.25 * tr * tr - det, 0)); // the most negative curvature
      let vx = sxy, vy = lam - sxx;
      if (Math.hypot(vx, vy) < 1e-12) (vx = lam - syy), (vy = sxy);
      const n = Math.max(Math.hypot(vx, vy), 1e-300);
      const dx = Math.round(vx / n), dy = Math.round(vy / n);
      ox[k] = dx;
      oy[k] = dy;
      if (!(lam < 0) || !mask[k] || !(v >= lo) || (!dx && !dy)) continue;
      if (v >= at(y + dy, x + dx, -Infinity) && v >= at(y - dy, x - dx, -Infinity)) {
        weak[k] = 1;
        if (v >= hi) strong[k] = 1;
      }
    }
  return { line: hysteresis(weak, strong, H, W), ox, oy };
}

// the weak pixels 8-connected to a strong one
function hysteresis(weak, strong, H, W) {
  const out = new Uint8Array(H * W), todo = [];
  for (let k = 0; k < H * W; k++) if (strong[k]) (out[k] = 1), todo.push(k);
  while (todo.length) {
    const k = todo.pop(), y = (k / W) | 0, x = k % W;
    for (let dy = -1; dy <= 1; dy++)
      for (let dx = -1; dx <= 1; dx++) {
        const yy = y + dy, xx = x + dx;
        if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
        const j = yy * W + xx;
        if (weak[j] && !out[j]) (out[j] = 1), todo.push(j);
      }
  }
  return out;
}

// Zhang-Suen thinning: lines one pixel wide
function thin(src, H, W) {
  const img = Uint8Array.from(src);
  const ring = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]; // north, then clockwise
  const at = (y, x) => (y < 0 || y >= H || x < 0 || x >= W ? 0 : img[y * W + x]);
  for (;;) {
    let changed = false;
    for (const second of [false, true]) {
      const kill = [];
      for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x++) {
          if (!img[y * W + x]) continue;
          const p = ring.map(([dy, dx]) => at(y + dy, x + dx));
          const count = p.reduce((a, b) => a + b, 0);
          let steps = 0;
          for (let i = 0; i < 8; i++) if (!p[i] && p[(i + 1) % 8]) steps++;
          const keep = second ? (p[0] && p[2] && p[6]) || (p[0] && p[4] && p[6]) : (p[0] && p[2] && p[4]) || (p[2] && p[4] && p[6]);
          if (count >= 2 && count <= 6 && steps === 1 && !keep) kill.push(y * W + x);
        }
      for (const k of kill) img[k] = 0;
      if (kill.length) changed = true;
    }
    if (!changed) return img;
  }
}

// the silhouette, relief or colour lines of one view: {rows, cols, pos, tri, bary, color} of their
// pixels, or null
function viewLines(mesh, tris, cn, R, eye, toward, opt, kind) {
  const { tri, bary, H, W, h } = R, n = H * W;
  const fg = Uint8Array.from(tri, (t) => (t >= 0 ? 1 : 0));
  const P = new Float64Array(3 * n), N = new Float64Array(3 * n), nearness = new Float64Array(n), facing = new Float64Array(n);
  const s = normalize(toward);
  for (let k = 0; k < n; k++) {
    if (!fg[k]) continue;
    const t = tri[k], B = [bary[3 * k], bary[3 * k + 1], bary[3 * k + 2]];
    for (let a = 0; a < 3; a++) {
      P[3 * k + a] = B[0] * tris[9 * t + a] + B[1] * tris[9 * t + 3 + a] + B[2] * tris[9 * t + 6 + a];
      N[3 * k + a] = B[0] * cn[9 * t + a] + B[1] * cn[9 * t + 3 + a] + B[2] * cn[9 * t + 6 + a];
    }
    const l = Math.hypot(N[3 * k], N[3 * k + 1], N[3 * k + 2]) || 1e-300;
    for (let a = 0; a < 3; a++) N[3 * k + a] /= l;
    if (!eye) {
      nearness[k] = dot3(P, 3 * k, s, 0);
      facing[k] = dot3(N, 3 * k, s, 0);
    } else {
      const v = [eye[0] - P[3 * k], eye[1] - P[3 * k + 1], eye[2] - P[3 * k + 2]], dist = Math.hypot(...v) || 1e-300;
      nearness[k] = -dist;
      facing[k] = dot3(N, 3 * k, v, 0) / dist;
    }
  }
  const cover = blur(Float64Array.from(fg), H, W, 1, VIEW_SIGMA);
  const cosAngle = Math.cos((opt.facadeAngle * Math.PI) / 180);
  const solid = Uint8Array.from(fg, (f, k) => (f && cover[k] > 0.5 ? 1 : 0)); // the field is measured here
  const inner = Uint8Array.from(solid, (f, k) => (f && facing[k] > cosAngle ? 1 : 0)); // lines kept where not grazing
  const smoothed = (field) => {
    const out = blur(Float64Array.from(field, (v, i) => (fg[(i / 3) | 0] ? v : 0)), H, W, 3, VIEW_SIGMA);
    for (let k = 0; k < n; k++) for (let a = 0; a < 3; a++) out[3 * k + a] /= Math.max(cover[k], 1e-300);
    return out;
  };
  let line, color = null;
  if (kind === "silhouette") {
    // the edge of the smoothed picture: fur and noise smaller than sigma leave it smooth
    const on = (y, x) => y >= 0 && y < H && x >= 0 && x < W && solid[y * W + x];
    line = new Uint8Array(n);
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++)
        if (solid[y * W + x] && !(on(y - 1, x) && on(y + 1, x) && on(y, x - 1) && on(y, x + 1))) line[y * W + x] = 1;
    // where one part of the model covers another, the nearer one draws the line
    for (const [dy, dx] of NEIGHBOURS)
      forPairs(H, W, dy, dx, (a, b) => {
        if (solid[a] && solid[b] && covers(P, facing, h, a, b, dy, dx)[1]) line[nearness[a] > nearness[b] ? a : b] = 1;
      });
    line = thin(line, H, W);
  } else if (kind === "relief") {
    const smooth = smoothed(N);
    for (let k = 0; k < n; k++) {
      const l = Math.hypot(smooth[3 * k], smooth[3 * k + 1], smooth[3 * k + 2]) || 1e-300;
      for (let a = 0; a < 3; a++) smooth[3 * k + a] /= l;
    }
    const bend = change(smooth, 3, P, facing, solid, H, W, h).map((v) => v * 4 * VIEW_SIGMA * h); // radians over ~2 sigma
    const angle = (opt.reliefAngle * Math.PI) / 180;
    line = thin(crest(bend, solid, H, W, 0.5 * angle, angle).line.map((v, k) => v & inner[k]), H, W);
  } else {
    const fgIds = [];
    fg.forEach((f, k) => f && fgIds.push(k));
    const under = surfaceColors(mesh, Int32Array.from(fgIds, (k) => tri[k]), Float64Array.from(fgIds.flatMap((k) => [bary[3 * k], bary[3 * k + 1], bary[3 * k + 2]])));
    const lin = new Float64Array(3 * n).fill(1);
    fgIds.forEach((k, i) => lin.set(under.subarray(3 * i, 3 * i + 3), 3 * k));
    const lab = linearToOklab(lin);
    // a step of colour smoothed over sigma: its steepest slope times sigma * sqrt(2 pi) is the step
    const step = change(smoothed(lab), 3, P, facing, solid, H, W, h).map((v) => v * VIEW_SIGMA * h * Math.sqrt(2 * Math.PI));
    const found = crest(step, solid, H, W, 0.5 * opt.colorContrast, opt.colorContrast);
    line = thin(found.line.map((v, k) => v & inner[k]), H, W);
    // each pixel shows the colour of the side that stands out from the model's mean colour
    const mean = [0, 0, 0];
    let count = 0;
    for (let k = 0; k < n; k++) if (inner[k]) (count++), [0, 1, 2].forEach((a) => (mean[a] += lab[3 * k + a]));
    for (let a = 0; a < 3; a++) mean[a] /= Math.max(count, 1);
    const r = Math.round(2 * VIEW_SIGMA);
    color = [];
    for (let k = 0; k < n; k++) {
      if (!line[k]) continue;
      const y = (k / W) | 0, x = k % W;
      let best = -2, pick = 3 * k;
      for (const sign of [1, -1]) {
        const sy = Math.min(Math.max(y + sign * r * found.oy[k], 0), H - 1), sx = Math.min(Math.max(x + sign * r * found.ox[k], 0), W - 1);
        const j = sy * W + sx;
        const off = fg[j] ? Math.hypot(lab[3 * j] - mean[0], lab[3 * j + 1] - mean[1], lab[3 * j + 2] - mean[2]) : -1;
        if (off > best) (best = off), (pick = 3 * j);
      }
      color.push(lin[pick], lin[pick + 1], lin[pick + 2]);
    }
  }
  const rows = [], cols = [], pos = [], tr = [], br = [];
  for (let k = 0; k < n; k++) {
    if (!line[k]) continue;
    rows.push((k / W) | 0);
    cols.push(k % W);
    pos.push(P[3 * k], P[3 * k + 1], P[3 * k + 2]);
    tr.push(tri[k]);
    br.push(bary[3 * k], bary[3 * k + 1], bary[3 * k + 2]);
  }
  if (!rows.length) return null;
  return { rows, cols, pos: Float64Array.from(pos), tri: Int32Array.from(tr), bary: Float64Array.from(br), color: Float64Array.from(color || new Array(pos.length).fill(NaN)) };
}

// edges {i, j} between the given line pixels: 4-neighbours, and diagonal neighbours with no line
// pixel at the corner between them
function pixelGraph(rows, cols) {
  const at = new Map(rows.map((y, k) => [`${y},${cols[k]}`, k]));
  const has = (y, x) => at.has(`${y},${x}`);
  const i = [], j = [];
  rows.forEach((y, k) => {
    const x = cols[k];
    for (const [dy, dx] of NEIGHBOURS) {
      const other = at.get(`${y + dy},${x + dx}`);
      if (other === undefined) continue;
      if (dy && dx && (has(y, x + dx) || has(y + dy, x))) continue; // an orthogonal path exists
      i.push(k);
      j.push(other);
    }
  });
  return { i, j };
}

// The lines to outline (null when the model has none). `spacing` is the fill spacing: it sets how
// fine the line points are and how small a line may be. `weights` are the fill's triangle weights;
// in facade mode `audience` drops what nobody sees. The lines do not depend on the seed.
export function findLines(mesh, areas, normals, weights, opt, spacing, rand, audience = null) {
  const tris = mesh.triangles;
  const step = spacing * STEP, window = CORNER_WINDOW * spacing;
  const live = Uint8Array.from(areas, (a) => (a > 0 ? 1 : 0));
  const sight = sightOf(opt);
  const smoothCos = Math.cos((opt.creaseAngle * Math.PI) / 180);
  const visible = (p, tri) => Uint8Array.from(tri, (t, i) => (audience.visibility(p[3 * i], p[3 * i + 1], p[3 * i + 2], t) > 0 ? 1 : 0));

  const kinds = []; // segment lines, the most certain first
  const views = opt.outlineSilhouette || opt.outlineRelief || opt.outlineColors;
  let W, E, nextKey, cn;
  if (opt.outlineBorders || opt.outlineCreases || views) {
    W = weld(tris);
    E = edgesOf(W.vid);
    nextKey = W.verts.length / 3 + E.count; // vertices, then edges, then cuts
  }
  if (views) cn = cornerNormals(W.vid, normals, areas, smoothCos);
  if (opt.outlineBorders || opt.outlineCreases) {
    const { border, crease } = featureEdges(tris, normals, live, E, smoothCos, sight);
    for (const [on, he] of [[opt.outlineBorders, border], [opt.outlineCreases, crease]]) {
      if (!on || !he.length) continue;
      let segs;
      ({ segs, nextKey } = cut(edgeSegs(he, E), tris, PIECE * step, nextKey));
      if (audience) segs = takeSegs(segs, visible(segMids(segs, tris), segs.tri));
      kinds.push(segs);
    }
  }

  // a line two sources find (a coloured edge is a crease and a colour border) is drawn once
  const kept = [];
  let dense = [];
  for (let segs of kinds) {
    if (dense.length && segs.tri.length) segs = takeSegs(segs, near(segMids(segs, tris), Float64Array.from(dense), 2 * step).map((v) => 1 - v));
    if (!segs.tri.length) continue;
    kept.push(segs);
    for (const v of segPoints(segs, tris, step).pos) dense.push(v);
  }
  let segs = concatSegs(kept);
  // lines that all fit in a small spot are noise: a nick in the mesh
  if (segs.tri.length) {
    let { p0, p1 } = segEnds(segs, tris);
    segs = takeSegs(segs, wide(segs.k0, segs.k1, p0, p1, MIN_LENGTH * spacing));
    ({ p0, p1 } = segEnds(segs, tris));
    segs = takeSegs(segs, unspurred(segs.k0, segs.k1, p0, p1, window));
  }

  const pts = segPoints(segs, tris, step);
  const pos = pts.pos, tri = pts.tri, bary = pts.bary, color = new Array(pts.tri.length * 3).fill(NaN);
  let chains = [], corners = [];
  const s = segs.tri.length;
  if (s) {
    const { p0, p1 } = segEnds(segs, tris);
    const prefer = Float64Array.from({ length: 2 * s }, (_, h) => {
      const P = h < s ? p0 : p1, e = h % s, v = sight(P[3 * e], P[3 * e + 1], P[3 * e + 2]);
      return dot3(normals, 3 * segs.tri[e], v, 0);
    });
    const walked = chainsOf(segs.k0, segs.k1, prefer);
    for (const { edges, forward, closed, start, end } of walked.chains) {
      const nodes = [], ba = [], bb = [], nodeBary = [];
      edges.forEach((e, k) => {
        const [A, B, bA, bB] = forward[k] ? [p0, p1, segs.b0, segs.b1] : [p1, p0, segs.b1, segs.b0];
        nodes.push(A[3 * e], A[3 * e + 1], A[3 * e + 2]);
        ba.push(bA[3 * e], bA[3 * e + 1], bA[3 * e + 2]);
        bb.push(bB[3 * e], bB[3 * e + 1], bB[3 * e + 2]);
        nodeBary.push(bA[3 * e], bA[3 * e + 1], bA[3 * e + 2]);
        if (k === edges.length - 1) nodes.push(B[3 * e], B[3 * e + 1], B[3 * e + 2]), nodeBary.push(bB[3 * e], bB[3 * e + 1], bB[3 * e + 2]);
      });
      const tr = Int32Array.from(edges, (e) => segs.tri[e]);
      chains.push(new Chain(Float64Array.from(nodes), tr, Float64Array.from(ba), Float64Array.from(bb),
        Int32Array.from([...tr, tr[tr.length - 1]]), Float64Array.from(nodeBary), closed, start, end, 0.5 * window));
    }
    for (const { node, end, strength } of walked.corners) {
      const e = end % s, [P, B] = end < s ? [p0, segs.b0] : [p1, segs.b1];
      corners.push({ node, pos: Array.from(P.subarray(3 * e, 3 * e + 3)), tri: segs.tri[e], bary: Array.from(B.subarray(3 * e, 3 * e + 3)), strength });
    }
  }

  // silhouette, relief and colours, read from views of the model
  let surface = null;
  if (views) {
    const on = [];
    live.forEach((l, t) => l && on.push(t));
    const sub = new Float64Array(on.length * 9);
    on.forEach((t, k) => sub.set(tris.subarray(9 * t, 9 * t + 9), 9 * k));
    surface = new SurfaceProjector(sub, Int32Array.from(on), 0.5 * spacing, 0.5 * spacing);
    let block = 0;
    viewsOf(opt, tris).forEach(({ eye, toward }, index) => {
      const R = render(tris, eye, toward, VIEW_PIXEL * spacing);
      if (!R) return;
      for (const kind of ["silhouette", "relief", "colors"]) {
        const flag = { silhouette: opt.outlineSilhouette, relief: opt.outlineRelief, colors: opt.outlineColors }[kind];
        if (!flag || (kind === "silhouette" && index)) continue; // the outline is the middle viewer's
        const found = viewLines(mesh, tris, cn, R, eye, toward, opt, kind);
        block++;
        if (!found) continue;
        const drawn = near(found.pos, Float64Array.from(dense), 2 * step); // drawn already, from another source or view
        const k0 = [];
        drawn.forEach((d, k) => !d && k0.push(k));
        const rows = k0.map((k) => found.rows[k]), cols = k0.map((k) => found.cols[k]);
        const p = Float64Array.from(k0.flatMap((k) => [found.pos[3 * k], found.pos[3 * k + 1], found.pos[3 * k + 2]]));
        const t = Int32Array.from(k0, (k) => found.tri[k]);
        const b = Float64Array.from(k0.flatMap((k) => [found.bary[3 * k], found.bary[3 * k + 1], found.bary[3 * k + 2]]));
        const c = Float64Array.from(k0.flatMap((k) => [found.color[3 * k], found.color[3 * k + 1], found.color[3 * k + 2]]));
        let { i, j } = pixelGraph(rows, cols);
        const dist = (u, v) => Math.hypot(p[3 * u] - p[3 * v], p[3 * u + 1] - p[3 * v + 1], p[3 * u + 2] - p[3 * v + 2]);
        if (kind !== "silhouette") {
          // neighbours in the picture on two parts of the model, one in front of the other, are two
          // lines; the outline is one line as the audience sees it, wherever its points lie in depth
          const close = i.map((u, e) => dist(u, j[e]) < 8 * R.h);
          i = i.filter((_, e) => close[e]);
          j = j.filter((_, e) => close[e]);
        }
        if (!i.length) continue;
        const m = Float64Array.from(rows.flatMap((y, k) => [cols[k] * R.h, -y * R.h, 0])); // the picture
        const ends = (list) => Float64Array.from(list.flatMap((u) => [m[3 * u], m[3 * u + 1], m[3 * u + 2]]));
        const keepWide = wide(i, j, ends(i), ends(j), MIN_LENGTH * spacing);
        i = i.filter((_, e) => keepWide[e]);
        j = j.filter((_, e) => keepWide[e]);
        if (i.length) {
          const keepShort = unspurred(i, j, ends(i), ends(j), window);
          i = i.filter((_, e) => keepShort[e]);
          j = j.filter((_, e) => keepShort[e]);
        }
        if (!i.length) continue;
        const base = block * VIEW_NODES;
        const walked = chainsOf(i.map((u) => u + base), j.map((u) => u + base), new Float64Array(2 * i.length));
        const sub3 = (A, ks) => Float64Array.from(ks.flatMap((u) => [A[3 * u], A[3 * u + 1], A[3 * u + 2]]));
        for (const { edges, forward, closed, start, end } of walked.chains) {
          const ks = edges.map((e, k) => (forward[k] ? i[e] : j[e]));
          const last = edges.length - 1;
          ks.push(forward[last] ? j[edges[last]] : i[edges[last]]);
          chains.push(new Chain(sub3(m, ks), null, null, null, Int32Array.from(ks, (u) => t[u]), sub3(b, ks), closed, start, end,
            0.5 * window, sub3(p, ks)));
        }
        for (const { node, end, strength } of walked.corners) {
          const e = end % i.length, u = end < i.length ? i[e] : j[e];
          corners.push({ node, pos: Array.from(p.subarray(3 * u, 3 * u + 3)), tri: t[u], bary: Array.from(b.subarray(3 * u, 3 * u + 3)), strength });
        }
        for (const u of new Set([...i, ...j])) {
          for (let a = 0; a < 3; a++) pos.push(p[3 * u + a]), bary.push(b[3 * u + a]), color.push(c[3 * u + a]), dense.push(p[3 * u + a]);
          tri.push(t[u]);
        }
      }
    });
  }
  if (!tri.length || !chains.length) return null;
  const bent = bends(chains, window);
  chains = bent.chains;
  corners = corners.concat(bent.corners).map((c, i) => ({ ...c, i })).sort((x, y) => y.strength - x.strength || x.i - y.i);
  return makeLines(Float64Array.from(pos), Int32Array.from(tri), Float64Array.from(bary), Float64Array.from(color), step, chains, corners, surface);
}

function makeLines(pos, tri, bary, color, step, chains, corners, surface) {
  return {
    pos, tri, bary, color, step, chains, corners, surface,
    length: chains.reduce((sum, c) => sum + c.length, 0),
    // the colour of the line point nearest to each point (NaN: the colour under it)
    colorsAt(points) {
      const { index } = nearest(points, pos, 2 * step);
      return Float64Array.from({ length: points.length }, (_, k) => color[3 * index[(k / 3) | 0] + (k % 3)]);
    },
  };
}

// How many of the `count` drones go on the lines and how far apart, and which fill candidates make
// room for them: {count, spacing, free, fillArea}; `area` is the (density-weighted) area of all candidates.
export function split(lines, candPos, area, count, opt) {
  let free = new Uint8Array(candPos.length / 3).fill(1), fillArea = area, k = 0;
  let dFill = 2 * poissonRadius(area, count);
  for (let round = 0; round < 3; round++) {
    // the gap depends on the fill spacing and that on the gap
    free = near(candPos, lines.pos, GAP * dFill).map((v) => 1 - v);
    let kept = 0;
    for (const f of free) kept += f;
    fillArea = area * Math.max(kept / free.length, 0.05);
    k = lineCount(opt, count, lines.length, fillArea);
    dFill = 2 * poissonRadius(fillArea, count - k);
  }
  const all = new Uint8Array(candPos.length / 3).fill(1);
  if (k < 3) return { count: 0, spacing: 0, free: all, fillArea: area };
  let kept = 0;
  for (const f of free) kept += f;
  if (kept < 2 * (count - k)) return { count: k, spacing: lines.length / k, free: all, fillArea: area }; // lines everywhere
  return { count: k, spacing: lines.length / k, free, fillArea };
}

function lineCount(opt, count, length, fillArea) {
  const k = opt.outlineAmount === "share"
    ? Math.min(opt.outlineShare * count, fit(count, length, fillArea, 1 / MIN_SPACING))
    : Math.min(fit(count, length, fillArea, Math.max(opt.outlineDensity, 1e-3)), MAX_SHARE * count);
  return Math.min(Math.max(Math.round(k), 0), count - 1);
}

// drones k on lines spaced 1 / ratio of the fill spacing 2 r(fillArea, count - k): k = a sqrt(count - k)
function fit(count, length, fillArea, ratio) {
  const a2 = ((ratio * length) / (2 * poissonRadius(fillArea, 1))) ** 2;
  return 0.5 * (Math.sqrt(a2 * a2 + 4 * a2 * count) - a2);
}

// whole numbers close to want (at most one above it rounded down, none where it is 0) adding up to
// total, by largest remainders; less than total when even that is too few
function allocate(want, total) {
  want = want.map((w) => Math.max(w, 0));
  const m = want.map(Math.floor);
  let diff = Math.max(total, 0) - m.reduce((a, b) => a + b, 0);
  if (diff > 0) {
    const up = m.map((_, i) => i).filter((i) => want[i] > 0).sort((x, y) => want[y] - m[y] - (want[x] - m[x]) || x - y);
    for (const i of up.slice(0, diff)) m[i]++;
  }
  while (diff < 0) {
    const have = m.map((_, i) => i).filter((i) => m[i] > 0).sort((x, y) => want[x] - m[x] - (want[y] - m[y]) || x - y);
    if (!have.length) break;
    const drop = have.slice(0, -diff);
    for (const i of drop) m[i]--;
    diff += drop.length;
  }
  return m;
}

// Snaps drones to the nearest line point: the relaxation's projector for an outline.
class LineProjector {
  constructor(lines, maxMove) {
    this.lines = lines;
    this.maxMove = maxMove;
    this.reach = maxMove + 2 * lines.step;
    this.grid = new Grid(lines.pos, this.reach);
  }

  project(p) {
    let best = Infinity, at = -1;
    this.grid.forEachNear(p[0], p[1], p[2], this.reach, (j, d) => {
      if (d < best || (d === best && j < at)) (best = d), (at = j);
    });
    if (at < 0) return null;
    const L = this.lines;
    return { pos: Array.from(L.pos.subarray(3 * at, 3 * at + 3)), tri: L.tri[at], bary: Array.from(L.bary.subarray(3 * at, 3 * at + 3)) };
  }
}

// At most `count` drones on the lines, `spacing` apart: a drone in every corner and evenly spaced
// drones along the lines between, relaxed along the lines where two come close. Fewer when the
// lines do not hold `count` and where lines run closer than LINE_GAP spacings.
export function place(lines, count, spacing, iterations, tris) {
  const corners = lines.corners, chosen = [];
  for (const c of corners) {
    if (chosen.length >= count) break;
    if (chosen.every((k) => Math.hypot(...k.pos.map((v, a) => v - c.pos[a])) >= CORNER_GAP * spacing)) chosen.push(c);
  }
  // how far into a line its first drone stands, in drone spacings: one from a corner drone, a bit
  // more from a corner merged into a stronger one nearby, half where no drone stands
  const lead = new Map();
  for (const c of corners) {
    let gap = Infinity;
    for (const k of chosen) gap = Math.min(gap, Math.hypot(...k.pos.map((v, a) => v - c.pos[a])));
    lead.set(c.node, gap < CORNER_GAP * spacing ? 1 + gap / spacing : 0.5);
  }
  const held = lines.chains.map((c) => (c.closed ? [0, 0] : [lead.get(c.start) ?? 0.5, lead.get(c.end) ?? 0.5]));
  const want = lines.chains.map((c, i) => (c.closed ? c.length / spacing : c.length / spacing - held[i][0] - held[i][1] + 1));
  const many = allocate(want, count - chosen.length);
  const out = { pos: chosen.flatMap((c) => c.pos), tri: chosen.map((c) => c.tri), bary: chosen.flatMap((c) => c.bary) };
  lines.chains.forEach((c, i) => {
    const m = many[i];
    if (m <= 0) return;
    const ell = c.length, [h0, h1] = held[i];
    const u = c.closed ? ell / m : ell / (m - 1 + h0 + h1);
    c.at(Array.from({ length: m }, (_, k) => (c.closed ? k * u : h0 * u + k * u)), tris, lines.surface, out);
  });
  let pos = Float64Array.from(out.pos), tri = Int32Array.from(out.tri), bary = Float64Array.from(out.bary);
  const nc = chosen.length;
  if (iterations > 0 && tri.length > nc) {
    const r = relax(pos.slice(3 * nc), tri.slice(nc), bary.slice(3 * nc), new LineProjector(lines, 0.15 * spacing), spacing,
      iterations, 0.2, null, 1, nc ? pos.slice(0, 3 * nc) : null);
    pos.set(r.points, 3 * nc);
    tri.set(r.triIds, nc);
    bary.set(r.bary, 3 * nc);
  }
  const keep = park(pos, LINE_GAP * spacing); // the corners first
  return {
    points: Float64Array.from(keep.flatMap((i) => Array.from(pos.subarray(3 * i, 3 * i + 3)))),
    triIds: Int32Array.from(keep, (i) => tri[i]),
    bary: Float64Array.from(keep.flatMap((i) => Array.from(bary.subarray(3 * i, 3 * i + 3)))),
  };
}
