// Procedural meshes for the JS core tests (same shapes as tests/helpers.py).

export function uvSphere(radius = 1, rings = 48, segments = 96) {
  const vert = (i, j) => {
    const t = (Math.PI * i) / rings, p = (2 * Math.PI * j) / segments;
    return [Math.sin(t) * Math.cos(p) * radius, Math.sin(t) * Math.sin(p) * radius, Math.cos(t) * radius];
  };
  const tris = [];
  const push = (a, b, c) => {
    const n = cross(sub(b, a), sub(c, a));
    const m = [0, 1, 2].map((k) => (a[k] + b[k] + c[k]) / 3);
    tris.push(...(dot(n, m) < 0 ? [...a, ...c, ...b] : [...a, ...b, ...c]));
  };
  for (let i = 0; i < rings; i++)
    for (let j = 0; j < segments; j++) {
      const a = vert(i, j), b = vert(i + 1, j), c = vert(i + 1, j + 1), d = vert(i, j + 1);
      if (i !== 0) push(a, b, d);
      if (i !== rings - 1) push(b, c, d);
    }
  return { triangles: Float64Array.from(tris) };
}

// Plane in XZ at height y, normal = facing * Y; u follows +X, v follows +Z.
export function gridPlane({ size = [2, 2], res = 8, y = 0, center = [0, 0], facing = -1 } = {}) {
  const tris = [], uvs = [];
  const X = (i) => -size[0] / 2 + (size[0] * i) / res + center[0];
  const Z = (j) => -size[1] / 2 + (size[1] * j) / res + center[1];
  for (let i = 0; i < res; i++)
    for (let j = 0; j < res; j++) {
      const q = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]];
      for (const tri of [[0, 1, 2], [0, 2, 3]]) {
        let corners = tri.map((k) => q[k]);
        const p = corners.map(([a, b]) => [X(a), y, Z(b)]);
        const n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
        if (Math.sign(n[1]) !== facing) corners = corners.reverse();
        for (const [a, b] of corners) {
          tris.push(X(a), y, Z(b));
          uvs.push(a / res, b / res);
        }
      }
    }
  return { triangles: Float64Array.from(tris), uvs: Float64Array.from(uvs) };
}

export const QUADRANTS = {
  top_left: [255, 0, 0],
  top_right: [0, 255, 0],
  bottom_left: [0, 0, 255],
  bottom_right: [255, 255, 255],
};

export function quadrantTexture(size = 64) {
  const data = new Uint8ClampedArray(size * size * 4);
  const h = size / 2;
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const key = (y < h ? "top" : "bottom") + "_" + (x < h ? "left" : "right");
      data.set([...QUADRANTS[key], 255], 4 * (y * size + x));
    }
  return { width: size, height: size, data, srgb: true };
}

export function concat(parts) {
  const tri = parts.reduce((a, p) => a + p.triangles.length, 0);
  const out = new Float64Array(tri);
  let o = 0;
  for (const p of parts) {
    out.set(p.triangles, o);
    o += p.triangles.length;
  }
  return { triangles: out };
}

function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
