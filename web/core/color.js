// Reading the model's surface color (read-only). UVs use v-up (OpenGL);
// texture rows are stored top row first (as canvas getImageData returns them).

export function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

export function linearToSrgb(c) {
  c = Math.min(Math.max(c, 0), 1);
  return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
}

const LUT = Float64Array.from({ length: 256 }, (_, i) => srgbToLinear(i / 255));

export function toSrgb8(linear) {
  const out = new Uint8Array(linear.length);
  for (let i = 0; i < linear.length; i++) out[i] = Math.round(linearToSrgb(linear[i]) * 255);
  return out;
}

// texture: {width, height, data: RGBA bytes, srgb, wrap}
function texel(tex, x, y, out, w) {
  const o = 4 * (y * tex.width + x);
  for (let a = 0; a < 3; a++) {
    const v = tex.data[o + a];
    out[a] += w * (tex.srgb === false ? v / 255 : LUT[v]);
  }
}

// Bilinear lookup -> linear RGB. "repeat" wraps the UV, but the filter
// footprint is clamped to the edge so texture atlases do not bleed.
export function sampleTexture(tex, u, v, out = [0, 0, 0]) {
  const { width: w, height: h } = tex;
  if (tex.wrap !== "clamp") {
    if (u < 0 || u > 1) u -= Math.floor(u);
    if (v < 0 || v > 1) v -= Math.floor(v);
  }
  const x = Math.min(Math.max(u * w - 0.5, 0), w - 1);
  const y = Math.min(Math.max((1 - v) * h - 0.5, 0), h - 1);
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const fx = x - x0, fy = y - y0;
  const x1 = Math.min(x0 + 1, w - 1), y1 = Math.min(y0 + 1, h - 1);
  out[0] = out[1] = out[2] = 0;
  texel(tex, x0, y0, out, (1 - fx) * (1 - fy));
  texel(tex, x1, y0, out, fx * (1 - fy));
  texel(tex, x0, y1, out, (1 - fx) * fy);
  texel(tex, x1, y1, out, fx * fy);
  return out;
}

// Linear RGB (flat, 3 per point) of the surface at barycentric points.
export function surfaceColors(mesh, triIds, bary) {
  const n = triIds.length;
  const out = new Float64Array(n * 3).fill(1);
  const mats = mesh.materials || [];
  const s = [0, 0, 0];
  for (let k = 0; k < n; k++) {
    const t = triIds[k];
    const b0 = bary[3 * k], b1 = bary[3 * k + 1], b2 = bary[3 * k + 2];
    if (mesh.vertexColors) {
      const o = 9 * t, vc = mesh.vertexColors;
      for (let a = 0; a < 3; a++) out[3 * k + a] *= b0 * vc[o + a] + b1 * vc[o + 3 + a] + b2 * vc[o + 6 + a];
    }
    if (!mats.length) continue;
    const mid = mesh.materialIds ? Math.min(Math.max(mesh.materialIds[t], 0), mats.length - 1) : 0;
    const mat = mats[mid];
    const base = mat.baseColor || [1, 1, 1, 1];
    for (let a = 0; a < 3; a++) out[3 * k + a] *= base[a];
    if (mat.texture && mesh.uvs) {
      const o = 6 * t, uv = mesh.uvs;
      const u = b0 * uv[o] + b1 * uv[o + 2] + b2 * uv[o + 4];
      const v = b0 * uv[o + 1] + b1 * uv[o + 3] + b2 * uv[o + 5];
      sampleTexture(mat.texture, u, v, s);
      for (let a = 0; a < 3; a++) out[3 * k + a] *= s[a];
    }
  }
  return out;
}

// mean color of the candidates owned by each drone (its Voronoi cell)
export function averageByOwner(colors, owner, count) {
  const sum = new Float64Array(count * 3);
  const num = new Float64Array(count);
  for (let k = 0; k < owner.length; k++) {
    const o = owner[k];
    num[o]++;
    for (let a = 0; a < 3; a++) sum[3 * o + a] += colors[3 * k + a];
  }
  for (let i = 0; i < count; i++) for (let a = 0; a < 3; a++) sum[3 * i + a] /= Math.max(num[i], 1);
  return sum;
}
