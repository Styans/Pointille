// Reading the model's surface color (read-only). UVs use v-up (OpenGL);
// texture rows are stored top row first (as canvas getImageData returns them).

export function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

export function linearToSrgb(c) {
  c = Math.min(Math.max(c, 0), 1);
  return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
}

// OKLab (Ottosson 2020) of linear RGB (flat, 3 per colour): distances follow how different colours
// look; black-white is 1
export function linearToOklab(linear) {
  const out = new Float64Array(linear.length);
  for (let k = 0; k < linear.length; k += 3) {
    const r = Math.max(linear[k], 0), g = Math.max(linear[k + 1], 0), b = Math.max(linear[k + 2], 0);
    const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
    const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
    const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
    out[k] = 0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s;
    out[k + 1] = 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s;
    out[k + 2] = 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s;
  }
  return out;
}

// sRGB brightness below which a colour has no hue left to keep, only noise: it glows an even white
const GLOW_NEAR_BLACK = [0.06, 0.2];
const GLOW_STRETCH = 0.3; // the brightest drone counts as at least this bright: a black model is not stretched to full

// numpy's default percentile (linear interpolation between the closest ranks)
function percentile(values, q) {
  const sorted = Float64Array.from(values).sort();
  const pos = ((sorted.length - 1) * q) / 100;
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, sorted.length - 1);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

// Colours a drone can show: an LED cannot be black, a drone that is off is not seen at night. Every colour
// keeps its hue, its saturation and its place among the drones' brightnesses: the brightest drone glows at
// full, the darkest at `floor` 0..1, near-black an even dim white. `floor` 0 keeps the colours as they are.
export function glow(linear, floor) {
  const n = linear.length / 3;
  if (!(floor > 0) || n === 0) return linear;
  floor = Math.min(floor, 1);
  const s = new Float64Array(linear.length);
  const v = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < 3; a++) s[3 * i + a] = linearToSrgb(linear[3 * i + a]);
    v[i] = Math.max(s[3 * i], s[3 * i + 1], s[3 * i + 2]);
  }
  const top = Math.max(percentile(v, 98), GLOW_STRETCH);
  const [nb0, nb1] = GLOW_NEAR_BLACK;
  const out = new Float64Array(linear.length);
  for (let i = 0; i < n; i++) {
    const hue = Math.min(Math.max((v[i] - nb0) / (nb1 - nb0), 0), 1);
    const level = floor + (1 - floor) * Math.min(Math.max(v[i] / top, 0), 1) * hue;
    for (let a = 0; a < 3; a++) {
      const full = s[3 * i + a] / Math.max(v[i], 1e-6); // the same hue and saturation at full brightness
      out[3 * i + a] = srgbToLinear((hue * full + 1 - hue) * level);
    }
  }
  return out;
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
