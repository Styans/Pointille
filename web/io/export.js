// Plain formats every drone-show and 3D tool can read.

// glTF (Y-up) -> Z-up show frame: (x, y, z) -> (x, -z, y)
export function toZUp(points, up = "y") {
  const out = Float64Array.from(points);
  if (up !== "y") return out;
  for (let i = 0; i < points.length; i += 3) {
    out[i + 1] = -points[i + 2];
    out[i + 2] = points[i + 1];
  }
  return out;
}

const hex = (v) => v.toString(16).toUpperCase().padStart(2, "0");

export function csvText(positions, colors8) {
  const lines = ["id,x,y,z,r,g,b,hex"];
  for (let i = 0; i < positions.length / 3; i++) {
    const [x, y, z] = [0, 1, 2].map((a) => positions[3 * i + a].toFixed(4));
    const [r, g, b] = [0, 1, 2].map((a) => colors8[3 * i + a]);
    lines.push(`${i},${x},${y},${z},${r},${g},${b},#${hex(r)}${hex(g)}${hex(b)}`);
  }
  return lines.join("\n") + "\n";
}

export function plyBytes(positions, colors8) {
  const n = positions.length / 3;
  const header = new TextEncoder().encode(
    "ply\nformat binary_little_endian 1.0\ncomment Pointille drones\n" +
      `element vertex ${n}\nproperty float x\nproperty float y\nproperty float z\n` +
      "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n",
  );
  const out = new Uint8Array(header.length + n * 15);
  out.set(header, 0);
  const view = new DataView(out.buffer, header.length);
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < 3; a++) view.setFloat32(15 * i + 4 * a, positions[3 * i + a], true);
    for (let a = 0; a < 3; a++) view.setUint8(15 * i + 12 + a, colors8[3 * i + a]);
  }
  return out;
}
