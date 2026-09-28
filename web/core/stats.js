// Spacing statistics and safety scaling.
import { nnDistances } from "./spatial.js";

export function spacing(points, guess) {
  if (points.length < 6) return { min: 0, mean: 0 };
  const nn = nnDistances(points, guess);
  let min = Infinity, sum = 0;
  for (const d of nn) {
    if (d < min) min = d;
    sum += d;
  }
  return { min, mean: sum / nn.length };
}

export function scaleFactor(mode, minDistance, size, upAxis, targetMinDistance, targetHeight) {
  if (mode === "none") return 1;
  if (mode === "min_distance") {
    if (!(minDistance > 0)) throw new Error("cannot scale: some drones coincide");
    return targetMinDistance / minDistance;
  }
  if (mode === "height") {
    if (!(size[upAxis] > 0)) throw new Error("cannot scale: formation is flat along the up axis");
    return targetHeight / size[upAxis];
  }
  throw new Error(`unknown scale mode ${mode}`);
}

// bottom centre of the bounding box: scaling keeps the figure on the ground
export function scalePivot(lo, hi, upAxis) {
  const p = [0, 1, 2].map((a) => (lo[a] + hi[a]) / 2);
  p[upAxis] = lo[upAxis];
  return p;
}
