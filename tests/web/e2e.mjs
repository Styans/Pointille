// Browser check of the web app: load a textured GLB, distribute drones,
// verify their colors follow the texture and that the downloaded GLB keeps
// the original bytes. three.js is served from node_modules instead of the CDN.
//
//   npm install && npm run e2e   (SCREENSHOT=path.png to save a screenshot)
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright";

import { readGlb } from "../../web/io/gltf-append.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const web = join(root, "web");
const threeDir = join(root, "node_modules", "three");
const fixture = join(root, "tests", "fixtures", "quadrant_sphere.glb");
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".glb": "model/gltf-binary", ".wasm": "application/wasm" };

const server = createServer((req, res) => {
  const path = normalize(join(web, decodeURIComponent(new URL(req.url, "http://x").pathname)));
  try {
    const body = readFileSync(path.endsWith("/") || path === web ? join(web, "index.html") : path);
    res.writeHead(200, { "content-type": TYPES[extname(path)] || "text/html" });
    res.end(body);
  } catch {
    res.writeHead(404).end();
  }
}).listen(0);
const port = server.address().port;

const browser = await chromium.launch({ args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader"] });
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 }, acceptDownloads: true });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.route(/cdn\.jsdelivr\.net\/npm\/three@[^/]+\/(.*)$/, (route) => {
    const rel = route.request().url().replace(/^.*three@[^/]+\//, "");
    try {
      route.fulfill({ body: readFileSync(join(threeDir, rel)), contentType: "text/javascript" });
    } catch {
      route.fulfill({ status: 404 });
    }
  });
  await page.goto(`http://localhost:${port}/`);
  await page.selectOption("#color", "point");

  // drones take the texture color under them, for glTF (flipY off) and OBJ (flipY on) textures
  const objDir = join(root, "tests", "fixtures", "obj");
  for (const [files, count] of [[readdirSync(objDir).map((f) => join(objDir, f)), 250], [[fixture], 300]]) {
    await page.fill("#count", String(count));
    await page.setInputFiles("#file", files);
    await page.waitForFunction((n) => window.pointille?.state.result?.count === n && !window.pointille.state.running, count, { timeout: 60000 });
    const check = await page.evaluate(checkColors);
    assert.ok(check.checked > 80 && check.textured, JSON.stringify(check));
    assert.equal(check.wrong, 0, JSON.stringify(check));
    console.log("colors ok:", files.length > 1 ? "OBJ+MTL+PNG" : "GLB", check);
  }

  const [download] = await Promise.all([page.waitForEvent("download"), page.click("#dl-glb")]);
  assert.equal(download.suggestedFilename(), "quadrant_sphere_pointille.glb");
  const out = readFileSync(await download.path());
  const original = readGlb(readFileSync(fixture).buffer.slice(0));
  const appended = readGlb(out.buffer.slice(out.byteOffset, out.byteOffset + out.length));
  assert.deepEqual(appended.bin.subarray(0, original.bin.length), original.bin);
  assert.equal(appended.doc.nodes.at(-1).name, "Pointille_Drones");
  assert.deepEqual(appended.doc.images, original.doc.images);

  // facade from the front (+Z in glTF) keeps drones on the z > 0 half
  await page.click('input[name="area"][value="facade"] + span');
  await page.waitForFunction(() => window.pointille.state.result?.options.facade === true, null, { timeout: 60000 });
  await page.waitForFunction(() => !window.pointille.state.running);
  const minZ = await page.evaluate(() => {
    const r = window.pointille.state.result;
    let m = Infinity;
    for (let i = 0; i < r.count; i++) m = Math.min(m, r.positions[3 * i + 2]);
    return m;
  });
  assert.ok(minZ > 0, `facade min z ${minZ}`);

  if (process.env.SCREENSHOT) await page.screenshot({ path: process.env.SCREENSHOT });
  assert.deepEqual(errors, []);
  console.log("e2e ok");
} finally {
  await browser.close();
  server.close();
}

// runs in the page: compare each drone's color with the fixture's texture quadrant
function checkColors() {
  const r = window.pointille.state.result;
  const Q = { top_left: [255, 0, 0], top_right: [0, 255, 0], bottom_left: [0, 0, 255], bottom_right: [255, 255, 255] };
  const toSrgb = (c) => Math.round((c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055) * 255);
  let checked = 0, wrong = 0;
  for (let i = 0; i < r.count; i++) {
    // fixture model space: sphere centred at y = 2; texture top = +Z, left = +Y side
    const y = r.positions[3 * i + 1] - 2, z = r.positions[3 * i + 2];
    if (Math.min(Math.abs(z), Math.abs(y)) < 0.2) continue;
    const key = (z > 0 ? "top" : "bottom") + "_" + (y > 0 ? "left" : "right");
    const c = [0, 1, 2].map((a) => toSrgb(r.colorsLinear[3 * i + a]));
    checked++;
    if (c.some((v, a) => Math.abs(v - Q[key][a]) > 1)) wrong++;
  }
  return { checked, wrong, textured: window.pointille.state.mesh.materials.some((m) => m.texture) };
}
