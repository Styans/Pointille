// Pointille web: load a model, spread N drones over it, preview and download.
// The loaded model is only read; downloads never re-encode it.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { FBXLoader } from "three/addons/loaders/FBXLoader.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { MTLLoader } from "three/addons/loaders/MTLLoader.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";

import { colorsSrgb8, distribute, scaledPositions, summary } from "./core/api.js";
import { csvText, plyBytes, toZUp } from "./io/export.js";
import { appendDrones, emptyGlb } from "./io/gltf-append.js";

const THREE_CDN = "https://cdn.jsdelivr.net/npm/three@0.186.1/examples/jsm/";
const MAX_TEXTURE = 2048; // colors are averaged anyway; keeps memory in check
const VIEWS = { front: [0, 0, 1], back: [0, 0, -1], left: [-1, 0, 0], right: [1, 0, 0], top: [0, 1, 0] };

const $ = (id) => document.getElementById(id);
const ui = {
  file: $("file"), drop: $("drop"), info: $("model-info"), count: $("count"), countRange: $("count-range"),
  facadeOpts: $("facade-opts"), view: $("view"), angle: $("angle"), angleOut: $("angle-out"), color: $("color"),
  scaleMode: $("scale-mode"), minDist: $("min-dist"), minDistField: $("min-dist-field"), height: $("height"),
  heightField: $("height-field"), seed: $("seed"), relax: $("relax"), oversample: $("oversample"), auto: $("auto"),
  run: $("run"), error: $("error"), stats: $("stats"), downloads: $("downloads"), dlHint: $("dl-hint"),
  empty: $("empty"), busy: $("busy"), busyText: $("busy-text"),
};

// ---------------------------------------------------------------- viewer
const renderer = new THREE.WebGLRenderer({ canvas: $("canvas"), antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07080c);
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
camera.position.set(0, 1.5, 4);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x404050, 2.2));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(3, 5, 4);
scene.add(sun);

const state = { source: null, model: null, drones: null, result: null, mesh: null, running: false, pending: false };

function resize() {
  const el = $("viewport");
  renderer.setSize(el.clientWidth, el.clientHeight, false);
  camera.aspect = el.clientWidth / Math.max(el.clientHeight, 1);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe($("viewport"));
renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});

function frame(object) {
  const box = new THREE.Box3().setFromObject(object);
  const size = box.getSize(new THREE.Vector3()).length() || 1;
  const center = box.getCenter(new THREE.Vector3());
  controls.target.copy(center);
  camera.near = size / 1000;
  camera.far = size * 100;
  camera.position.copy(center).add(new THREE.Vector3(0, size * 0.25, size * 1.1));
  camera.updateProjectionMatrix();
}

const glowSprite = (() => {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.35, "rgba(255,255,255,0.9)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
})();

function showDrones(result) {
  if (state.drones) {
    scene.remove(state.drones);
    state.drones.geometry.dispose();
    state.drones.material.dispose();
  }
  const lift = result.minDistance * 0.03; // keep the glow off the surface
  const pos = new Float32Array(result.count * 3);
  for (let i = 0; i < pos.length; i++) pos[i] = result.positions[i] + result.normals[i] * lift;
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geom.setAttribute("color", new THREE.BufferAttribute(Float32Array.from(result.colorsLinear), 3));
  const mat = new THREE.PointsMaterial({
    size: result.minDistance * 0.55, map: glowSprite, vertexColors: true, transparent: true,
    depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true,
  });
  state.drones = new THREE.Points(geom, mat);
  state.drones.userData.pointille = true;
  state.drones.renderOrder = 1;
  scene.add(state.drones);
  applyVisibility();
}

function applyVisibility() {
  const show = document.querySelector('input[name="show"]:checked').value;
  if (state.model) state.model.visible = show !== "drones";
  if (state.drones) state.drones.visible = show !== "model";
}
document.querySelectorAll('input[name="show"]').forEach((r) => r.addEventListener("change", applyVisibility));

// ---------------------------------------------------------------- loading
function loadingManager(files) {
  const urls = new Map(files.map((f) => [f.name.toLowerCase(), URL.createObjectURL(f)]));
  const manager = new THREE.LoadingManager();
  const own = new Set(urls.values());
  // map every reference (relative path, or a path glued onto a blob: URL by
  // MTL/FBX loaders) to the dropped file with the same name
  manager.setURLModifier((url) => {
    if (own.has(url) || url.startsWith("data:")) return url;
    const name = decodeURIComponent(url.split(/[?#]/)[0].split(/[\\/]/).pop()).toLowerCase();
    return urls.get(name) ?? url;
  });
  // OBJ/FBX textures keep loading after the model itself: count what is in flight
  let pending = 0;
  const waiting = [];
  const start = manager.itemStart, end = manager.itemEnd;
  manager.itemStart = (u) => (pending++, start(u));
  manager.itemEnd = (u) => {
    pending--;
    end(u);
    if (!pending) waiting.splice(0).forEach((r) => r());
  };
  const idle = () => (pending ? new Promise((r) => waiting.push(r)) : Promise.resolve());
  return { manager, urls, idle };
}

async function loadFiles(files) {
  files = Array.from(files);
  const main =
    files.find((f) => /\.(glb|gltf)$/i.test(f.name)) ||
    files.find((f) => /\.fbx$/i.test(f.name)) ||
    files.find((f) => /\.obj$/i.test(f.name));
  if (!main) throw new Error("Нужен файл модели: .glb, .gltf, .obj или .fbx");
  const ext = main.name.split(".").pop().toLowerCase();
  const { manager, urls, idle } = loadingManager(files);
  const url = urls.get(main.name.toLowerCase());
  let root, bytes = null;
  if (ext === "glb" || ext === "gltf") {
    const draco = new DRACOLoader(manager).setDecoderPath(THREE_CDN + "libs/draco/gltf/");
    const loader = new GLTFLoader(manager).setDRACOLoader(draco).setMeshoptDecoder(MeshoptDecoder);
    root = (await loader.loadAsync(url)).scene;
    if (ext === "glb") bytes = await main.arrayBuffer();
  } else if (ext === "fbx") {
    root = await new FBXLoader(manager).loadAsync(url);
  } else {
    const obj = new OBJLoader(manager);
    const mtlFile = files.find((f) => /\.mtl$/i.test(f.name));
    if (mtlFile) {
      const materials = await new MTLLoader(manager).loadAsync(urls.get(mtlFile.name.toLowerCase()));
      materials.preload();
      obj.setMaterials(materials);
    }
    root = await obj.loadAsync(url);
  }
  await nextFrame();
  await idle();
  return { root, name: main.name, stem: main.name.replace(/\.[^.]+$/, ""), ext, bytes };
}

// ---------------------------------------------------------------- scene -> core arrays
const textureCache = new Map();

function readTexture(tex) {
  if (textureCache.has(tex)) return textureCache.get(tex);
  let out = null;
  const img = tex.image;
  if (img && img.width && img.height) {
    const raw = img.data instanceof Uint8Array || img.data instanceof Uint8ClampedArray;
    if (img.data && !(raw && img.data.length >= img.width * img.height * 4)) {
      textureCache.set(tex, null); // float / packed data textures: use the base color only
      return null;
    }
    const scale = raw ? 1 : Math.min(1, MAX_TEXTURE / Math.max(img.width, img.height));
    const w = Math.max(1, Math.round(img.width * scale)), h = Math.max(1, Math.round(img.height * scale));
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (raw) {
      // data rows go to the canvas as-is; the flipY-aware UV mapping handles orientation
      ctx.putImageData(new ImageData(new Uint8ClampedArray(img.data.buffer, img.data.byteOffset, w * h * 4), w, h), 0, 0);
    } else {
      ctx.drawImage(img, 0, 0, w, h);
    }
    out = { width: w, height: h, data: ctx.getImageData(0, 0, w, h).data, srgb: tex.colorSpace !== THREE.LinearSRGBColorSpace };
  }
  textureCache.set(tex, out);
  return out;
}

function extractMesh(root) {
  root.updateMatrixWorld(true);
  const meshes = [];
  root.traverse((o) => o.isMesh && o.geometry?.attributes?.position && meshes.push(o));
  let total = 0;
  for (const m of meshes) total += (m.geometry.index ? m.geometry.index.count : m.geometry.attributes.position.count) / 3;
  const triangles = new Float64Array(total * 9);
  const uvs = new Float64Array(total * 6);
  const materialIds = new Int32Array(total);
  let vertexColors = null;
  const materials = [];
  const matIndex = new Map();
  let t = 0, hasUv = false;
  for (const mesh of meshes) {
    const g = mesh.geometry, e = mesh.matrixWorld.elements;
    const pos = g.attributes.position, index = g.index, color = g.attributes.color;
    const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    const groups = g.groups.length ? g.groups : [{ start: 0, count: index ? index.count : pos.count, materialIndex: 0 }];
    for (const group of groups) {
      const mat = mats[group.materialIndex] || mats[0];
      if (!matIndex.has(mat)) {
        matIndex.set(mat, materials.length);
        const map = mat?.map?.image ? mat.map : null;
        const c = mat?.color || new THREE.Color(1, 1, 1);
        materials.push({ baseColor: [c.r, c.g, c.b, 1], texture: map ? readTexture(map) : null, map, useVc: !!mat?.vertexColors });
      }
      const mid = matIndex.get(mat);
      const info = materials[mid];
      const uvAttr = info.map ? g.attributes[info.map.channel ? `uv${info.map.channel}` : "uv"] : null;
      let uvm = null;
      if (uvAttr) {
        info.map.updateMatrix();
        uvm = info.map.matrix.elements;
        hasUv = true;
      }
      const end = Math.min(group.start + group.count, index ? index.count : pos.count);
      for (let k = group.start; k + 2 < end; k += 3, t++) {
        materialIds[t] = mid;
        for (let c = 0; c < 3; c++) {
          const i = index ? index.getX(k + c) : k + c;
          const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
          const o = 9 * t + 3 * c;
          triangles[o] = e[0] * x + e[4] * y + e[8] * z + e[12];
          triangles[o + 1] = e[1] * x + e[5] * y + e[9] * z + e[13];
          triangles[o + 2] = e[2] * x + e[6] * y + e[10] * z + e[14];
          if (uvm) {
            const u0 = uvAttr.getX(i), v0 = uvAttr.getY(i);
            const u = uvm[0] * u0 + uvm[3] * v0 + uvm[6];
            const v = uvm[1] * u0 + uvm[4] * v0 + uvm[7];
            uvs[6 * t + 2 * c] = u;
            uvs[6 * t + 2 * c + 1] = info.map.flipY ? v : 1 - v; // core UVs are v-up
          }
          if (info.useVc && color) {
            vertexColors ??= new Float64Array(total * 9).fill(1);
            vertexColors[o] = color.getX(i);
            vertexColors[o + 1] = color.getY(i);
            vertexColors[o + 2] = color.getZ(i);
          }
        }
      }
    }
  }
  return {
    triangles: triangles.subarray(0, t * 9),
    uvs: hasUv ? uvs.subarray(0, t * 6) : null,
    materialIds: materialIds.subarray(0, t),
    vertexColors: vertexColors ? vertexColors.subarray(0, t * 9) : null,
    materials: materials.map(({ baseColor, texture }) => ({ baseColor, texture })),
  };
}

// ---------------------------------------------------------------- UI state
const LOG_MAX = Math.log10(20000);
const countFromRange = (v) => Math.max(1, Math.round(10 ** ((v / 1000) * LOG_MAX)));
const rangeFromCount = (n) => Math.round((Math.log10(Math.max(n, 1)) / LOG_MAX) * 1000);

function options() {
  const facade = document.querySelector('input[name="area"]:checked').value === "facade";
  let viewDir = VIEWS[ui.view.value];
  if (ui.view.value === "camera") {
    const d = camera.position.clone().sub(controls.target).normalize();
    viewDir = [d.x, d.y, d.z];
  }
  return {
    count: Math.max(1, Math.floor(+ui.count.value || 1)),
    seed: Math.max(0, Math.floor(+ui.seed.value || 0)),
    relaxIterations: Math.max(0, Math.floor(+ui.relax.value || 0)),
    oversample: Math.max(2, +ui.oversample.value || 8),
    facade,
    viewDir,
    facadeAngle: +ui.angle.value,
    colorMode: ui.color.value,
    scaleMode: ui.scaleMode.value,
    targetMinDistance: +ui.minDist.value || 1.5,
    targetHeight: +ui.height.value || 20,
    upAxis: 1, // three.js / glTF are Y-up
  };
}

function syncControls() {
  const facade = document.querySelector('input[name="area"]:checked').value === "facade";
  ui.facadeOpts.hidden = !facade;
  ui.angleOut.textContent = `${ui.angle.value}°`;
  ui.minDistField.hidden = ui.scaleMode.value !== "min_distance";
  ui.heightField.hidden = ui.scaleMode.value !== "height";
}

const fmt = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : "—");

function showStats(result, ms) {
  const s = summary(result);
  $("s-count").textContent = s.count;
  $("s-min").textContent = `${fmt(s.minDistance, 3)} м`;
  $("s-mean").textContent = `${fmt(s.meanDistance, 3)} м`;
  $("s-scale").textContent = s.scale === 1 ? "×1" : `×${s.scale.toPrecision(4)}`;
  // width x depth x height in the show frame (Z up)
  $("s-size").textContent = `${fmt(s.size[0])} × ${fmt(s.size[2])} × ${fmt(s.size[1])} м`;
  $("s-time").textContent = `Расчёт: ${(ms / 1000).toFixed(2)} с`;
  ui.stats.hidden = false;
  ui.downloads.hidden = false;
  const isGlb = state.source?.ext === "glb";
  $("dl-glb").textContent = isGlb ? "GLB: модель + дроны" : "GLB: дроны";
  ui.dlHint.textContent = isGlb
    ? "В GLB исходная модель сохранена байт в байт, дроны добавлены отдельным объектом Pointille_Drones."
    : "Исходный файл не меняется; GLB/PLY с дронами кладутся в той же системе координат. CSV — в метрах, Z вверх.";
}

function setError(msg) {
  ui.error.hidden = !msg;
  ui.error.textContent = msg || "";
}

const nextFrame = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));

async function run() {
  if (!state.mesh) return;
  if (state.running) {
    state.pending = true;
    return;
  }
  state.running = true;
  ui.busy.hidden = false;
  ui.busyText.textContent = `Распределяю ${ui.count.value} дронов…`;
  await nextFrame();
  try {
    const t0 = performance.now();
    const result = distribute(state.mesh, options());
    state.result = result;
    showDrones(result);
    showStats(result, performance.now() - t0);
    setError("");
  } catch (e) {
    console.error(e);
    setError(translate(e.message));
  } finally {
    state.running = false;
    ui.busy.hidden = true;
    if (state.pending) {
      state.pending = false;
      run();
    }
  }
}

function translate(msg) {
  if (/faces the audience/.test(msg)) return "Ни одна часть модели не повёрнута к зрителям — смените направление или отключите «Фасад».";
  if (/only fits about (\d+)/.test(msg)) return `Видимая поверхность вмещает примерно ${RegExp.$1} дронов — уменьшите количество или отключите «Фасад».`;
  if (/coincide/.test(msg)) return "Нельзя масштабировать: дроны совпадают.";
  return msg;
}

let timer = null;
function schedule() {
  syncControls();
  if (!ui.auto.checked || !state.mesh) return;
  clearTimeout(timer);
  timer = setTimeout(run, 350);
}

ui.countRange.addEventListener("input", () => {
  ui.count.value = countFromRange(+ui.countRange.value);
  schedule();
});
ui.count.addEventListener("input", () => {
  ui.countRange.value = rangeFromCount(+ui.count.value);
  schedule();
});
for (const el of [ui.view, ui.angle, ui.color, ui.scaleMode, ui.minDist, ui.height, ui.seed, ui.relax, ui.oversample]) {
  el.addEventListener("input", schedule);
}
document.querySelectorAll('input[name="area"]').forEach((r) => r.addEventListener("change", schedule));
ui.run.addEventListener("click", run);
controls.addEventListener("end", () => ui.view.value === "camera" && isFacade() && schedule());
const isFacade = () => document.querySelector('input[name="area"]:checked').value === "facade";

async function open(files) {
  ui.busy.hidden = false;
  ui.busyText.textContent = "Загружаю модель…";
  setError("");
  try {
    const src = await loadFiles(files);
    if (state.model) scene.remove(state.model);
    if (state.drones) scene.remove(state.drones);
    state.drones = null;
    state.source = src;
    state.model = src.root;
    scene.add(src.root);
    textureCache.clear();
    state.mesh = extractMesh(src.root);
    const tris = state.mesh.triangles.length / 9;
    if (!tris) throw new Error("В файле нет полигональной геометрии");
    frame(src.root);
    const size = new THREE.Box3().setFromObject(src.root).getSize(new THREE.Vector3());
    const textured = state.mesh.materials.some((m) => m.texture);
    ui.info.innerHTML = `<b>${escapeHtml(src.name)}</b><br>${tris.toLocaleString("ru")} треуг. · ${fmt(size.x)} × ${fmt(size.z)} × ${fmt(size.y)} · ${textured ? "с текстурой" : "без текстуры"}`;
    ui.info.hidden = false;
    ui.empty.hidden = true;
    ui.run.disabled = false;
    ui.busy.hidden = true;
    await run();
  } catch (e) {
    console.error(e);
    ui.busy.hidden = true;
    setError(`Не удалось открыть модель: ${e.message}`);
  }
}

const escapeHtml = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

ui.file.addEventListener("change", () => ui.file.files.length && open(ui.file.files));
for (const target of [ui.drop, $("viewport")]) {
  target.addEventListener("dragover", (e) => {
    e.preventDefault();
    ui.drop.classList.add("over");
  });
  target.addEventListener("dragleave", () => ui.drop.classList.remove("over"));
  target.addEventListener("drop", (e) => {
    e.preventDefault();
    ui.drop.classList.remove("over");
    if (e.dataTransfer.files.length) open(e.dataTransfer.files);
  });
}

// ---------------------------------------------------------------- downloads
function download(data, name, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([data], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

$("dl-glb").addEventListener("click", () => {
  const r = state.result, src = state.source;
  if (!r) return;
  const extras = { ...summary(r), seed: r.options.seed, facade: r.options.facade, color: r.options.colorMode };
  if (src.ext === "glb") {
    const glb = appendDrones(src.bytes, r.positions, r.colorsLinear, { scale: r.scale, pivot: r.pivot, extras });
    download(glb, `${src.stem}_pointille.glb`, "model/gltf-binary");
  } else {
    download(appendDrones(emptyGlb(), scaledPositions(r), r.colorsLinear, { extras }), `${src.stem}_drones.glb`, "model/gltf-binary");
  }
});
$("dl-csv").addEventListener("click", () => {
  const r = state.result;
  if (r) download(csvText(toZUp(scaledPositions(r), "y"), colorsSrgb8(r)), `${state.source.stem}_drones.csv`, "text/csv");
});
$("dl-ply").addEventListener("click", () => {
  const r = state.result;
  if (r) download(plyBytes(scaledPositions(r), colorsSrgb8(r)), `${state.source.stem}_drones.ply`, "application/octet-stream");
});

syncControls();
resize();

// for automated checks
window.pointille = { state, open, run };
