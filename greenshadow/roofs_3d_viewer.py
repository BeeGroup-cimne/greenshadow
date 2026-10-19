# -*- coding: utf-8 -*-
"""
Standalone WebGL viewer for ``roof_planes_3d``.

One self-contained HTML file: three.js is loaded from a CDN and the scene
travels inside the page as binary arrays, base64-encoded. That is what makes
it cheaper than a charting library. Point coordinates are quantized to 16
bits per axis (under a centimetre over a few hundred metres), so a return
costs about 7 bytes rather than the ~30 characters a JSON float triple
takes, and the browser decodes it straight into GPU buffers.

Everything interactive runs on the GPU without rebuilding geometry: point
classes are hidden with a uniform, the point size and colour mode are
uniforms, and the frame is only redrawn when something changed.
"""

import base64
import html
import json

import numpy as np

THREE_VERSION = "0.160.0"


def _b64(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def _json_for_script(value):
    """JSON that can sit inside a <script> element without closing it."""
    return json.dumps(value, separators=(",", ":")).replace("</", "<\\/")


def build_viewer_html(
    *,
    title,
    crs,
    origin,
    faces,
    face_positions,
    surface_positions,
    points,
    point_class_index,
    point_classes,
    color_by,
    face_palette,
    tilt_scale,
    tilt_high,
    surface_color,
):
    """
    Assemble the page.

    Parameters are already in local coordinates (metres from ``origin``):
    ``face_positions`` is (T, 3, 3) float32 with each face's triangles
    contiguous and ``faces`` giving their ``start``/``count``;
    ``surface_positions`` likewise for the recessive DSM layer; ``points``
    (P, 3) float64 with ``point_class_index`` (P,) indexing
    ``point_classes``.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)

    if points.shape[0]:
        low = points.min(axis=0)
        span = np.maximum(points.max(axis=0) - low, 1e-6)
        scale = span / 65535.0
        quantized = np.round((points - low) / scale).astype(np.uint16)
    else:
        low = np.zeros(3)
        scale = np.ones(3)
        quantized = np.zeros((0, 3), dtype=np.uint16)

    meta = {
        "title": title,
        "crs": str(crs) if crs is not None else None,
        "origin": [float(v) for v in origin],
        "faces": faces,
        "colorBy": color_by,
        "facePalette": list(face_palette),
        "tiltScale": [[float(stop), color] for stop, color in tilt_scale],
        "tiltHigh": float(tilt_high),
        "surfaceColor": surface_color,
        "points": {
            "count": int(points.shape[0]),
            "offset": [float(v) for v in low],
            "scale": [float(v) for v in scale],
            "classes": point_classes,
        },
        "surfaceTriangles": int(np.asarray(surface_positions).size // 9),
    }

    blobs = {
        "face-pos": _b64(np.asarray(face_positions, dtype=np.float32).reshape(-1)),
        "surface-pos": _b64(np.asarray(surface_positions, dtype=np.float32).reshape(-1)),
        "points-q": _b64(quantized.reshape(-1)),
        "points-cls": _b64(np.asarray(point_class_index, dtype=np.uint8).reshape(-1)),
    }

    blob_tags = "\n".join(
        '<script type="application/octet-stream" id="%s">%s</script>' % (name, data)
        for name, data in blobs.items()
    )

    return (
        _TEMPLATE
        .replace("__TITLE__", html.escape(title))
        .replace("__THREE_VERSION__", THREE_VERSION)
        .replace("__META__", _json_for_script(meta))
        .replace("__BLOBS__", blob_tags)
    )


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #fcfcfb; --fg: #0b0b0b; --muted: #52514e; --panel: rgba(252,252,251,0.94);
    --border: #e2e1dc; --scene: #f4f4f1; --accent: #2a78d6; --row-hover: #efeeea;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #141517; --fg: #ededeb; --muted: #a3a29d; --panel: rgba(28,29,32,0.94);
      --border: #34353a; --scene: #1b1c1f; --accent: #6aa6ec; --row-hover: #26272b;
    }
  }
  * { box-sizing: border-box; }
  [hidden] { display: none !important; }
  body { margin: 0; background: var(--bg); color: var(--fg);
         font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
  .wrap { max-width: 1600px; margin: 0 auto; padding-block: 16px; padding-inline: 16px; }
  h1 { font-size: 18px; margin: 0 0 2px; }
  .note { color: var(--muted); margin: 0 0 12px; max-width: 90ch; }
  #stage { position: relative; height: 78vh; min-height: 420px; border: 1px solid var(--border);
           border-radius: 8px; overflow: hidden; background: var(--scene); }
  #stage-wrap { position: relative; }
  #view { position: absolute; inset: 0; }
  #view canvas { display: block; width: 100%; height: 100%; touch-action: none; }
  #panel { position: absolute; top: 10px; left: 10px; width: 270px; max-height: calc(100% - 20px);
           overflow-y: auto; background: var(--panel); border: 1px solid var(--border);
           border-radius: 8px; padding: 8px 10px; font-size: 13px; backdrop-filter: blur(4px); }
  #panel details { border-bottom: 1px solid var(--border); padding: 4px 0 8px; }
  #panel details:last-of-type { border-bottom: 0; }
  #panel summary { cursor: pointer; font-weight: 600; padding: 4px 0; }
  #panel label { display: flex; align-items: center; gap: 8px; margin: 5px 0; }
  #panel label.grow input[type=range], #panel label.grow select { flex: 1; min-width: 0; }
  #panel output { color: var(--muted); min-width: 3.5em; text-align: right; font-variant-numeric: tabular-nums; }
  #class-list { list-style: none; margin: 4px 0 0; padding: 0; }
  #class-list li label { margin: 3px 0; }
  .swatch { width: 11px; height: 11px; border-radius: 50%; flex: none; }
  .count { margin-left: auto; color: var(--muted); font-variant-numeric: tabular-nums; }
  .buttons { display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap; }
  button { font: inherit; font-size: 12px; padding: 4px 10px; border-radius: 6px; cursor: pointer;
           border: 1px solid var(--border); background: var(--bg); color: var(--fg); }
  button:hover { border-color: var(--accent); }
  .stats { color: var(--muted); font-size: 12px; margin: 8px 0 0; }
  .ramp { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--muted); margin: 4px 0; }
  .ramp .bar { flex: 1; height: 9px; border-radius: 5px; }
  #tooltip { position: absolute; pointer-events: none; background: var(--panel); color: var(--fg);
             border: 1px solid var(--border); border-radius: 6px; padding: 6px 9px; font-size: 12px;
             white-space: nowrap; transform: translate(12px, 12px); }
  #hint { position: absolute; right: 10px; bottom: 8px; color: var(--muted); font-size: 12px;
          background: var(--panel); border-radius: 6px; padding: 3px 8px; }
  #error { position: absolute; inset: 0; display: grid; place-items: center; color: var(--muted); padding: 24px; text-align: center; }
  .table-wrap { overflow-x: auto; margin-top: 16px; }
  table { border-collapse: collapse; font-size: 13px; min-width: 560px; }
  caption { text-align: left; font-weight: 600; padding-bottom: 6px; }
  th, td { padding: 5px 12px; text-align: right; border-bottom: 1px solid var(--border);
           font-variant-numeric: tabular-nums; }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--muted); font-weight: 600; cursor: pointer; user-select: none; }
  tbody tr { cursor: pointer; }
  tbody tr:hover, tbody tr.selected { background: var(--row-hover); }
  tbody tr.selected td:first-child { box-shadow: inset 3px 0 0 var(--accent); }
  @media (max-width: 720px) {
    #stage { height: 62vh; }
    #panel { position: static; width: auto; max-height: none; margin-top: 10px; }
    #hint { display: none; }
  }
</style>
</head>
<body>
<div class="wrap">
  <h1>__TITLE__</h1>
  <p class="note" id="note"></p>
  <div id="stage-wrap">
    <div id="stage">
      <div id="view"></div>
      <div id="tooltip" hidden></div>
      <div id="hint">drag: orbit · right-drag: pan · wheel: zoom · double-click a face: orbit around it</div>
    </div>
    <aside id="panel">
      <details open id="faces-section">
        <summary>Roof faces</summary>
        <label><input type="checkbox" id="faces-on" checked> Show faces</label>
        <label class="grow">Colour
          <select id="face-mode">
            <option value="plane">by plane</option>
            <option value="tilt">by tilt</option>
            <option value="building">by building</option>
          </select>
        </label>
        <div id="tilt-legend" hidden></div>
        <label class="grow">Opacity <input type="range" id="face-opacity" min="0.1" max="1" step="0.05" value="1"><output id="face-opacity-out">1.00</output></label>
        <label><input type="checkbox" id="edges-on" checked> Face outlines</label>
      </details>
      <details open id="points-section" hidden>
        <summary>LiDAR points</summary>
        <label><input type="checkbox" id="points-on" checked> Show points</label>
        <label class="grow">Colour
          <select id="point-mode">
            <option value="0">by class</option>
            <option value="1">by height</option>
          </select>
        </label>
        <label class="grow">Size <input type="range" id="point-size" min="0.03" max="0.8" step="0.01" value="0.15"><output id="point-size-out">0.15 m</output></label>
        <ul id="class-list"></ul>
      </details>
      <details id="surface-section" hidden>
        <summary>Measured surface</summary>
        <label><input type="checkbox" id="surface-on" checked> Show DSM surface</label>
      </details>
      <div class="buttons">
        <button id="reset">Reset view</button>
        <button id="top">Top view</button>
      </div>
      <p class="stats" id="stats"></p>
    </aside>
  </div>
  <div class="table-wrap">
    <table>
      <caption>Detected faces — click a row to fly to it</caption>
      <thead><tr>
        <th data-key="part">Building</th><th data-key="plane">Face</th>
        <th data-key="tilt">Tilt (°)</th><th data-key="azimuth">Azimuth (°)</th>
        <th data-key="area">Area (m²)</th>
      </tr></thead>
      <tbody id="face-rows"></tbody>
    </table>
  </div>
</div>

<script type="application/json" id="meta">__META__</script>
__BLOBS__

<script type="importmap">
{ "imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@__THREE_VERSION__/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@__THREE_VERSION__/examples/jsm/"
} }
</script>
<script>
  // Shown only if the module below never runs (offline, or no WebGL).
  window.__viewerTimer = setTimeout(function () {
    if (window.__viewerStarted) return;
    var stage = document.getElementById('stage');
    var div = document.createElement('div');
    div.id = 'error';
    div.textContent = 'The 3D view could not start. It loads three.js from cdn.jsdelivr.net, so it needs a network connection and a browser with WebGL.';
    stage.appendChild(div);
  }, 8000);
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

window.__viewerStarted = true;

const META = JSON.parse(document.getElementById('meta').textContent);

function blob(id, Type) {
  const text = document.getElementById(id).textContent.trim();
  if (!text) return new Type(0);
  const bin = atob(text);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Type(bytes.buffer);
}

const facePos = blob('face-pos', Float32Array);
const surfacePos = blob('surface-pos', Float32Array);
const pointsQ = blob('points-q', Uint16Array);
const pointsCls = blob('points-cls', Uint8Array);

const $ = (id) => document.getElementById(id);
const fmt = new Intl.NumberFormat();

// ---------------------------------------------------------------- scene
const view = $('view');
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
view.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 50000);
camera.up.set(0, 0, 1);   // map convention: x east, y north, z up

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
controls.screenSpacePanning = true;
controls.maxPolarAngle = Math.PI * 0.98;

const hemi = new THREE.HemisphereLight(0xffffff, 0x6d6b64, 2.2);
hemi.position.set(0, 0, 1);
scene.add(hemi);
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(-0.6, -1.0, 1.6);
scene.add(sun);

let needsRender = true;
const requestRender = () => { needsRender = true; };

function applyTheme() {
  const css = getComputedStyle(document.documentElement);
  scene.background = new THREE.Color(css.getPropertyValue('--scene').trim());
  requestRender();
}
applyTheme();
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);

// ---------------------------------------------------------------- faces
const faces = META.faces;
const nTri = facePos.length / 9;
const palette = META.facePalette.map((h) => new THREE.Color(h));
const tiltStops = META.tiltScale.map(([s, h]) => [s, new THREE.Color(h)]);

function tiltColor(tilt) {
  const t = Math.min(Math.max(tilt / META.tiltHigh, 0), 1);
  for (let k = 0; k < tiltStops.length - 1; k++) {
    const [s0, c0] = tiltStops[k], [s1, c1] = tiltStops[k + 1];
    if (t <= s1) return c0.clone().lerp(c1, s1 > s0 ? (t - s0) / (s1 - s0) : 0);
  }
  return tiltStops[tiltStops.length - 1][1].clone();
}

let faceMesh = null, edgeLines = null, faceColors = null;
const triFace = new Uint32Array(nTri);
faces.forEach((f, i) => { for (let t = f.start; t < f.start + f.count; t++) triFace[t] = i; });

if (nTri > 0) {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(facePos, 3));
  faceColors = new Float32Array(facePos.length);
  geom.setAttribute('color', new THREE.BufferAttribute(faceColors, 3));
  geom.computeVertexNormals();
  geom.computeBoundingSphere();
  const mat = new THREE.MeshLambertMaterial({
    vertexColors: true, side: THREE.DoubleSide,
    polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
  });
  faceMesh = new THREE.Mesh(geom, mat);
  faceMesh.renderOrder = 2;
  scene.add(faceMesh);
  edgeLines = buildOutlines();
  scene.add(edgeLines);
}

// Boundary of each face: triangle edges used once within that face.
function faceEdges(i) {
  const f = faces[i];
  const seen = new Map();
  const key = (a) => `${Math.round(facePos[a] * 1000)},${Math.round(facePos[a + 1] * 1000)},${Math.round(facePos[a + 2] * 1000)}`;
  for (let t = f.start; t < f.start + f.count; t++) {
    for (let e = 0; e < 3; e++) {
      const a = (t * 3 + e) * 3, b = (t * 3 + (e + 1) % 3) * 3;
      const ka = key(a), kb = key(b);
      const k = ka < kb ? ka + '|' + kb : kb + '|' + ka;
      const hit = seen.get(k);
      if (hit) hit.n++; else seen.set(k, { n: 1, a, b });
    }
  }
  const out = [];
  for (const { n, a, b } of seen.values()) {
    if (n !== 1) continue;
    out.push(facePos[a], facePos[a + 1], facePos[a + 2], facePos[b], facePos[b + 1], facePos[b + 2]);
  }
  return out;
}

function buildOutlines() {
  const all = [];
  for (let i = 0; i < faces.length; i++) {
    const e = faceEdges(i);
    for (let k = 0; k < e.length; k++) all.push(e[k]);
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(all, 3));
  const mat = new THREE.LineBasicMaterial({ color: 0x1d1d1b, transparent: true, opacity: 0.55 });
  const lines = new THREE.LineSegments(geom, mat);
  lines.renderOrder = 3;
  return lines;
}

let hovered = -1, selected = -1;

function paintFaces() {
  if (!faceMesh) return;
  const mode = $('face-mode').value;
  for (let i = 0; i < faces.length; i++) {
    const f = faces[i];
    let c = mode === 'tilt' ? tiltColor(f.tilt)
          : mode === 'building' ? palette[f.part % palette.length].clone()
          : palette[f.slot % palette.length].clone();
    if (i === hovered || i === selected) c.lerp(new THREE.Color(0xffffff), 0.4);
    for (let t = f.start; t < f.start + f.count; t++) {
      for (let v = 0; v < 3; v++) {
        const o = (t * 3 + v) * 3;
        faceColors[o] = c.r; faceColors[o + 1] = c.g; faceColors[o + 2] = c.b;
      }
    }
  }
  faceMesh.geometry.attributes.color.needsUpdate = true;
  $('tilt-legend').hidden = mode !== 'tilt';
  requestRender();
}

// Selected face: a bright outline drawn over everything.
const selectionLines = new THREE.LineSegments(
  new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({ color: 0xffc400, depthTest: false })
);
selectionLines.renderOrder = 10;
scene.add(selectionLines);

function setSelectionOutline(i) {
  const pts = i >= 0 ? faceEdges(i) : [];
  selectionLines.geometry.dispose();
  selectionLines.geometry = new THREE.BufferGeometry();
  selectionLines.geometry.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
}

// ---------------------------------------------------------------- measured surface
let surfaceMesh = null;
if (surfacePos.length) {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(surfacePos, 3));
  geom.computeVertexNormals();
  surfaceMesh = new THREE.Mesh(geom, new THREE.MeshLambertMaterial({
    color: new THREE.Color(META.surfaceColor), side: THREE.DoubleSide,
    transparent: true, opacity: 0.55, depthWrite: false,
  }));
  surfaceMesh.renderOrder = 1;
  scene.add(surfaceMesh);
  $('surface-section').hidden = false;
}

// ---------------------------------------------------------------- points
const P = META.points;
let pointCloud = null;
const MAX_CLASSES = 64;
const classVisible = new Array(MAX_CLASSES).fill(1);

if (P.count > 0) {
  const pos = new Float32Array(P.count * 3);
  for (let i = 0, n = P.count * 3; i < n; i += 3) {
    pos[i] = P.offset[0] + pointsQ[i] * P.scale[0];
    pos[i + 1] = P.offset[1] + pointsQ[i + 1] * P.scale[1];
    pos[i + 2] = P.offset[2] + pointsQ[i + 2] * P.scale[2];
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geom.setAttribute('cls', new THREE.BufferAttribute(pointsCls, 1));
  geom.computeBoundingSphere();

  const classColors = new Array(MAX_CLASSES).fill(0).map(() => new THREE.Color(0x999999));
  P.classes.forEach((c, i) => { classColors[i] = new THREE.Color(c.color); });

  const zLow = P.offset[2], zHigh = P.offset[2] + 65535 * P.scale[2];
  const material = new THREE.ShaderMaterial({
    uniforms: {
      palette: { value: classColors },
      visible: { value: classVisible },
      size: { value: 0.15 },
      pixelScale: { value: 1 },
      mode: { value: 0 },
      zMin: { value: zLow }, zMax: { value: zHigh },
      rampLow: { value: new THREE.Color('#2c4f8a') },
      rampMid: { value: new THREE.Color('#e3c86a') },
      rampHigh: { value: new THREE.Color('#b8412f') },
    },
    vertexShader: `
      attribute float cls;
      uniform vec3 palette[${MAX_CLASSES}];
      uniform float visible[${MAX_CLASSES}];
      uniform float size;
      uniform float pixelScale;
      uniform int mode;
      uniform float zMin, zMax;
      uniform vec3 rampLow, rampMid, rampHigh;
      varying vec3 vColor;
      void main() {
        int i = int(cls + 0.5);
        if (visible[i] < 0.5) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mv;
        gl_PointSize = clamp(size * pixelScale / max(-mv.z, 0.001), 1.0, 32.0);
        if (mode == 0) {
          vColor = palette[i];
        } else {
          float t = clamp((position.z - zMin) / max(zMax - zMin, 0.001), 0.0, 1.0);
          vColor = t < 0.5 ? mix(rampLow, rampMid, t * 2.0) : mix(rampMid, rampHigh, (t - 0.5) * 2.0);
        }
      }`,
    fragmentShader: `
      varying vec3 vColor;
      void main() {
        vec2 d = gl_PointCoord - 0.5;
        if (dot(d, d) > 0.25) discard;
        gl_FragColor = vec4(vColor, 1.0);
        #include <colorspace_fragment>
      }`,
  });
  pointCloud = new THREE.Points(geom, material);
  pointCloud.frustumCulled = false;
  scene.add(pointCloud);

  // Height ramp spans the ground-to-rooftop range, not stray high noise.
  const zs = [];
  const step = Math.max(1, Math.floor(P.count / 20000));
  for (let i = 0; i < P.count; i += step) zs.push(pos[i * 3 + 2]);
  zs.sort((a, b) => a - b);
  material.uniforms.zMin.value = zs[Math.floor(zs.length * 0.01)];
  material.uniforms.zMax.value = zs[Math.floor(zs.length * 0.99)];

  const list = $('class-list');
  P.classes.forEach((c, i) => {
    const li = document.createElement('li');
    li.innerHTML = `<label><input type="checkbox" checked data-index="${i}">
      <span class="swatch" style="background:${c.color}"></span>
      <span>${c.code} · ${c.name}</span><span class="count">${fmt.format(c.count)}</span></label>`;
    list.appendChild(li);
  });
  list.addEventListener('change', (e) => {
    const i = +e.target.dataset.index;
    classVisible[i] = e.target.checked ? 1 : 0;
    material.uniforms.visible.value = classVisible;
    requestRender();
  });
  $('points-section').hidden = false;
}

// ---------------------------------------------------------------- framing
const sceneBox = new THREE.Box3();
[faceMesh, surfaceMesh, pointCloud].forEach((o) => { if (o) sceneBox.expandByObject(o); });
const sceneCenter = sceneBox.getCenter(new THREE.Vector3());
const sceneRadius = Math.max(sceneBox.getSize(new THREE.Vector3()).length() / 2, 5);
camera.near = Math.max(sceneRadius / 5000, 0.05);
camera.far = sceneRadius * 50;
camera.updateProjectionMatrix();

let tween = null;
function flyTo(target, position, ms = 650) {
  tween = {
    t0: performance.now(), ms,
    fromTarget: controls.target.clone(), toTarget: target.clone(),
    fromPos: camera.position.clone(), toPos: position.clone(),
  };
  requestRender();
}

function resetView(animate = true) {
  const dir = new THREE.Vector3(-0.55, -1.0, 0.85).normalize();
  const pos = sceneCenter.clone().addScaledVector(dir, sceneRadius * 2.1);
  if (animate) flyTo(sceneCenter, pos);
  else { controls.target.copy(sceneCenter); camera.position.copy(pos); controls.update(); }
}

function topView() {
  const pos = sceneCenter.clone().add(new THREE.Vector3(0, -0.001, sceneRadius * 2.4));
  flyTo(sceneCenter, pos);
}

function faceCentre(i) {
  const f = faces[i];
  const box = new THREE.Box3();
  const v = new THREE.Vector3();
  for (let t = f.start; t < f.start + f.count; t++) {
    for (let k = 0; k < 3; k++) {
      const o = (t * 3 + k) * 3;
      box.expandByPoint(v.set(facePos[o], facePos[o + 1], facePos[o + 2]));
    }
  }
  return { centre: box.getCenter(new THREE.Vector3()), radius: box.getSize(new THREE.Vector3()).length() / 2 };
}

function focusFace(i) {
  const { centre, radius } = faceCentre(i);
  const dir = camera.position.clone().sub(controls.target).normalize();
  if (dir.z < 0.35) { dir.z = 0.8; dir.normalize(); }
  flyTo(centre, centre.clone().addScaledVector(dir, Math.max(radius * 4, 25)));
}

// ---------------------------------------------------------------- picking
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const tooltip = $('tooltip');
let pickQueued = false, lastEvent = null;

function pickFace(event) {
  if (!faceMesh || !faceMesh.visible) return null;
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1,
              -((event.clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObject(faceMesh, false)[0];
  return hit ? { face: triFace[hit.faceIndex], point: hit.point } : null;
}

function describe(i) {
  const f = faces[i];
  return `<strong>Building ${f.part} · face ${f.plane}</strong><br>` +
         `tilt ${f.tilt.toFixed(1)}° · azimuth ${f.azimuth.toFixed(1)}°<br>` +
         `area ${f.area.toFixed(1)} m²`;
}

renderer.domElement.addEventListener('pointermove', (event) => {
  lastEvent = event;
  if (pickQueued || event.buttons) return;
  pickQueued = true;
  requestAnimationFrame(() => {
    pickQueued = false;
    const hit = pickFace(lastEvent);
    const i = hit ? hit.face : -1;
    if (i !== hovered) { hovered = i; paintFaces(); }
    renderer.domElement.style.cursor = i >= 0 ? 'pointer' : '';
    if (i >= 0) {
      const rect = $('stage').getBoundingClientRect();
      tooltip.innerHTML = describe(i);
      tooltip.style.left = `${lastEvent.clientX - rect.left}px`;
      tooltip.style.top = `${lastEvent.clientY - rect.top}px`;
      tooltip.hidden = false;
    } else {
      tooltip.hidden = true;
    }
  });
});
renderer.domElement.addEventListener('pointerleave', () => {
  tooltip.hidden = true;
  if (hovered !== -1) { hovered = -1; paintFaces(); }
});

let downAt = null;
renderer.domElement.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('click', (event) => {
  if (downAt && Math.hypot(event.clientX - downAt[0], event.clientY - downAt[1]) > 4) return; // a drag
  const hit = pickFace(event);
  select(hit ? hit.face : -1, false);
});
renderer.domElement.addEventListener('dblclick', (event) => {
  const hit = pickFace(event);
  if (!hit) return;
  const offset = camera.position.clone().sub(controls.target);
  flyTo(hit.point, hit.point.clone().add(offset), 450);
});
window.addEventListener('keydown', (e) => { if (e.key === 'Escape') select(-1, false); });

// ---------------------------------------------------------------- table
const rows = $('face-rows');
let sortKey = 'part', sortDir = 1;

function renderTable() {
  const order = faces.map((f, i) => i).sort((a, b) => {
    const fa = faces[a], fb = faces[b];
    return (fa[sortKey] - fb[sortKey]) * sortDir || (fa.part - fb.part) || (fa.plane - fb.plane);
  });
  rows.innerHTML = order.map((i) => {
    const f = faces[i];
    return `<tr data-index="${i}"${i === selected ? ' class="selected"' : ''}><td>${f.part}</td><td>${f.plane}</td>` +
           `<td>${f.tilt.toFixed(1)}</td><td>${f.azimuth.toFixed(1)}</td><td>${f.area.toFixed(1)}</td></tr>`;
  }).join('');
}

function select(i, fly) {
  selected = i;
  setSelectionOutline(i);
  paintFaces();
  rows.querySelectorAll('tr.selected').forEach((r) => r.classList.remove('selected'));
  if (i >= 0) {
    const row = rows.querySelector(`tr[data-index="${i}"]`);
    if (row) { row.classList.add('selected'); if (!fly) row.scrollIntoView({ block: 'nearest' }); }
    if (fly) focusFace(i);
  }
  requestRender();
}

rows.addEventListener('click', (e) => {
  const row = e.target.closest('tr');
  if (!row) return;
  select(+row.dataset.index, true);
  $('stage').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
});
document.querySelectorAll('th[data-key]').forEach((th) => th.addEventListener('click', () => {
  const key = th.dataset.key;
  sortDir = key === sortKey ? -sortDir : 1;
  sortKey = key;
  renderTable();
}));

// ---------------------------------------------------------------- controls panel
$('face-mode').value = META.colorBy;
$('face-mode').addEventListener('change', paintFaces);
$('faces-on').addEventListener('change', (e) => {
  if (faceMesh) faceMesh.visible = e.target.checked;
  if (edgeLines) edgeLines.visible = e.target.checked && $('edges-on').checked;
  selectionLines.visible = e.target.checked;
  requestRender();
});
$('edges-on').addEventListener('change', (e) => {
  if (edgeLines) edgeLines.visible = e.target.checked && $('faces-on').checked;
  requestRender();
});
$('face-opacity').addEventListener('input', (e) => {
  const v = +e.target.value;
  $('face-opacity-out').textContent = v.toFixed(2);
  if (faceMesh) {
    faceMesh.material.opacity = v;
    faceMesh.material.transparent = v < 1;
    faceMesh.material.depthWrite = v >= 1;
    faceMesh.material.needsUpdate = true;
  }
  requestRender();
});
$('points-on').addEventListener('change', (e) => { if (pointCloud) pointCloud.visible = e.target.checked; requestRender(); });
$('point-mode').addEventListener('change', (e) => { if (pointCloud) pointCloud.material.uniforms.mode.value = +e.target.value; requestRender(); });
$('point-size').addEventListener('input', (e) => {
  $('point-size-out').textContent = `${(+e.target.value).toFixed(2)} m`;
  if (pointCloud) pointCloud.material.uniforms.size.value = +e.target.value;
  requestRender();
});
$('surface-on').addEventListener('change', (e) => { if (surfaceMesh) surfaceMesh.visible = e.target.checked; requestRender(); });
$('reset').addEventListener('click', () => resetView(true));
$('top').addEventListener('click', topView);

const legend = $('tilt-legend');
legend.className = 'ramp';
legend.innerHTML = `<span>0°</span><span class="bar" style="background:linear-gradient(90deg,${
  META.tiltScale.map(([s, h]) => `${h} ${Math.round(s * 100)}%`).join(',')})"></span><span>${Math.round(META.tiltHigh)}°</span>`;

$('note').textContent = META.points.count
  ? 'Detected roof faces over the LiDAR returns of the whole area, as measured and coloured by class. ' +
    'Hide a class in the panel to look at the roofs’ own returns; a face floating off them, or roof returns with no face, stand out.'
  : 'Detected roof faces, each lifted onto its own plane. Hover a face for its numbers.';
$('stats').textContent = [
  `${fmt.format(faces.length)} faces`,
  `${fmt.format(nTri)} triangles`,
  P.count ? `${fmt.format(P.count)} points` : null,
  META.crs,
].filter(Boolean).join(' · ');

// ---------------------------------------------------------------- loop
function resize() {
  const w = view.clientWidth, h = view.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  requestRender();
}
new ResizeObserver(resize).observe(view);
controls.addEventListener('change', requestRender);

function frame(now) {
  requestAnimationFrame(frame);
  if (tween) {
    const k = Math.min((now - tween.t0) / tween.ms, 1);
    const e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
    controls.target.lerpVectors(tween.fromTarget, tween.toTarget, e);
    camera.position.lerpVectors(tween.fromPos, tween.toPos, e);
    if (k >= 1) tween = null;
    needsRender = true;
  }
  if (controls.update()) needsRender = true;
  if (!needsRender) return;
  if (pointCloud) {
    const size = renderer.getDrawingBufferSize(new THREE.Vector2());
    pointCloud.material.uniforms.pixelScale.value = size.y * 0.5 * camera.projectionMatrix.elements[5];
  }
  renderer.render(scene, camera);
  needsRender = false;
}

paintFaces();
renderTable();
resize();
resetView(false);
requestAnimationFrame(frame);
clearTimeout(window.__viewerTimer);
</script>
</body>
</html>
"""
