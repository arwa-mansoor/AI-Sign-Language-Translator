// PSL Translator in the browser: webcam -> MediaPipe Holistic -> the same live
// windowing, feature pipeline and BiLSTM as src/app.py -> English + Urdu.
// Everything runs locally; the video never leaves the device.
import {
  FilesetResolver,
  HolisticLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/vision_bundle.mjs";
import { createPipeline } from "./pipeline.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);

const RECOGNISED_SECONDS = 1.5;
const CAPTION_SECONDS = 8;
const TRAIL_FRAMES = 14;
const COLOURS = { pose: "#f0e6f0", left: "#cd50d7", right: "#28cdeb" };

const ui = {
  status: "idle", current: null, confidence: 0, confirmedAt: -1e9,
  history: [], recognised: new Set(), tracking: true,
};
const live = { frames: [], frameCount: 0, stable: 0, lastLabel: "", busy: false, lastTs: -1 };
const trails = { left: [], right: [] };
let manifest, pipe, worker, landmarker, video, ctx;

// ------------------------------------------------------------------ boot ----

boot().catch((err) => fail(`Could not load the AI models (${err.message || err}). Check your connection and reload.`));

async function boot() {
  manifest = await (await fetch("model/manifest.json")).json();
  pipe = createPipeline(manifest.pipeline);
  buildSignList();
  const [ready] = await Promise.all([startWorker(), loadLandmarker()]);
  const start = $("start");
  start.disabled = false;
  start.textContent = "Start camera";
  start.onclick = startCamera;
  return ready;
}

async function startWorker() {
  const weights = await (await fetch(`model/${manifest.model.weights}`)).arrayBuffer();
  worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  await new Promise((resolve, reject) => {
    worker.onmessage = (e) => e.data.type === "ready" && resolve();
    worker.onerror = (e) => reject(new Error(e.message || "model worker failed"));
    worker.postMessage({ type: "init", manifest, weights }, [weights]);
  });
  worker.onmessage = (e) => e.data.type === "result" && onPrediction(e.data.probs);
}

async function loadLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(manifest.mediapipe.wasm);
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: manifest.mediapipe.model, delegate },
    runningMode: "VIDEO",
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minHandLandmarksConfidence: 0.5,
  });
  try {
    landmarker = await HolisticLandmarker.createFromOptions(vision, options("GPU"));
  } catch (err) {
    console.warn("GPU delegate unavailable, using CPU", err);
    landmarker = await HolisticLandmarker.createFromOptions(vision, options("CPU"));
  }
}

async function startCamera() {
  $("start").disabled = true;
  $("start").textContent = "Starting camera…";
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }, audio: false,
    });
    video = document.createElement("video");
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    await video.play();
  } catch (err) {
    fail("Camera access was blocked or no camera was found. Allow the camera and reload.");
    return;
  }
  const canvas = $("view");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx = canvas.getContext("2d");
  $("intro").hidden = true;
  requestAnimationFrame(loop);
}

function fail(message) {
  const el = $("error");
  el.textContent = message;
  el.hidden = false;
  $("start").textContent = "Unavailable";
}

// ------------------------------------------------------------- live loop ----

let lastVideoTime = -1;
function loop() {
  requestAnimationFrame(loop);
  if (video.readyState < 2 || video.currentTime === lastVideoTime) {
    animate();
    return;
  }
  lastVideoTime = video.currentTime;
  const ts = Math.max(performance.now(), live.lastTs + 1); // VIDEO mode: increasing timestamps
  live.lastTs = ts;
  const result = landmarker.detectForVideo(video, ts);

  live.frames.push({ t: ts, f: toFrame(result) });
  while (ts - live.frames[0].t > manifest.app.window_seconds * 1000) live.frames.shift();
  live.frameCount++;
  if (live.frameCount % manifest.app.interval === 0 && live.frames.length >= 2) infer();

  draw(result);
  animate();
}

// One frame in the training CSV layout: pose 0-32, left hand 33-53, right hand
// 54-74, each (x, y, z, visibility, presence) in MediaPipe world coordinates.
function toFrame(result) {
  const frame = new Float32Array(pipe.FRAME);
  const fill = (offset, landmarks, limit) => {
    if (!landmarks) return;
    landmarks.slice(0, limit).forEach((lm, i) => {
      const o = (offset + i) * 5;
      frame[o] = lm.x; frame[o + 1] = lm.y; frame[o + 2] = lm.z;
      frame[o + 3] = lm.visibility ?? 0; frame[o + 4] = lm.presence ?? 0;
    });
  };
  fill(0, result.poseWorldLandmarks?.[0], 33);
  fill(33, result.leftHandWorldLandmarks?.[0], 21);
  fill(54, result.rightHandWorldLandmarks?.[0], 21);
  return frame;
}

// The live app's inference step (src/app.py): resample to 25 fps, take the latest
// hand-active span, and classify it once it is long enough.
function infer() {
  const A = manifest.app;
  const resampled = pipe.resampleWindow(live.frames.map((x) => x.t), live.frames.map((x) => x.f));
  const found = pipe.lastSign(resampled, A.sign_gap_frames);
  if (!found || found.idle > A.hold_seconds * manifest.pipeline.fps) {
    ui.status = "idle";
    live.stable = 0;
    live.lastLabel = "";
    return;
  }
  ui.status = found.idle <= A.sign_gap_frames ? "signing" : "hold";
  if (found.sign.length < A.min_sign_frames || live.busy) return;
  const features = pipe.preprocess(found.sign);
  live.busy = true;
  worker.postMessage({ type: "predict", features }, [features.buffer]);
}

function onPrediction(probs) {
  live.busy = false;
  const threshold = Number(params.get("threshold")) || manifest.app.threshold;
  let best = 0;
  probs.forEach((p, i) => { if (p > probs[best]) best = i; });
  const confidence = probs[best];
  if (confidence < threshold) {
    live.stable = 0;
    live.lastLabel = "";
    return;
  }
  const cls = manifest.classes[best];
  if (cls.label === live.lastLabel) live.stable++;
  else { live.stable = 1; live.lastLabel = cls.label; }
  if (live.stable === manifest.app.stable_confirm) confirmSign(cls, confidence);
  else if (live.stable > manifest.app.stable_confirm && ui.current === cls) setRing(confidence, false);
}

// --------------------------------------------------------------- drawing ----

function draw(result) {
  const W = ctx.canvas.width;
  const H = ctx.canvas.height;
  ctx.save();
  ctx.translate(W, 0);
  ctx.scale(-1, 1); // mirrored selfie view; detection ran on the unmirrored frame
  ctx.drawImage(video, 0, 0, W, H);
  ctx.restore();

  const px = (lm) => [(1 - lm.x) * W, lm.y * H];
  const pose = result.poseLandmarks?.[0];
  const hands = { left: result.leftHandLandmarks?.[0], right: result.rightHandLandmarks?.[0] };
  for (const side of ["left", "right"]) {
    trails[side].push(hands[side] ? px(hands[side][8]) : null); // index fingertip
    if (trails[side].length > TRAIL_FRAMES) trails[side].shift();
  }
  if (!ui.tracking) return;

  const lines = [];
  if (pose) {  // shoulders, arms, pose hand points (the face and legs are not drawn)
    for (const c of HolisticLandmarker.POSE_CONNECTIONS) {
      if (c.start >= 11 && c.end >= 11 && c.start < 23 && c.end < 23) lines.push([px(pose[c.start]), px(pose[c.end]), COLOURS.pose]);
    }
  }
  for (const side of ["left", "right"]) {
    if (!hands[side]) continue;
    for (const c of HolisticLandmarker.HAND_CONNECTIONS) {
      lines.push([px(hands[side][c.start]), px(hands[side][c.end]), COLOURS[side]]);
    }
  }
  ctx.lineCap = "round";
  for (const [glow, width] of [[true, 5], [false, 1.6]]) {  // neon: blurred glow, then a bright core
    ctx.shadowBlur = glow ? 14 : 0;
    ctx.lineWidth = width;
    for (const [a, b, colour] of lines) {
      ctx.shadowColor = colour;
      ctx.strokeStyle = glow ? colour : "rgba(255,255,255,0.85)";
      ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
    }
  }
  ctx.shadowBlur = 0;
  for (const side of ["left", "right"]) {  // fingertip light trails
    const t = trails[side];
    for (let i = 1; i < t.length; i++) {
      if (!t[i - 1] || !t[i] || Math.hypot(t[i][0] - t[i - 1][0], t[i][1] - t[i - 1][1]) > 0.12 * W) continue;
      ctx.globalAlpha = i / t.length;
      ctx.strokeStyle = COLOURS[side];
      ctx.lineWidth = 1 + 4 * (i / t.length);
      ctx.beginPath(); ctx.moveTo(...t[i - 1]); ctx.lineTo(...t[i]); ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
  ctx.fillStyle = "#fff";
  const joints = [...(pose ? pose.slice(11, 23) : []), ...(hands.left || []), ...(hands.right || [])];
  for (const lm of joints) {
    const [x, y] = px(lm);
    ctx.beginPath(); ctx.arc(x, y, 2.2, 0, 2 * Math.PI); ctx.fill();
  }
}

// -------------------------------------------------------------- panel UI ----

function animate() {
  const now = performance.now();
  const since = (now - ui.confirmedAt) / 1000;
  const state = since < RECOGNISED_SECONDS ? "recognised" : ui.status;
  $("stage").dataset.state = state;
  const status = $("status");
  status.dataset.state = state;
  $("status-text").textContent = {
    recognised: "Recognised!", idle: "Ready - raise your hands",
    signing: "Watching your sign…", hold: "Reading the sign…",
  }[state];

  const showing = ui.current && since < CAPTION_SECONDS && !(ui.status === "signing" && since > RECOGNISED_SECONDS);
  $("result").hidden = !showing;
  $("placeholder").hidden = !!showing;
  if (!showing) $("ph-main").textContent = ui.status === "idle" ? "Sign a word" : "Watching…";

  const f = live.frames;
  if (f.length > 1) {
    const span = f[f.length - 1].t - f[0].t;
    $("fps").textContent = span > 0 ? `${Math.round(((f.length - 1) * 1000) / span)} fps` : "";
  }
  stepConfetti(now);
}

function confirmSign(cls, confidence) {
  ui.current = cls;
  ui.confirmedAt = performance.now();
  ui.history.push(cls);
  if (ui.history.length > 6) ui.history.shift();
  ui.recognised.add(cls.label);

  $("word-en").textContent = cls.en;
  $("word-ur").textContent = cls.ur;
  for (const id of ["word-en", "word-ur"]) restartAnimation($(id), "pop");
  restartAnimation($("card"), "glow");
  restartAnimation($("flash"), "on");
  setRing(confidence, true);
  renderChips();
  updateSignList();
  burstConfetti();
  speak(cls);
}

function restartAnimation(el, cls) {
  el.classList.remove(cls);
  void el.offsetWidth; // reflow so the animation plays again
  el.classList.add(cls);
}

function setRing(confidence, animateFromZero) {
  const ring = $("ring");
  const circumference = 2 * Math.PI * 25;
  ring.classList.toggle("low", confidence < 0.7);
  if (animateFromZero) {
    ring.style.transition = "none";
    ring.style.strokeDashoffset = circumference;
    void ring.getBoundingClientRect();
    ring.style.transition = "";
  }
  ring.style.strokeDashoffset = circumference * (1 - confidence);
  const target = Math.round(confidence * 100);
  const start = performance.now();
  const from = animateFromZero ? 0 : Number($("ring-value").textContent) || 0;
  const tick = (now) => {
    const t = Math.min(Math.max((now - start) / 600, 0), 1); // rAF time can precede `start`
    $("ring-value").textContent = Math.round(from + (target - from) * (1 - (1 - t) ** 3));
    if (t < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function renderChips() {
  const chips = $("chips");
  chips.replaceChildren(...ui.history.map((cls, i) => {
    const el = document.createElement("span");
    el.className = i === ui.history.length - 1 ? "chip new" : "chip";
    el.textContent = cls.en;
    return el;
  }));
  // keep the newest visible: drop the oldest chips that overflow
  while (chips.scrollWidth > chips.clientWidth && chips.children.length > 1) chips.firstChild.remove();
}

function buildSignList() {
  const list = $("sign-list");
  const n = manifest.classes.length;
  if (n > 30) { $("signs").classList.add("hidden"); return; }
  const rows = Math.ceil(n / 2);
  const ordered = [];
  for (let r = 0; r < rows; r++) for (const c of [r, r + rows]) if (c < n) ordered.push(manifest.classes[c]);
  list.replaceChildren(...ordered.map((cls) => {
    const li = document.createElement("li");
    li.dataset.label = cls.label;
    li.innerHTML = `<span class="mark">•</span><span class="en"></span><span class="ur" lang="ur" dir="rtl"></span>`;
    li.querySelector(".en").textContent = cls.en;
    li.querySelector(".ur").textContent = cls.ur;
    return li;
  }));
  updateSignList();
}

function updateSignList() {
  const n = manifest.classes.length;
  const done = manifest.classes.filter((c) => ui.recognised.has(c.label)).length;
  for (const li of $("sign-list").children) {
    const isDone = ui.recognised.has(li.dataset.label);
    li.classList.toggle("done", isDone);
    li.classList.toggle("current", ui.current?.label === li.dataset.label);
    li.querySelector(".mark").textContent = isDone ? "✓" : "•";
  }
  $("count").textContent = `${done} / ${n}`;
  $("count").classList.toggle("some", done > 0);
  $("progress").style.width = `${(100 * done) / n}%`;
}

// -------------------------------------------------------------- confetti ----

const particles = [];
function burstConfetti() {
  const canvas = $("confetti");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const inset = 60 * dpr; // the canvas extends 60px beyond the card
  const [x0, y0, x1, y1] = [inset, inset, canvas.width - inset, canvas.height - inset];
  const colours = ["#28cdeb", "#cd50d7", "#5ae17d", "#ffd46e", "#ffffff"];
  const now = performance.now();
  for (let i = 0; i < 70; i++) {
    const edge = Math.floor(Math.random() * 4);
    const u = Math.random();
    const [x, y, nx, ny] = [[x0 + u * (x1 - x0), y0, 0, -1], [x0 + u * (x1 - x0), y1, 0, 1],
                            [x0, y0 + u * (y1 - y0), -1, 0], [x1, y0 + u * (y1 - y0), 1, 0]][edge];
    const speed = (60 + Math.random() * 200) * dpr;
    const spread = Math.random() * 1.2 - 0.6;
    particles.push({ x, y, vx: (nx + spread * ny) * speed, vy: (ny + spread * nx) * speed, born: now,
                     colour: colours[i % colours.length], r: (2 + Math.random() * 2.5) * dpr });
  }
}

let lastConfetti = performance.now();
function stepConfetti(now) {
  const canvas = $("confetti");
  const c = canvas.getContext("2d");
  const dt = Math.min((now - lastConfetti) / 1000, 0.1);
  lastConfetti = now;
  c.clearRect(0, 0, canvas.width, canvas.height);
  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    const age = (now - p.born) / 1100;
    if (age >= 1) { particles.splice(i, 1); continue; }
    p.vx *= 0.97;
    p.vy += 140 * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    c.globalAlpha = 1 - age;
    c.fillStyle = p.colour;
    c.beginPath(); c.arc(p.x, p.y, p.r, 0, 2 * Math.PI); c.fill();
  }
  c.globalAlpha = 1;
}

// ---------------------------------------------------------------- speech ----

const speech = { en: true, ur: true, busy: false, pending: null };

// English with the browser's voice, then the Urdu clip; a sign confirmed while
// speaking replaces any waiting one, so the voice never lags behind the screen.
function speak(cls) {
  if (speech.busy) { speech.pending = cls; return; }
  speech.busy = true;
  sayEnglish(cls, () => playUrdu(cls, () => {
    speech.busy = false;
    if (speech.pending) { const next = speech.pending; speech.pending = null; speak(next); }
  }));
}

function once(fn, timeoutMs) {
  let done = false;
  const call = () => { if (!done) { done = true; fn(); } };
  setTimeout(call, timeoutMs); // some browsers never fire onend
  return call;
}

function sayEnglish(cls, next) {
  if (!speech.en || !window.speechSynthesis) return next();
  const done = once(next, 4000);
  const u = new SpeechSynthesisUtterance(cls.en);
  u.lang = "en-US";
  u.rate = 1.05;
  u.onend = done;
  u.onerror = done;
  speechSynthesis.speak(u);
}

function playUrdu(cls, next) {
  if (!speech.ur || !cls.audio) return next();
  const done = once(next, 6000);
  const audio = new Audio(cls.audio);
  audio.onended = done;
  audio.onerror = done;
  audio.play().catch(done);
}

// -------------------------------------------------------------- controls ----

function toggle(id, on) { $(id).classList.toggle("on", on); }

$("speak-en").onclick = () => toggle("speak-en", (speech.en = !speech.en));
$("speak-ur").onclick = () => toggle("speak-ur", (speech.ur = !speech.ur));
$("btn-skeleton").onclick = () => toggle("btn-skeleton", (ui.tracking = !ui.tracking));
$("btn-full").onclick = () => (document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen());
$("btn-clear").onclick = () => {
  ui.history = [];
  ui.recognised.clear();
  ui.current = null;
  renderChips();
  updateSignList();
};
document.addEventListener("fullscreenchange", () => toggle("btn-full", !!document.fullscreenElement));
document.addEventListener("keydown", (e) => {
  const key = e.key.toLowerCase();
  if (key === "s") $("btn-skeleton").click();
  else if (key === "f") $("btn-full").click();
  else if (key === "c") $("btn-clear").click();
  else if (key === "d") $("signs").classList.toggle("hidden");
});
