// Node side of src/check_web.py: runs the browser pipeline + model on the
// fixture clips and reports how far they are from the Python results.
// usage: node src/check_web.mjs <fixture.json> <docs dir>
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const [fixturePath, docsDir] = process.argv.slice(2);
const { createPipeline } = await import(pathToFileURL(join(docsDir, "js", "pipeline.js")));
const { BiLSTM } = await import(pathToFileURL(join(docsDir, "js", "model.js")));

const manifest = JSON.parse(readFileSync(join(docsDir, "model", "manifest.json"), "utf8"));
const weights = readFileSync(join(docsDir, "model", manifest.model.weights));
const model = new BiLSTM(manifest, weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.length));
const pipe = createPipeline(manifest.pipeline);
const fixture = JSON.parse(readFileSync(fixturePath, "utf8"));

const maxDiff = (a, b) => {
  if (a.length !== b.length) return Infinity;
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};
const argmax = (a) => a.reduce((best, v, i) => (v > a[best] ? i : best), 0);

const report = { clips: 0, window: 0, features: 0, logits: 0, idle_mismatch: 0, argmax_mismatch: 0,
                 sign_length_mismatch: 0, ms_per_inference: 0 };
let msTotal = 0;
for (const clip of fixture.clips) {
  const frames = clip.frames.map((f) => Float32Array.from(f));
  const window = pipe.resampleWindow(clip.times, frames);
  report.window = Math.max(report.window, maxDiff(window.flatMap((f) => Array.from(f)), clip.window));
  const found = pipe.lastSign(window, manifest.app.sign_gap_frames);
  if (found.idle !== clip.idle) report.idle_mismatch++;
  if (found.sign.length !== clip.sign_length) report.sign_length_mismatch++;
  const feats = pipe.preprocess(found.sign);
  report.features = Math.max(report.features, maxDiff(feats, clip.features));
  const t0 = performance.now();
  const logits = model.predict(feats);
  msTotal += performance.now() - t0;
  report.logits = Math.max(report.logits, maxDiff(logits, clip.logits));
  if (argmax(logits) !== argmax(clip.logits)) report.argmax_mismatch++;
  report.clips++;
}
report.ms_per_inference = msTotal / report.clips;
console.log(JSON.stringify(report));
