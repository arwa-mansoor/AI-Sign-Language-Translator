// The trained BiLSTM (src/model.py, build_bilstm) in plain JavaScript.
// Weights come from model/weights.bin + manifest.json (src/export_web.py).
//
// Keras semantics reproduced here:
//   - Masking: a frame whose features are all zero (padding) is skipped — the
//     LSTM state carries over it, so padding never affects the result.
//   - LSTM gates in Keras order i, f, c, o; sigmoid gates, tanh cell/output.
//   - Bidirectional, merge "concat": [forward, backward]; the backward
//     direction reads the sequence end to start.
//   - Layer 1 returns the whole sequence, layer 2 only its final states.
//   - Dropout is inactive at inference.

export class BiLSTM {
  constructor(manifest, buffer) {
    const m = manifest.model;
    this.hidden = m.hidden;
    this.layers = m.layers;
    this.featureDim = m.feature_dim;
    this.numClasses = m.num_classes;
    const all = new Float32Array(buffer);
    this.w = {};
    for (const [name, t] of Object.entries(m.tensors)) {
      const size = t.shape.reduce((a, b) => a * b, 1);
      this.w[name] = all.subarray(t.offset, t.offset + size);
    }
  }

  // features: Float32Array(frames * featureDim) -> logits Float64Array(numClasses)
  predict(features) {
    const D = this.featureDim;
    const T = features.length / D;
    const mask = new Array(T);
    for (let t = 0; t < T; t++) {
      let any = false;
      for (let k = t * D; k < (t + 1) * D; k++) if (features[k] !== 0) { any = true; break; }
      mask[t] = any;
    }
    let x = features;
    let dim = D;
    let final = null;
    for (let i = 0; i < this.layers; i++) {
      const lastLayer = i === this.layers - 1;
      const fw = this.runLSTM(x, T, dim, `lstm${i}.forward`, mask, false, !lastLayer);
      const bw = this.runLSTM(x, T, dim, `lstm${i}.backward`, mask, true, !lastLayer);
      const u = this.hidden;
      if (lastLayer) {
        final = new Float64Array(2 * u);
        final.set(fw, 0);
        final.set(bw, u);
      } else {
        const next = new Float64Array(T * 2 * u);
        for (let t = 0; t < T; t++) {
          next.set(fw.subarray(t * u, (t + 1) * u), t * 2 * u);
          next.set(bw.subarray(t * u, (t + 1) * u), t * 2 * u + u);
        }
        x = next;
        dim = 2 * u;
      }
    }
    const K = this.w["dense.kernel"];
    const b = this.w["dense.bias"];
    const C = this.numClasses;
    const logits = Float64Array.from(b);
    for (let k = 0; k < final.length; k++) {
      const v = final[k];
      for (let c = 0; c < C; c++) logits[c] += v * K[k * C + c];
    }
    return logits;
  }

  // One LSTM direction over the unmasked steps. Returns the per-step outputs
  // (zero on masked steps) or, if !returnSequences, the final hidden state.
  runLSTM(x, T, dim, prefix, mask, reverse, returnSequences) {
    const u = this.hidden;
    const G = 4 * u;
    const W = this.w[`${prefix}.kernel`];           // (dim, 4u)
    const U = this.w[`${prefix}.recurrent_kernel`]; // (u, 4u)
    const bias = this.w[`${prefix}.bias`];          // (4u)
    const h = new Float64Array(u);
    const c = new Float64Array(u);
    const z = new Float64Array(G);
    const out = returnSequences ? new Float64Array(T * u) : null;
    for (let s = 0; s < T; s++) {
      const t = reverse ? T - 1 - s : s;
      if (!mask[t]) continue;
      z.set(bias);
      const xo = t * dim;
      for (let k = 0; k < dim; k++) {
        const v = x[xo + k];
        if (v === 0) continue;
        const row = k * G;
        for (let j = 0; j < G; j++) z[j] += v * W[row + j];
      }
      for (let k = 0; k < u; k++) {
        const v = h[k];
        if (v === 0) continue;
        const row = k * G;
        for (let j = 0; j < G; j++) z[j] += v * U[row + j];
      }
      for (let j = 0; j < u; j++) {
        const ig = sigmoid(z[j]);
        const fg = sigmoid(z[u + j]);
        const cg = Math.tanh(z[2 * u + j]);
        const og = sigmoid(z[3 * u + j]);
        c[j] = fg * c[j] + ig * cg;
        h[j] = og * Math.tanh(c[j]);
      }
      if (out) out.set(h, t * u);
    }
    return out ?? h;
  }
}

function sigmoid(v) {
  return 1 / (1 + Math.exp(-v));
}

export function softmax(logits) {
  let max = -Infinity;
  for (const v of logits) max = Math.max(max, v);
  const e = logits.map((v) => Math.exp(v - max));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}
