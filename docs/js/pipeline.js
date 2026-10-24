// Feature pipeline of the PSL translator, ported from src/data/dataset.py and the
// live-window code in src/app.py. Every constant comes from model/manifest.json
// (written by src/export_web.py), and `python src/check_web.py` compares this
// file against the Python code on real clips.
//
// A frame is a Float32Array of 75 landmarks x 5 values (x, y, z, visibility,
// presence): pose 0-32, left hand 33-53, right hand 54-74, zero-filled when not
// detected — the layout of the training CSVs. Work after resampling is done in
// float64, like the Python code.

export function createPipeline(p) {
  const V = p.n_values;
  const N = p.n_landmarks;
  const FRAME = N * V;
  const [poseA, poseB] = p.pose_slice;
  const [leftA, leftB] = p.left_hand_slice;
  const [rightA, rightB] = p.right_hand_slice;
  const groups = [p.pose_slice, p.left_hand_slice, p.right_hand_slice];
  const kept = p.feature_landmarks;           // 65 landmarks used as features
  const nPose = p.n_pose_features;           // the first 23 kept are pose
  const keptGroup = kept.map((l) => (l < poseB ? 0 : l < leftB ? 1 : 2));
  const nPositions = nPose * 2 + (kept.length - nPose) * 3; // 172: pose x/y, hands x/y/z
  const featureDim = nPositions * 2;                         // + velocity

  // True where any of landmarks [a, b) has a nonzero x/y/z (i.e. was detected).
  function detected(frame, a, b) {
    for (let l = a; l < b; l++) {
      const o = l * V;
      if (frame[o] !== 0 || frame[o + 1] !== 0 || frame[o + 2] !== 0) return true;
    }
    return false;
  }
  const handsDetected = (frame) => detected(frame, leftA, rightB);

  // ---- live window (src/app.py: resample_window, last_sign) ----

  // Frames captured at the camera's variable rate -> the same span at `fps`,
  // ending at the newest frame. Linear interpolation; a landmark group missing
  // on either side takes the nearer frame's values instead.
  function resampleWindow(times, frames, fps = p.fps) {
    const n = times.length;
    const last = times[n - 1];
    const nOut = Math.floor(((last - times[0]) * fps) / 1000) + 1;
    const step = 1000 / fps;
    const out = new Array(nOut);
    for (let g = 0; g < nOut; g++) {
      const t = last - (nOut - 1 - g) * step;
      let right = lowerBound(times, t);
      right = Math.min(Math.max(right, 1), n - 1);
      const left = right - 1;
      const span = Math.max(times[right] - times[left], 1e-6);
      const w = Math.min(Math.max((t - times[left]) / span, 0), 1);
      const a = frames[left];
      const b = frames[right];
      const frame = new Float64Array(FRAME);
      for (let i = 0; i < FRAME; i++) frame[i] = a[i] * (1 - w) + b[i] * w;
      const nearest = w < 0.5 ? a : b;
      for (const [ga, gb] of groups) {
        if (!(detected(a, ga, gb) && detected(b, ga, gb))) {
          for (let i = ga * V; i < gb * V; i++) frame[i] = nearest[i];
        }
      }
      out[g] = frame;
    }
    return out;
  }

  // The most recent hand-active span (gaps up to maxGap frames bridged) with
  // idle_margin frames of lead-in; idle = frames since the hands were last seen.
  function lastSign(seq, maxGap) {
    const hands = [];
    for (let i = 0; i < seq.length; i++) if (handsDetected(seq[i])) hands.push(i);
    if (hands.length === 0) return null;
    let start = hands[0];
    for (let k = hands.length - 2; k >= 0; k--) {
      if (hands[k + 1] - hands[k] - 1 > maxGap) { start = hands[k + 1]; break; }
    }
    return { sign: seq.slice(Math.max(start - p.idle_margin, 0)),
             idle: seq.length - 1 - hands[hands.length - 1] };
  }

  // ---- feature pipeline (src/data/dataset.py: preprocess) ----

  function trimIdle(seq) {
    let first = -1;
    let last = -1;
    for (let i = 0; i < seq.length; i++) {
      if (handsDetected(seq[i])) { if (first < 0) first = i; last = i; }
    }
    if (first < 0) return seq;
    return seq.slice(Math.max(first - p.idle_margin, 0), last + p.idle_margin + 1);
  }

  // Pose centred on the shoulder midpoint, hands on their palm, all / shoulder width.
  function normalize(seq) {
    let sum = 0;
    let count = 0;
    for (const f of seq) {
      if (!detected(f, poseA, poseB)) continue;
      const a = p.left_shoulder * V;
      const b = p.right_shoulder * V;
      sum += Math.hypot(f[a] - f[b], f[a + 1] - f[b + 1], f[a + 2] - f[b + 2]);
      count++;
    }
    let scale = count ? sum / count : 1;
    if (scale < 1e-6) scale = 1;
    return seq.map((f) => {
      const out = Float64Array.from(f);
      const a = p.left_shoulder * V;
      const b = p.right_shoulder * V;
      const pose = [(f[a] + f[b]) / 2, (f[a + 1] + f[b + 1]) / 2, (f[a + 2] + f[b + 2]) / 2];
      const palm = (start) => {
        const c = [0, 0, 0];
        for (const l of p.palm) for (let d = 0; d < 3; d++) c[d] += f[(start + l) * V + d];
        return c.map((v) => v / p.palm.length);
      };
      const origins = [[poseA, poseB, pose], [leftA, leftB, palm(leftA)], [rightA, rightB, palm(rightA)]];
      for (const [ga, gb, origin] of origins) {
        for (let l = ga; l < gb; l++) {
          for (let d = 0; d < 3; d++) out[l * V + d] = (f[l * V + d] - origin[d]) / scale;
        }
      }
      return out;
    });
  }

  // (frames) -> Float64Array(frames * 344): pose x/y + hand x/y/z of the kept
  // landmarks (hands scaled up), then their per-frame velocity.
  function toFeatures(seq) {
    const F = seq.length;
    const out = new Float64Array(F * featureDim);
    const present = seq.map((f) => {
      const g = [false, false, false];
      kept.forEach((l, j) => {
        const o = l * V;
        if (f[o] !== 0 || f[o + 1] !== 0 || f[o + 2] !== 0) g[keptGroup[j]] = true;
      });
      return g;
    });
    for (let t = 0; t < F; t++) {
      const f = seq[t];
      const prev = t > 0 ? seq[t - 1] : null;
      let k = t * featureDim;
      for (const pass of [0, 1]) { // 0: positions, 1: velocity
        kept.forEach((l, j) => {
          const dims = j < nPose ? 2 : 3;
          const g = keptGroup[j];
          for (let d = 0; d < dims; d++) {
            const v = f[l * V + d];
            if (pass === 0) {
              out[k++] = j < nPose ? v : v * p.hand_feature_scale;
            } else {
              const moving = prev && present[t][g] && present[t - 1][g];
              out[k++] = moving ? (v - prev[l * V + d]) * p.velocity_scale : 0;
            }
          }
        });
      }
    }
    return out;
  }

  // Uniformly resample (numpy linspace + round-half-even) or zero-pad to max_frames.
  function fitLength(feats, F) {
    const M = p.max_frames;
    const out = new Float32Array(M * featureDim);
    if (F > M) {
      const step = (F - 1) / (M - 1);
      for (let i = 0; i < M; i++) {
        const src = i === M - 1 ? F - 1 : roundHalfEven(i * step);
        out.set(feats.subarray(src * featureDim, (src + 1) * featureDim), i * featureDim);
      }
    } else {
      out.set(feats.subarray(0, F * featureDim));
    }
    return out;
  }

  function preprocess(seq) {
    const trimmed = trimIdle(seq);
    return fitLength(toFeatures(normalize(trimmed)), trimmed.length);
  }

  return { FRAME, featureDim, detected, handsDetected, resampleWindow, lastSign, trimIdle,
           normalize, toFeatures, fitLength, preprocess };
}

// First index with arr[i] >= x (numpy searchsorted, side="left").
function lowerBound(arr, x) {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < x) lo = mid + 1; else hi = mid;
  }
  return lo;
}

// numpy's round(): halves go to the even neighbour.
function roundHalfEven(x) {
  const f = Math.floor(x);
  const diff = x - f;
  if (diff > 0.5) return f + 1;
  if (diff < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}
