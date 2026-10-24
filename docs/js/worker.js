// Runs the BiLSTM off the main thread, so recognition never stalls the video.
import { BiLSTM, softmax } from "./model.js";

let model = null;

self.onmessage = (event) => {
  const msg = event.data;
  if (msg.type === "init") {
    model = new BiLSTM(msg.manifest, msg.weights);
    self.postMessage({ type: "ready" });
  } else if (msg.type === "predict") {
    const t0 = performance.now();
    const probs = softmax(model.predict(msg.features));
    self.postMessage({ type: "result", probs, ms: performance.now() - t0 });
  }
};
