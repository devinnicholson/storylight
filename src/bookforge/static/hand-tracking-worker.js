/* MediaPipe's WASM loader uses importScripts, so this is a classic worker. */
self.exports = {};
let detector = null;
self.onmessage = async ({data}) => {
  try {
    if (data.type === "init") {
      importScripts("/workbench-assets/mediapipe/vision_bundle.js");
      const files = await self.exports.FilesetResolver.forVisionTasks("/workbench-assets/mediapipe/wasm");
      detector = await self.exports.HandLandmarker.createFromOptions(files, {
        baseOptions: {
          modelAssetPath: "/workbench-assets/mediapipe/hand_landmarker.task",
          delegate: "CPU",
        },
        canvas: new OffscreenCanvas(320, 240),
        runningMode: "VIDEO",
        numHands: 1,
        minHandDetectionConfidence: 0.65,
        minHandPresenceConfidence: 0.65,
        minTrackingConfidence: 0.65,
      });
      self.postMessage({type: "ready"});
    } else if (data.type === "frame") {
      const started = performance.now();
      try {
        const result = detector.detectForVideo(data.bitmap, data.timestamp);
        const tip = result.landmarks[0]?.[8];
        self.postMessage({
          type: "result", id: data.id,
          point: tip ? {x: tip.x, y: tip.y} : null,
          inferenceMs: performance.now() - started,
        });
      } finally {
        data.bitmap.close();
      }
    }
  } catch (error) {
    self.postMessage({type: "error", message: error.message});
  }
};
