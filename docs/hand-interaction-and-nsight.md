# Local hand interaction and Nsight

September 3, 2026. The optional interaction is implemented; physical Jetson camera acceptance
remains open. No cloud service, model routing, power mode, or permanent profiler was added.

## What the interaction does

One fingertip attracts 28 warm fireflies over the existing scene. These are procedural particles,
not new generated content or automatic understanding of objects in the artwork. The effect is
off at startup. Opening its controls does not request a camera, start a worker, or load a model.

In the projector, press **I**, or **H → Hand interaction**. **Pointer preview** demonstrates
the effect without a camera and is explicitly labeled as a simulation. For real tracking:

1. Connect the webcam to the device running the projector browser.
2. Open the projector on `localhost` (or HTTPS), select **Start camera**, and grant video access.
   No microphone access is requested. A laptop's browser uses the laptop camera, not the Jetson's.
3. Select **Align camera**. Click projected markers 1–4 in order in the unmirrored camera preview.
   The preview moves to the center and the normal HUD hides so the corner markers remain visible.
4. Keep the camera fixed and move a fingertip near the projection surface. The four-point
   homography assumes that plane; hovering well above it causes parallax misalignment.
5. **Stop & release camera** terminates the worker and releases every track. Hiding the tab also
   stops tracking; it never restarts the camera without another explicit action.

Projection-profile changes invalidate camera alignment. Camera identity/resolution and surface
points stay in local browser storage. Physically moving the camera requires recalibration even
when its device ID is unchanged. This version does not infer page turns or identify story objects.

## Performance and privacy boundaries

- MediaPipe Hand Landmarker `float16/1`, web package `0.10.32`, CPU delegate in a classic Worker.
  A CPU delegate is the initial choice to avoid GPU contention, not a claim of Jetson optimality.
- Only one transferable 320-pixel-wide bitmap in flight, at most 10 submissions/second; no queue.
- One fingertip result crosses back to the main thread. Frames and coordinates are never sent
  to Bookforge's server or to a cloud API. Local timing diagnostics contain no image/landmark data.
- A pre-rendered glow sprite and half-resolution canvas draw at most 30 times/second.
- Scene planning, blackout, and projector calibration pause submissions and the effect.
- Five consecutive inference results above 150 ms, a 2.5-second worker-response timeout, or two
  five-second degraded cadence windows stop the effect and release the camera.
  Cadence checks cover rAF relative to baseline and, when available, depth-renderer frame counts
  relative to its target. This is a safety heuristic, not proof of physical display smoothness.
- No CDN requests during playback. Package integrity and model SHA-256 are checked at installation.
  Model assets are optional, Git-ignored, and deliberately not in the normal wheel.

Install assets from the repo with:

```sh
.venv/bin/python scripts/install_hand_tracking_assets.py
```

For the installed Jetson wheel, run the same script with:

```sh
--destination /opt/bookforge/.venv/lib/python3.12/site-packages/bookforge/static/mediapipe
```

Install again after a package rebuild if the optional assets are missing. Missing assets produce
a clear error and never trigger a runtime download. The upstream runtime retains its license
headers; see the [MediaPipe repository and license](https://github.com/google-ai-edge/mediapipe)
and [Google's hand-task documentation](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/web_js).

The Jetson installation was verified with wheel SHA-256
`27098647bd0666011b70f41b34a1df9bdda0c689370bc65d3446d9e426682f16`.
The served interaction script matches the source SHA-256
`8c2cda85c3d3831658c061e4caca6063f6c268c46a2fd862b8e671f58020b15f`.
All optional model assets are installed there. The current kiosk was not reloaded during rollout;
reload the projector page to expose the new controls. Tracking remains off until explicitly started.

## Verification completed

The real browser worker detected a hand in all 20 repetitions of Google's official static hand
fixture. Mac warm median was **14.5 ms**, first inference **89.9 ms**. These are neither 20 distinct
accuracy cases nor Jetson measurements. The pointer effect was visually checked in the projector.
Node tests cover opt-in startup, invalid geometry, backpressure, mismatched replies, slow-worker
shutdown, permission-grant races, and hidden-tab camera cleanup. Capture failure tests verify
planner/kiosk restoration after either window fails. Full Python suite: 1,148 passed.

Reproduce the actual-worker test without camera access:

```sh
.venv/bin/python scripts/serve_hand_worker_check.py --fixture /absolute/path/to/right_hands.jpg
```

Visit `http://127.0.0.1:18086/` and run the test. The fixture source and observations are recorded
in `benchmarks/mediapipe-worker-2026-09-03.json`. The test server exposes only its page, fixture,
and static assets, binds to loopback, and has no generation routes.

## Nsight: real Jetson capture completed

`deploy/jetson/profile-planner-nsight.py` runs one unprofiled baseline and one profiled copy of the
accepted TensorRT engine on an isolated loopback port. Each has one warmup and three short,
synthetic requests. It temporarily stops the kiosk and resident planner for memory headroom,
then restores them without changing configuration. It refuses root and an occupied test port.

The completed capture used Nsight Systems **2026.3.1**, `cuda,nvtx,osrt`, graph-level CUDA tracing,
and no CPU sampling. Its watchdog unit had a five-minute runtime limit and an independent restore
hook. The model's existing TensorRT NVTX ranges were captured; no vendor source modifications
were needed. Both planner and kiosk were verified active afterward, and the API was ready.

| Observation | Result |
| --- | --- |
| Unprofiled median, three synthetic requests | 746.8 ms |
| Profiled median, same requests | 815.0 ms |
| Observed instrumentation overhead | 9.1% |
| Response choices | Identical in all three pairs |
| CUDA graph executions | 91 |
| Summed GPU graph duration, entire capture | 2,623.4 ms |

The profile includes startup, warmup, requests, and shutdown. Startup deserialization appears
prominently in NVTX. `cudaStreamSynchronize` leads CUDA API duration, but includes waiting for
GPU work; that time is not automatically waste. Graph-level mode does not expose inner graph
kernels, so the kernel table is not a complete attribution. Nested ranges overlap.

The next useful investigation is a bounded node-level capture of steady-state decode, followed
by a semantic-preserving A/B. Merely turning on CUDA Graphs is not an optimization opportunity:
the accepted engine is already executing them. These synthetic sub-second timings must not be
advertised as production scene-generation latency or improved planner accuracy.

Raw reports stay on the Jetson in `/home/operator/bookforge-nsight-20260903` and locally
under `.bookforge/nsight-20260903`, outside Git, because traces can contain process/environment
metadata. The sanitized summary is `benchmarks/jetson-nsight-2026-09-03.json`.

For a new authorized capture, choose a fresh output directory and run through a bounded user unit:

```sh
systemd-run --user --unit=bookforge-nsight-capture \
  --property=RuntimeMaxSec=300 --property=TimeoutStopSec=100 \
  --property='ExecStopPost=/usr/bin/systemctl --user --no-block start bookforge-tensorrt-planner.service bookforge-kiosk.service' \
  /usr/bin/python3 /absolute/path/to/profile-planner-nsight.py \
  --output /home/operator/bookforge-nsight-NEW-RUN
```

The restore hook assumes both services should be on after the capture. Do not use it during
a reading or an active generation. The script's fixed profiler and engine paths match this
accepted Jetson setup. No root/perf sysctl changes are necessary.

## Remaining hardware gate

No `/dev/video*` webcam was found on the Jetson during this pass. Connect it, grant camera access,
and calibrate once. Then compare hands off/on during actual depth playback and local planning:
retain approximately 30 rendered FPS, verify tracking latency and memory, check camera loss and
blackout, and confirm all services remain healthy. Until then, keep real tracking opt-in.
