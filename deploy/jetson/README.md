# Storylight on Jetson Orin Nano

This directory installs the Storylight API and projector on an NVIDIA Jetson Orin Nano. The validated target is JetPack 7.2.1 / Jetson Linux 39.2.1. Check NVIDIA's current JetPack documentation before changing the board-support package; none of these scripts flashes or upgrades the device.

## Supported topology

The Jetson owns the private API, scene validation, asset cache, reader state, and attached projector. It can use a local Gemma planner and a remote image renderer. A phone controller is optional and should be exposed only on an operator-controlled private network.

```text
phone :8081 → paired controller gateway → API 127.0.0.1:8080
                                             ├─ local planner 127.0.0.1:11434/11435
                                             └─ authenticated image provider

projector browser ────────────────────────→ API 127.0.0.1:8080
```

The API, model endpoints, microphone routes, diagnostics, and projector remain on loopback. Only the allowlisted controller gateway may bind to the private LAN. Pairing is access control, not transport encryption; do not expose it through router port forwarding or public Wi-Fi.

## Inspect before installing

Run the read-only device report:

```bash
./deploy/jetson/check-device.sh --strict
```

The check reports L4T, CUDA, TensorRT, Docker, Python, power mode, storage, media devices, display state, browser availability, and thermal zones. It does not change the machine.

## Install the runtime

The safe bootstrap is diagnostic-only:

```bash
./deploy/jetson/bootstrap.sh
```

Installation is explicit:

```bash
./deploy/jetson/bootstrap.sh --install-system-packages
./deploy/jetson/bootstrap.sh --create-venv --install-app
```

The first command installs the small Ubuntu support set used by Storylight. It does not install or upgrade JetPack. The second creates a virtual environment with access to JetPack's system Python packages and installs a built Storylight package.

## Verify the application interactively

Start with fake inference and ASR disabled:

```bash
cp deploy/jetson/storylight.env.example .env
set -a
source .env
set +a
.venv/bin/python -m uvicorn storylight.api:app \
  --host 127.0.0.1 --port 8080 --timeout-graceful-shutdown 3
```

Then verify the local surfaces:

```bash
curl -fsS http://127.0.0.1:8080/healthz
curl -fsS http://127.0.0.1:8080/readyz
curl -fsS http://127.0.0.1:8080/v1/runtime:status
```

Open `http://127.0.0.1:8080/projector` on the attached display. Remove the checkout `.env` before installing the hardened system service.

## Install the appliance services

Stage the repository at `/opt/storylight`, then run:

```bash
cd /opt/storylight
./deploy/jetson/bootstrap.sh --create-venv --install-modal-runtime
sudo ./deploy/jetson/install-standalone.sh --user "$USER"
```

The installer refuses non-Jetson hosts and checkouts outside `/opt/storylight`. It creates root-only environment files and installs the API, controller, local model, and projector units. It does not change JetPack, Wi-Fi credentials, display login, or power mode.

Configure `/etc/storylight/storylight.env` from `storylight.standalone.env.example`. Keep cloud credentials in that root-owned file; never place them in browser or phone configuration.

For a portable private network, review and run:

```bash
sudo /opt/storylight/deploy/jetson/configure-portable-network.sh
sudo /opt/storylight/deploy/jetson/show-controller-pairing.sh
```

Verify that only the controller gateway is exposed:

```bash
curl -fsS http://127.0.0.1:8080/readyz
curl -fsS http://127.0.0.1:8081/healthz
ss -ltn | grep -E '127\.0\.0\.1:8080|0\.0\.0\.0:8081|127\.0\.0\.1:1143[45]'
```

If the network is not private, stop the controller service. Local projector operation remains available.

## Local speech recognition

Storylight supports a lazy Whisper TensorRT backend through the portable `AsrBackend` contract. Leave `STORYLIGHT_ASR_BACKEND=disabled` during first boot. Install JetPack-compatible dependencies from NVIDIA guidance, select `whisper_trt` in the service environment, and warm the engine before a live reading:

```bash
sudo systemctl stop "storylight@${USER}.service"
sudo -u "$USER" /opt/storylight/deploy/jetson/warm-asr.sh
sudo systemctl start "storylight@${USER}.service"
```

Model and engine state is stored beneath `/var/cache/storylight`, outside the read-only application tree.

## Local Gemma and TensorRT

The default production-safe compiler remains deterministic. Local Gemma and TensorRT planner paths are optional and must pass the same grounding, privacy, schema, and refusal checks before activation.

The retained tooling supports:

- a user-scoped loopback Ollama service;
- TensorRT Edge-LLM installation;
- a finite Gemma 4 checkpoint export and checksum-verified install;
- a loopback-only resident TensorRT server;
- explicit planner configuration and readiness checks.

Use `install-tensorrt-edge-llm.sh`, `install-gemma4-tensorrt-checkpoint.py`, `run-tensorrt-edge-server.sh`, and `configure-tensorrt-planner.sh` only after reviewing their pinned versions and storage requirements. Experimental planner timings do not establish live microphone-to-projector latency.

## Projector kiosk

Copy `kiosk.env.example` to the service configuration and run `check-kiosk-session.sh` before enabling the kiosk. `launch-kiosk.sh` prefers Chromium with its sandbox intact and falls back to Firefox. The installer does not enable graphical autologin or user lingering.

## Privacy verification

Run the socket and traffic audit after configuration:

```bash
./deploy/jetson/check-privacy.sh
```

The intended boundary is documented in [Privacy and data flow](../../docs/privacy.md). The Jetson profile is a local appliance configuration, not an internet-facing multi-user service.

## Troubleshooting

- Run `check-device.sh --strict` before debugging application code.
- Confirm `/readyz` and `/v1/runtime:status` before opening the projector.
- Keep ASR disabled until its model and engine have been warmed explicitly.
- Confirm provider credentials are readable by the service user and absent from browser state.
- Inspect the active systemd units before changing ports or model endpoints.
- Keep minimum cloud instances at zero outside supervised demonstrations.

Historical promotion campaigns, profiler receipts, power-mode comparisons, and rejected checkpoint candidates are intentionally omitted from this guide. The measured conclusions that remain relevant are in [Research results](../../docs/research-results.md).
