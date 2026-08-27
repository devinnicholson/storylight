# Bookforge on Jetson Orin Nano

This directory packages the existing Bookforge API and projector UI for an NVIDIA Jetson Orin
Nano running **JetPack 7.2.1 / Jetson Linux 39.2.1**. NVIDIA lists JetPack 7.2.1 with CUDA 13.2.1
and TensorRT 10.16.2. Use NVIDIA's current
[JetPack downloads and release notes](https://developer.nvidia.com/embedded/jetpack/downloads) as
the source of truth.

Nothing here flashes a board. `bootstrap.sh` is read-only with no arguments and refuses mutation
when the host is not an aarch64 Jetson on L4T 39.2.1. Installing or upgrading the BSP is a separate,
explicit device-administration task that must follow NVIDIA's documentation.

## Included files

- `check-device.sh`: read-only report for L4T, CUDA, TensorRT, Docker, Python, power mode, NVMe,
  camera, microphone, display, projector browser, and thermal zones.
- `bootstrap.sh`: diagnostic-first setup; mutation requires an explicit option.
- `bookforge.env.example`: conservative API environment with ASR disabled by default.
- `bookforge.standalone.env.example`: local-Gemma/Modal profile for portable operation.
- `controller.env.example`: paired phone-gateway configuration without a committed secret.
- `install-standalone.sh`: guarded service installer for an already-staged `/opt/bookforge` tree.
- `show-controller-pairing.sh`: prints the private pairing URL and an optional terminal QR code.
- `check-kiosk-session.sh`: fail-closed, read-only lock/idle/DPMS preflight for the X11 projector
  session.
- `launch-kiosk.sh`: Chromium-first projector launcher with a Firefox fallback and no privilege
  escalation. Chromium retains its sandbox and background-network hardening.
- `collect-evidence.sh`: one-command JSON acceptance artifact, with optional real I/O exercises.
- `check-privacy.sh`: fail-closed process socket audit for listeners plus active TCP/UDP traffic.
- `warm-asr.sh`: supervised-service-safe Whisper checkpoint and TensorRT engine warmup.
- `install-tensorrt-edge-llm.sh`: pinned, user-owned TensorRT Edge-LLM v0.10.0 build for Jetson
  Orin and JetPack 7.2.1.
- `build-tensorrt-edge-engine.sh`: bounded target-device INT4 engine build with automatic Gemma
  unload/restore and thermal evidence.
- `benchmark-tensorrt-edge-llm.py` and `run-tensorrt-edge-benchmark.sh`: local-only five-passage
  schema, privacy, fidelity, latency, memory, and power acceptance; never auto-promotes a model.
- `run-power-mode-ab.sh`: explicit, reboot-aware 25W/MAXN_SUPER comparison with persistent evidence
  and a required restore verification.
- `bookforge-admin` and `install-bookforge-admin.sh`: root-owned, fixed-command administration with
  a narrowly scoped passwordless sudo rule; no password storage or arbitrary shell access.
- `systemd/bookforge@.service`: system API service parameterized by the Linux user.
- `systemd/bookforge-controller@.service`: authenticated, allowlisted phone gateway on port 8081.
- `systemd/bookforge-kiosk.service`: graphical-session user service for the projector browser.
- `systemd/bookforge-gemma.service`: loopback-only, user-scoped Ollama service for the local Gemma
  scene planner.
- `kiosk.env.example`: kiosk URL/browser overrides.

## Standalone portable topology

Bookforge can run without a Mac. The Jetson hosts the private API, local Gemma planner, cached
artwork, and attached-projector browser. A phone on the same **private WPA2/WPA3 network** controls
generation through a separate paired gateway:

```text
phone :8081 -> paired allowlisted gateway -> Jetson loopback API :8080
                                             |-> local Gemma :11434
                                             |-> authenticated Modal renderer
projector browser --------------------------> Jetson loopback API :8080
```

The story API, local model, audio routes, camera routes, diagnostics, and projector are never bound
to the LAN. The gateway exposes only the workbench, scene status/assets, planner preparation, and
scene-generation routes. A 256-bit token is exchanged for a temporary HttpOnly, SameSite session;
the token remains in the pairing URL fragment and is neither sent in an HTTP URL nor forwarded to
the story API or renderer.

For a portable demo, connect the Jetson and phone to a small travel router. The router can use
Wi-Fi or phone tethering as its upstream internet connection, while keeping the demo devices on a
stable private network. Cloud artwork requires upstream internet; the projector, cached scenes,
procedural draft, and local Gemma remain device-local. Do not expose port 8081 on public Wi-Fi or
configure router port forwarding. Pairing is access control, not transport encryption; this HTTP
profile is intentionally limited to a private, operator-controlled WLAN.

After the Jetson has successfully joined that WLAN once, apply the bounded portable-network
profile:

```bash
sudo /opt/bookforge/deploy/jetson/configure-portable-network.sh
```

The helper keeps the active saved Wi-Fi connection on autoconnect, disables client power saving,
enables mDNS for that connection, and limits Avahi advertisements to the active Wi-Fi interface
over IPv4. It never changes the SSID credential, address, route, or DNS configuration. This avoids
`jetson.local` resolving to the USB gadget, Docker bridge, or stale IPv6 address when the unit is
running cable-free. It stores the original Avahi file once at
`/etc/avahi/avahi-daemon.conf.bookforge-backup`; restore that file and restart Avahi to roll the
discovery policy back. A travel-router DHCP reservation remains the most deterministic fallback.

After staging the committed repository at `/opt/bookforge`, create the venv with the Modal runtime:

```bash
cd /opt/bookforge
./deploy/jetson/bootstrap.sh --create-venv --install-modal-runtime
sudo ./deploy/jetson/install-standalone.sh \
  --user "$USER" \
  --import-modal-profile \
  --deploy-renderer
```

The installer refuses non-Jetson hosts and any checkout outside `/opt/bookforge`. It does not
change Wi-Fi, JetPack, power mode, storage, display, login, or browser settings. It preserves any
existing API environment and pairing secret. On the first run it creates root-only environment
files; add `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` to `/etc/bookforge/bookforge.env`, then rerun
the installer to start the services. Never paste those credentials into the phone or browser.
It installs and enables the existing local-Gemma and kiosk user units, but deliberately does not
enable autologin or user lingering. Sign in on the attached projector after a reboot; that normal
graphical login starts Gemma and the projector kiosk without requiring a Mac.
`--deploy-renderer` is explicit because it changes the authenticated Modal app definition. The app
has zero minimum containers, so deployment allocates no idle GPU; the existing per-call billing
gate still runs before any prewarm or scene. Omit the flag when the exact app revision is already
deployed.
`--import-modal-profile` reads the one active profile from the service user's mode-0600
`~/.modal.toml`, validates both credential fields without displaying them, and atomically writes
them into the root-only service environment. Omit it when the service environment is already
configured or when the Jetson should not retain cloud-renderer credentials.

The immutable Modal budget plan stays under `/opt/bookforge`. Its runtime ledger is written to
`/var/lib/bookforge/live-scenes/modal-ledger.json`, so the hardened application tree remains
read-only while cost reservations and settlements remain durable across service restarts.

Show the private pairing URL only when the operator is ready to connect the phone:

```bash
sudo /opt/bookforge/deploy/jetson/show-controller-pairing.sh
```

Open or scan the printed URL. The phone will land on
`/workbench?session=bookforge-live`; the attached projector continues using the same canonical
session on `127.0.0.1:8080`. `jetson.local` requires working mDNS on the private network. If the
phone cannot resolve it, use the router's reserved Jetson address by running:

```bash
sudo BOOKFORGE_CONTROLLER_HOST=192.168.8.20 \
  /opt/bookforge/deploy/jetson/show-controller-pairing.sh
```

The override accepts a DNS name or IPv4 address. A travel-router DHCP reservation is preferred over
hard-coding an address on the Jetson.

Verify the boundary from the Jetson:

```bash
curl -fsS http://127.0.0.1:8080/readyz
curl -fsS http://127.0.0.1:8081/healthz
ss -ltn | grep -E '127\.0\.0\.1:8080|0\.0\.0\.0:8081|127\.0\.0\.1:11434'
systemctl status "bookforge@${USER}.service" --no-pager
systemctl status "bookforge-controller@${USER}.service" --no-pager
```

Only the controller gateway should listen on all interfaces. If the network is not private, stop
it immediately with `sudo systemctl stop "bookforge-controller@${USER}.service"`; local projector
operation remains available.

## 1. Inspect the device

From the repository checkout:

```bash
./deploy/jetson/check-device.sh
```

The default always completes the report and returns success so it is convenient during bring-up.
For an acceptance check that fails when required platform components are missing or mismatched:

```bash
./deploy/jetson/check-device.sh --strict
```

Warnings identify optional or permission-dependent capabilities. The script does not change power
modes, clocks, fan behavior, storage, device permissions, packages, or services.

## 2. Install the Python runtime

The safe first run is equivalent to the device check:

```bash
./deploy/jetson/bootstrap.sh
```

The following options are deliberately independent and explicit:

```bash
./deploy/jetson/bootstrap.sh --install-system-packages
./deploy/jetson/bootstrap.sh --create-venv --install-app
```

The first command uses Ubuntu `apt` to install only Python venv/pip, FFmpeg, V4L2 and ALSA
utilities, curl, CA certificates, and Git. It does **not** install or upgrade JetPack. The second
creates a virtual environment that can see JetPack's system Python packages and installs a built
Bookforge package rather than an editable checkout. Set `BOOKFORGE_VENV_PATH` to use a different
location. An existing environment without `include-system-site-packages = true` is rejected rather
than silently hiding TensorRT and other JetPack bindings.

Installing the application downloads Python packages. Review the command and network policy before
running it. Re-running either action is safe: apt and Python venv/package installation are
idempotent for the same checkout.

## 3. Run interactively first

Start with deterministic model behavior and speech recognition disabled:

```bash
cp deploy/jetson/bookforge.env.example .env
set -a
source .env
set +a
.venv/bin/python -m uvicorn bookforge.api:app --host 127.0.0.1 --port 8080 --timeout-graceful-shutdown 3
```

In another terminal:

```bash
curl -fsS http://127.0.0.1:8080/healthz
curl -fsS http://127.0.0.1:8080/readyz
curl -fsS http://127.0.0.1:8080/v1/runtime:status
```

Open `http://127.0.0.1:8080/projector` on the attached display. The environment intentionally uses
the fake model backend for the first boot. Change the model settings only after its endpoint has
been independently tested. On a new data directory, `pack=latest` falls back visibly to the bundled
Moon Gate fixture rather than opening an empty projector. Remove the checkout `.env` before using
the system service; Jetson preflight deliberately rejects `/opt/bookforge/.env`.

The API builds every speech engine behind the portable `AsrBackend` contract. It includes a lazy,
serialized NVIDIA-AI-IOT WhisperTRT adapter; importing Bookforge does not import CUDA, TensorRT,
Torch, or WhisperTRT. Keep `BOOKFORGE_ASR_BACKEND=disabled` for first boot. The checked upstream
WhisperTRT revision is `268eff10a1e38118a2734745b9db14f7419a08a5`.

WhisperTRT itself does not declare its Torch, TensorRT, torch2trt, Whisper, NumPy, or psutil
dependencies. Install the JetPack-compatible dependencies from current NVIDIA guidance. Install
the pinned adapter only after `/opt/bookforge/.venv` exists in the next section; installing it into
the temporary checkout environment would be discarded during staging.

Do not make a live reading perform the first build. The pinned upstream loader also needs the
OpenAI Whisper checkpoint under the service home, while the hardened unit makes the normal user
home read-only. After installing the service below, stop it, select `whisper_trt` in the environment,
and run the provided warmup as the service user. It writes both checkpoint state and the TensorRT
engine beneath `/var/cache/bookforge`, then the acceptance run measures warm inference. Bookforge
holds GPU serialization through cancellation, so a cancelled request cannot start a second build
against the same engine. JetPack 7.2.1 compatibility remains a physical-device acceptance gate
because the upstream repository does not currently state a JetPack 7 support matrix.

### Install the local Gemma planner without sudo

The accepted Orin Nano configuration uses Ollama `0.32.15` and
`gemma3:1b-it-q4_K_M`. The model is an actual 999.89M-parameter Gemma 3 instruction
model, not a fake backend. Its 815 MB Q4_K_M weights leave enough unified memory for the
projector desktop on an 8 GB board. Do not substitute Gemma 4 E2B or another model larger than
1B without a separate memory and latency acceptance run; this device has no swap.

#### TensorRT Edge-LLM evaluation

The pinned NVIDIA TensorRT Edge-LLM v0.10.0 runtime and its official plugin now build natively on
this Orin Nano. The target engine build is deliberately separate from the accepted Gemma service:
it unloads the resident model, builds or benchmarks one candidate, and restores Gemma with infinite
judged-demo residency on every exit path. Checkpoint export may run CPU-only on Modal, but the
hardware-specific TensorRT engine is always built and executed locally.

The first public control used `Qwen/Qwen2.5-0.5B-Instruct-AWQ` only to validate the toolchain. Its
465,500,604-byte INT4 engine built in 88.034 seconds, peaked at 916 MiB of TensorRT GPU allocation,
and stayed below 51.2°C GPU temperature. Inference reached 97.45 generated tokens/second on the
exact production prompt and 101.43 tokens/second on a 76% shorter prompt—roughly 3.4–3.6 times the
accepted Gemma decode throughput. It nevertheless returned zero valid JSON plans across both
five-passage runs. The control is therefore rejected and is not selectable by production.

The result is useful: TensorRT has enough performance to change the live experience, but the next
candidate must preserve Gemma-level understanding. NVIDIA lists `google/gemma-4-E2B-it` as supported
by this pinned runtime. The exact revision used here is public and ungated; no Hugging Face token or
click-through acceptance is required. The repository now contains a finite INT4-AWQ, text-only
exporter that externalizes FFN weights for the 8 GB Jetson, plus an on-device engine builder and
shadow benchmark. Modal refused both A100-80GB and L40S allocation without a payment method, so the
candidate was not quantized, downloaded, built, or promoted. The linked GCP project was also
checked after enabling Compute Engine: billing is active, no resources exist, and its global GPU
quota is zero. Gemma 3 remains production. Full
control evidence is in `benchmarks/bookforge-tensorrt-edge-llm-2026-08-26.json`; the exact warm
baseline, blocked export attempts, cost reconciliation, and promotion gate are in
`benchmarks/bookforge-gemma4-tensorrt-edge-llm-2026-08-26.json`.

Reproduce the already-pinned control only when validating a new JetPack image:

```bash
deploy/jetson/install-tensorrt-edge-llm.sh
deploy/jetson/build-tensorrt-edge-engine.sh
BOOKFORGE_EDGELLM_PROMPT_PROFILE=production \
  deploy/jetson/run-tensorrt-edge-benchmark.sh
```

The benchmark exits nonzero when any output misses the strict wire schema. A high token rate is not
an acceptance result.

When an authorized cloud GPU is available, the Gemma 4 sequence is deliberately staged:

```bash
# Cloud: pinned BF16 -> INT4-AWQ -> text-only ONNX with external FFN weights.
modal run deploy/modal_gemma4_tensorrt_edge_export.py::export_cli
# Transfer only gemma4-e2b-it-int4-awq-v010/onnx to the matching Jetson model root.
deploy/jetson/build-gemma4-tensorrt-edge-engine.sh
deploy/jetson/run-gemma4-tensorrt-edge-benchmark.sh
```

The first pass excludes Gemma 4 MTP. Only add the assistant after the target-only engine passes all
schema, privacy, semantic, memory, and measured end-to-end gates. The shadow runner always restores
Gemma 3 and cannot change the production backend.

#### Reboot-safe 25W versus MAXN_SUPER measurement

On the measured JetPack 7.2.1 Orin Nano, changing from power mode 1 (`25W`) to mode 2
(`MAXN_SUPER`) required a reboot; returning from mode 2 to mode 1 applied immediately. Do not use a
one-process switch/benchmark/restore script: a requested reboot destroys that process and `/tmp`
evidence. The repository runner preserves the accepted 25W baseline under
`/var/lib/bookforge/power-mode-ab`, records a durable phase before each reboot, validates the mode
after reconnect, and refuses out-of-order commands.

Run exactly one phase at a time. A mode-change phase may prompt for a reboot; enter `YES` only after
the script prints its matching durable phase. If restoring 25W applies immediately, run `finalize`
without rebooting:

```bash
sudo /opt/bookforge/deploy/jetson/run-power-mode-ab.sh prepare-maxn
# Reconnect after the MAXN_SUPER reboot.
sudo /opt/bookforge/deploy/jetson/run-power-mode-ab.sh benchmark-maxn
sudo /opt/bookforge/deploy/jetson/run-power-mode-ab.sh restore-25w
# Reconnect only if NVIDIA requested a 25W restore reboot.
sudo /opt/bookforge/deploy/jetson/run-power-mode-ab.sh finalize
```

The benchmark phase never changes power mode. The final phase must observe mode 1 and both local
services before it writes `result=complete`. Evidence remains on the Jetson until it is explicitly
collected; rebooting cannot erase it.

#### Restricted unattended Bookforge administration

Never store the Linux password in the repository, an environment file, a shell command, or a file
for automation to read. Install the root-owned, allowlisted administrator once instead:

```bash
sudo /opt/bookforge/deploy/jetson/install-bookforge-admin.sh --user operator
sudo -n /usr/local/sbin/bookforge-admin status
```

The sudo rule permits only `/usr/local/sbin/bookforge-admin`. That root-owned wrapper accepts fixed
status, Bookforge service restart, and power-acceptance actions; it exposes no shell, arbitrary
systemd unit, arbitrary path, package installation, network mutation, or general root command. The
power runner is copied to a separate root-owned path so editing the Git checkout cannot alter code
executed through passwordless sudo. Removing `/etc/sudoers.d/bookforge-admin-operator`
revokes the delegation, but do so only through an explicitly authorized root maintenance action.

Download the pinned official ARM64 archive into a versioned, user-owned directory. Verify the
release digest before extracting it; do not pipe an unverified installer into a shell:

```bash
BOOKFORGE_OLLAMA_VERSION=0.32.15
BOOKFORGE_OLLAMA_ROOT="$HOME/.local/opt/ollama-v${BOOKFORGE_OLLAMA_VERSION}"
BOOKFORGE_OLLAMA_ARCHIVE="$HOME/.cache/bookforge/downloads/ollama-linux-arm64-v${BOOKFORGE_OLLAMA_VERSION}.tar.zst"

test ! -e "$BOOKFORGE_OLLAMA_ROOT"
install -d -m 0700 \
  "$HOME/.cache/bookforge/downloads" \
  "$BOOKFORGE_OLLAMA_ROOT" \
  "$HOME/.local/share/bookforge/ollama/models" \
  "$HOME/.config/systemd/user"
curl --fail --location --retry 3 \
  --output "$BOOKFORGE_OLLAMA_ARCHIVE" \
  "https://github.com/ollama/ollama/releases/download/v${BOOKFORGE_OLLAMA_VERSION}/ollama-linux-arm64.tar.zst"
printf '%s  %s\n' \
  c898270b1690eab0f51aa9e9197686b7b4c6a7d88b83967763818f3127e477e9 \
  "$BOOKFORGE_OLLAMA_ARCHIVE" | sha256sum --check --strict
zstd --test "$BOOKFORGE_OLLAMA_ARCHIVE"
zstd -dc "$BOOKFORGE_OLLAMA_ARCHIVE" | tar -xf - -C "$BOOKFORGE_OLLAMA_ROOT"
chmod 0755 "$BOOKFORGE_OLLAMA_ROOT/bin/ollama"
sha256sum "$BOOKFORGE_OLLAMA_ROOT/bin/ollama"
```

The accepted binary SHA-256 is
`db3793652a24aaf4bbfcab4460a4539e413e50193322a224bd5fb521930292e0`.
Install and start the user service; it binds only to `127.0.0.1`, permits one loaded model and one
request at a time, defaults to a 4096-token context, keeps history off, and disables Ollama cloud:

```bash
install -m 0644 deploy/jetson/systemd/bookforge-gemma.service \
  "$HOME/.config/systemd/user/bookforge-gemma.service"
systemd-analyze --user verify "$HOME/.config/systemd/user/bookforge-gemma.service"
systemctl --user daemon-reload
systemctl --user enable --now bookforge-gemma.service
curl -fsS http://127.0.0.1:11434/api/version
ss -ltnp | grep ':11434\b'
loginctl show-user "$USER" -p Linger
```

The accepted device reports `Linger=no`, so this user service starts with the normal Jetson login
session rather than before login. That matches the projector demo, which also requires the user's
graphical session. Enabling linger is an optional administrator change and was deliberately not
performed by this no-sudo install.

Pull and verify only the accepted 1B model:

```bash
OLLAMA_HOST=http://127.0.0.1:11434 \
  "$BOOKFORGE_OLLAMA_ROOT/bin/ollama" pull gemma3:1b-it-q4_K_M
curl -fsS http://127.0.0.1:11434/api/tags | python3 -c '
import json, sys
model = json.load(sys.stdin)["models"][0]
assert model["name"] == "gemma3:1b-it-q4_K_M"
assert model["digest"] == "8648f39daa8fbf5b18c7b4e6a8fb4990c692751d49917417b8842ca5758e7ffc"
assert model["details"]["parameter_size"] == "999.89M"
assert model["details"]["quantization_level"] == "Q4_K_M"
print(model["digest"])
'
sha256sum \
  "$HOME/.local/share/bookforge/ollama/models/blobs/sha256-7cd4618c1faf8b7233c6c906dac1694b6a47684b37b8895d470ac688520b9c01"
```

The final command must print the model-layer digest encoded in its filename. The Ollama manifest
digest is `8648f39daa8fbf5b18c7b4e6a8fb4990c692751d49917417b8842ca5758e7ffc`; the
815,310,432-byte model layer is
`7cd4618c1faf8b7233c6c906dac1694b6a47684b37b8895d470ac688520b9c01`.

Run one long-timeout strict-schema request to populate the CUDA kernel cache, then repeat it under
the steady-state acceptance deadline before connecting Bookforge:

```bash
curl --fail --silent --show-error --max-time 240 \
  --header 'Content-Type: application/json' \
  --data-binary @benchmarks/jetson-gemma3-optimized-schema-request.json \
  http://127.0.0.1:11434/api/chat | python3 -m json.tool
curl --fail --silent --show-error --max-time 20 \
  --header 'Content-Type: application/json' \
  --data-binary @benchmarks/jetson-gemma3-optimized-schema-request.json \
  http://127.0.0.1:11434/api/chat | python3 -m json.tool
OLLAMA_HOST=http://127.0.0.1:11434 \
  "$BOOKFORGE_OLLAMA_ROOT/bin/ollama" ps
```

`ollama ps` must report `100% GPU`. The accepted repeatable cold reload was 10.40 seconds and the
warm new-passage request was 4.57 seconds at roughly 27-29 generated tokens per second. The very
first request took about 152 seconds while CUDA compiled and cached kernels; always prewarm before
a live reading. Full evidence is in
`benchmarks/jetson-gemma3-ollama-2026-08-23.json`. The optimized request fixture preserves the
hardware-accepted style-bound prompt and the still-current `LiveSceneWirePlan` schema; its canonical
JSON Schema SHA-256 is `d08b410c34d519a12410e2b22beb89beb2ec9d89a06887c783d3a6ce44839c14`.
Current code no longer sends visual style to the semantic planner, so the next exact Jetson report
must come from the counterbalanced harness below rather than mutating this historical fixture.
The earlier `jetson-gemma3-schema-request.json` remains immutable historical evidence for the first
accepted end-to-end run and is not the current production contract.

For Bookforge on the same Jetson, use these settings after the independent probe passes:

```dotenv
BOOKFORGE_MODEL_BACKEND=ollama
BOOKFORGE_MODEL_NAME=gemma3:1b-it-q4_K_M
BOOKFORGE_MODEL_BASE_URL=http://127.0.0.1:11434
BOOKFORGE_MODEL_TIMEOUT_SECONDS=20
BOOKFORGE_MODEL_KEEP_ALIVE=-1m
BOOKFORGE_MODEL_CONTEXT_TOKENS=4096
BOOKFORGE_MODEL_MAX_OUTPUT_TOKENS=180
BOOKFORGE_MODEL_REQUIRE_GPU=true
BOOKFORGE_LIVE_SCENE_PLANNER=model
BOOKFORGE_LIVE_SCENE_PLANNER_TIMEOUT_SECONDS=12
BOOKFORGE_LIVE_SCENE_PLANNER_MODEL_REVISION=ollama-manifest-sha256:8648f39daa8fbf5b18c7b4e6a8fb4990c692751d49917417b8842ca5758e7ffc
BOOKFORGE_LIVE_SCENE_AUTO_PREWARM_ON_SUBMIT=true
# Enable only after the text-free warmup passes the exact Jetson latency/memory gate.
BOOKFORGE_LIVE_SCENE_PLANNER_AUTO_WARMUP=true
```

The experimental short-key tuple contract is deliberately unavailable in runtime configuration.
Exact Jetson testing on the warmed 1B model reduced mean planning from 3.766 seconds to 1.398
seconds, but four of five outputs echoed schema placeholders and all five missed their required
transformation. The Mac-only candidate therefore failed the hardware semantic gate and was removed
instead of exposing a dangerous speed switch. Its class remains only for the explicit offline
benchmark harness and rejection evidence.

`BOOKFORGE_MODEL_REQUIRE_GPU=true` verifies the warmed Ollama model has a nonzero VRAM allocation.
This closes a failure seen after a boot where the kernel reported `ACR bootstrap failed`, the GPU
device was absent, and Ollama silently fell back to `100% CPU` while the model-install probe still
looked healthy. A clean reboot restored `/dev/nvhost-gpu` and `100% GPU`; Bookforge now fails the
warmup/first generation instead of accepting that slow path.

`BOOKFORGE_MODEL_MAX_OUTPUT_TOKENS` is a hard decode ceiling, not a target. The final compact contract
removed redundant camera, lighting, palette, and region fields; its accepted hero repeats used
107-109 of 180 tokens without truncation. Keep the 180 ceiling until a broader benchmark proves that
a lower ceiling never truncates valid JSON. The 12-second planner deadline passed comfortably on a
prewarmed model; keep the independent
20-second model-client timeout for
diagnostics and ensure the prewarm completes before a live reading.

The optional workbench warmup sends only a fixed `{"ready":true}` readiness task to the local
model while the user types. It never includes the textarea, visual style, audio, or a renderer call.
While the visible workbench remains open, it refreshes that text-free readiness task every eight
minutes and when a stale tab becomes visible again, staying inside Ollama's ten-minute keep-alive.
If Gemma still times out or fails privacy/validation, the job now stops before the paid renderer;
it never promotes a generic fallback as if it were the requested scene.
The Mac A/B converted a 12-second cold planner timeout into a 3.53-second uncached plan after a
6.39-second background warmup. Keep it opt-in until the same unload/warmup/plan sequence passes on
Jetson with the projector browser running and the service memory limits enforced.

The live planner keeps up to 32 privacy-gated semantic plans in memory. Identical-passage rereads,
visual-style auditions, and alternate-seed retries skip Gemma decode while still deriving a new
styled, seeded SceneSpec;
the workbench reports `local cache` instead of presenting that path as fresh inference. The digest
also binds the model revision and wire contract, and nothing is persisted or sent off-device.
Common breeds/species, plants, objects, and actions remain explicit while a local normalizer removes
duplicate actors across background/support layers. Open landscape plans also reject unrequested
walls, caves, portals, frames, monoliths, and giant abstract structures before rendering.

The final speed acceptance kept Gemma 1B after rejecting the 270M model for invented settings and
missing transformations. The prior 896x512, 8-step SANA 1.5 renderer produced `master_ready` in
6.971 seconds. The current 1024x576, 2-step SANA-Sprint renderer has now passed the exact prepared
API path in 499.6 ms after a privacy-gated Jetson plan cache hit; its same-prompt resolution A/B
added 28.6% source pixels for 18.9 ms of inference. The renderer's 90-second scale-down window is
intentional: the earlier
30-second window expired during local planning/operator handoff and produced a 44.8-second cold
request. Automatic prewarm overlaps its text-free preparation with Gemma planning; it does not send
the passage to Modal. Full evidence is in
`benchmarks/bookforge-speed-optimization-2026-08-23.json`.

When the Jetson is available, compare the accepted and short-key contracts with the local-only
five-passage harness. A warmup is run and excluded, and which contract runs first alternates by
passage so shared-prefix cache reuse cannot systematically favor one side. The report fails
technical acceptance when any case exceeds 12 seconds or 180 output tokens, and it still requires
human semantic review:

```bash
python -m bookforge.planner_benchmark \
  --contract both \
  --model-revision ollama-manifest-sha256:8648f39daa8fbf5b18c7b4e6a8fb4990c692751d49917417b8842ca5758e7ffc \
  --output benchmarks/jetson-gemma3-short-key-acceptance.json
```

For the Mac-to-Jetson SSH loopback forward, add `--base-url http://127.0.0.1:11435`. The harness
rejects non-loopback model URLs and never calls Modal or GCP.

Keep the model endpoint on loopback. When a Mac control plane needs it during development, use an
explicit SSH local forward rather than changing `OLLAMA_HOST` to a LAN address.

#### Integrated Gemma-to-scene acceptance

Job `scene_7ad76946940540f4b2f8878e` completed on August 23, 2026 with
`planning_status=model`, no fallback, no warning, and no error. The path was:

```text
local passage -> Jetson Gemma LiveScenePlan -> normalized SceneSpec v2
              -> finite Modal SANA master + Depth Anything map -> projector assets
```

The warm Gemma call processed 537 prompt tokens and generated 283 tokens in 9.03 seconds. The
application's validated planning stage took 9.165 seconds, below its 12-second deadline. Ollama
reported 32.48 generated tokens per second and confirmed that the response was not truncated. The
complete job reached `master_ready` in 15.005 seconds: 9.165 seconds planning, 5.807 seconds in the
finite provider call, 4 milliseconds of cache work, and 30 milliseconds of remaining orchestration.
Inside the provider call, SANA took 3.826 seconds and Depth Anything took 115 milliseconds. The
provider manifest estimated `$0.001289` of L4 GPU cost for the 5.807-second generation RPC alone.
The conservative ledger entry for the complete warm session is `$0.012836`: 22.013 seconds of
prewarm, 5.807 seconds of generation, and a 30-second scale-down allowance at `$0.000222/second`.
These are not the same scope. A read-only Modal billing report at 14:48:52 PDT showed `$0.05987830`
for the current `bookforge-fast-scene` app/day interval since the 13:46:13 baseline; that wider
interval may include deployment, prewarm, generation, and billing lag, so it is not a job-only
price. At that capture the workspace total was `$13.95606460`, leaving `$16.04393540` of the
monthly credit and `$15.04393540` before the project's `$29` hard-stop threshold.

All three model revisions were captured:

- Gemma scene planner: manifest
  `8648f39daa8fbf5b18c7b4e6a8fb4990c692751d49917417b8842ca5758e7ffc`
- SANA 1.5 1.6B: revision `caa51e5ea874be07d3a9c7c2d0fd800570b18440`
- Depth Anything V2 Small: revision `b4769fd619394250528294b658587285526fab1c`

The accepted 1024x576 RGB master is 621,636 bytes with SHA-256
`dbcd5d176857656aa4c7a41c848e36626a8fd77bd5a8223df1f3212e3fc3e049`. The accepted 1024x576
grayscale depth map is 73,142 bytes with SHA-256
`bf9b4a185b272a083ad95efbd3953b43b931ec7376a85726b4e26ad45890e5c8`. Both generated-file
hashes independently matched their content-addressed cache copies. The provider manifest is
`artifacts/live-scenes/generated/scene_7ad76946940540f4b2f8878e/scene.manifest.json`, SHA-256
`567eac7b1a50ab05696854bfe75f67a4beb526b831b96fcd828f005eb4babce1`.

The privacy boundary is explicit: the raw passage was processed by the local Bookforge/Jetson
path, while the model-authored semantic visual prompt was sent to Modal for image generation. The
exact source passage and local session ID are absent from the provider manifest request. The
manifest confirms a finite authenticated call and no persistent endpoint; this is semantic-data
minimization, not a claim that cloud image generation sees no story information. This is also a
job-specific observation, not a code-enforced guarantee for that accepted run: at the time of the
job, the path had no post-Gemma source-overlap or PII gate, so another model response that echoed
private input could have reached Modal.

After this job, the planner prompt gained grammar and trailing-punctuation guidance. Its JSON
Schema did not change, and no additional inference was run merely to validate that wording-only
edit. The current request fixture is byte-for-byte checked against the current Python contract;
the integrated job and this distinction are recorded in
`benchmarks/jetson-gemma3-ollama-2026-08-23.json`.

#### Update and rollback

Treat runtime updates like deploys. Download the new official release into a different versioned
directory, verify the digest published with that release, and preserve the known-good directory and
service file. Change only `ExecStart` in a reviewed copy of `bookforge-gemma.service`, run
`systemd-analyze --user verify`, then restart and repeat the schema, GPU, memory, latency, and
loopback checks. Do not update the runtime and model in the same acceptance run.

Before switching versions, save the known-good unit:

```bash
cp "$HOME/.config/systemd/user/bookforge-gemma.service" \
  "$HOME/.config/systemd/user/bookforge-gemma.service.known-good"
```

Rollback is a unit-file restore; the old versioned runtime and model remain intact:

```bash
install -m 0644 "$HOME/.config/systemd/user/bookforge-gemma.service.known-good" \
  "$HOME/.config/systemd/user/bookforge-gemma.service"
systemctl --user daemon-reload
systemctl --user restart bookforge-gemma.service
curl -fsS http://127.0.0.1:11434/api/version
curl -fsS http://127.0.0.1:11434/api/tags
```

Ollama may create a private identity key under `$HOME/.ollama`. It is runtime state: never copy it
into the repository, benchmark evidence, logs, or a demo package.

## 4. Install the API service

The application must be available at `/opt/bookforge`, which is the explicit path in the unit
templates. Stage only committed files into an empty target; do not copy `.env`, `.git`, a laptop
virtual environment, caches, or evidence:

```bash
git archive --format=tar --output=/tmp/bookforge-source.tar HEAD
sudo install -d -m 0755 /opt/bookforge
sudo tar --extract --file=/tmp/bookforge-source.tar --directory=/opt/bookforge --no-same-owner
sudo chown -R "$USER":"$(id -gn)" /opt/bookforge
cd /opt/bookforge
./deploy/jetson/bootstrap.sh --create-venv --install-app
```

Review and install the environment and service template:

```bash
sudo install -d /etc/bookforge
sudo install -o root -g "$(id -gn)" -m 0640 \
  deploy/jetson/bookforge.env.example /etc/bookforge/bookforge.env
sudo install -m 0644 deploy/jetson/systemd/bookforge@.service /etc/systemd/system/bookforge@.service
sudo systemctl daemon-reload
sudo systemctl enable --now "bookforge@${USER}.service"
```

Verify it without exposing the API beyond the device:

```bash
systemctl status "bookforge@${USER}.service" --no-pager
journalctl -u "bookforge@${USER}.service" -n 100 --no-pager
curl -fsS http://127.0.0.1:8080/healthz
curl -fsS http://127.0.0.1:8080/readyz
curl -fsS http://127.0.0.1:8080/v1/runtime:status
```

The unit binds only to loopback, reads `/etc/bookforge/bookforge.env`, runs without elevated
privileges, and applies conservative systemd hardening. It creates private, service-owned
`/var/lib/bookforge`, `/var/cache/bookforge`, and `/run/bookforge` directories. Startup preflight
refuses relative Jetson paths, a Mac-only ASR backend, an unavailable configured ASR backend, or a
remote model endpoint. It does not grant camera, audio, GPIO, or Docker permissions.

### Warm and activate WhisperTRT

After the disabled-ASR service has started once and created its private cache directory:

```bash
cd /opt/bookforge
.venv/bin/python -c 'import torch, tensorrt, torch2trt, whisper, numpy, psutil'
.venv/bin/pip install \
  'git+https://github.com/NVIDIA-AI-IOT/whisper_trt.git@268eff10a1e38118a2734745b9db14f7419a08a5'
sudo systemctl stop "bookforge@${USER}.service"
sudoedit /etc/bookforge/bookforge.env
# Set BOOKFORGE_ASR_BACKEND=whisper_trt, then:
sudo -u "$USER" BOOKFORGE_SERVICE_USER="$USER" /opt/bookforge/deploy/jetson/warm-asr.sh
sudo systemctl start "bookforge@${USER}.service"
```

The warmup downloads/builds while network access is intentionally available. Before acceptance,
disconnect networking or apply the demo network policy, restart the service, and prove warm ASR
still works. Never run the warmup concurrently with the service.

### Install a prepared offline book package

Copy a validated Story Pack and its referenced media directory onto the device, then install it as
the same Linux user that runs Bookforge:

```bash
.venv/bin/python -m bookforge.pack_installer /path/to/story-pack.json \
  --asset-root /path/to/media \
  --data-dir /var/lib/bookforge \
  --cache-dir /var/cache/bookforge
curl -fsS http://127.0.0.1:8080/v1/story-packs/latest
```

The installer rejects schema errors, checksum mismatches, package path traversal, unsupported media
types, and ready assets without a local source. It atomically copies permitted image/video files
with private permissions, rewrites their manifest locations to loopback asset URLs, and only then
promotes the Story Pack to `latest`. The projector browser never receives a raw filesystem path.

## 5. Install the projector browser kiosk

The launcher prefers `chromium`, then `chromium-browser`, and accepts `firefox` or `firefox-esr` as
a fallback. Set `BOOKFORGE_BROWSER_BIN` to choose either browser explicitly. The legacy
`BOOKFORGE_CHROMIUM_BIN` override remains supported and is treated as Chromium. Chromium keeps the
existing kiosk/app sandbox and background-network hardening flags; Firefox uses only its supported
`--kiosk` and `--private-window` flags. This repository never silently installs a browser.

On JetPack 7.2.1, the Ubuntu Firefox Snap was measured exposing WebGL as Mesa `llvmpipe`; Bookforge
then pinned a CPU core and delivered only 19 distinct projector frames in two seconds. The verified
Mozilla ARM64 build exposed NVIDIA WebGL and held 60.48 browser fps (17.10 ms median, 17.14 ms p95),
with 118 distinct framebuffer frames in a 120-frame final capture. Install that pinned, checksum-
verified build without sudo, then select its stable path explicitly:

```bash
./deploy/jetson/install-firefox-arm64.sh
install -d -m 700 "${XDG_CONFIG_HOME:-${HOME}/.config}/bookforge"
cp deploy/jetson/kiosk.env.example "${XDG_CONFIG_HOME:-${HOME}/.config}/bookforge/kiosk.env"
# Edit kiosk.env and set:
# BOOKFORGE_BROWSER_BIN=/home/your-user/.local/opt/firefox-bookforge/firefox
```

The Snap remains installed as rollback. The projector runtime also rejects known software WebGL
renderers instead of silently starting a continuously animated depth shader on the CPU.

Test the launcher inside the logged-in graphical desktop session:

```bash
./deploy/jetson/check-kiosk-session.sh
BOOKFORGE_KIOSK_URL='http://127.0.0.1:8080/projector?pack=latest&session=bookforge-live&live=1' \
  ./deploy/jetson/launch-kiosk.sh
```

The preflight requires an active, local X11 session owned by the kiosk user with
`LockedHint=no`, `IdleHint=no`, and `Monitor is On`. It never unlocks the desktop, synthesizes
input, or changes DPMS. If it reports a locked or idle session, unlock it on the physical display,
interact with the desktop, confirm the projector is visibly on, and run the check again. Kiosk
startup maps this operator-state failure to exit status 78; the user service deliberately does not
restart-loop on that status. Restart it manually after the physical session passes:

```bash
systemctl --user restart bookforge-kiosk.service
```

For a dedicated unattended demo account whose automatic lock and blanking policies have already
been disabled, set `BOOKFORGE_KIOSK_ALLOW_IDLE=true` in the private kiosk environment. This permits
only the idle hint: the preflight still rejects a locked, remote, inactive, non-X11, unreadable, or
display-off session. The default remains fail-closed for ordinary accounts.

When the Mac hosts the development API on port 18081 through the loopback-only reverse SSH tunnel,
override the projector and readiness URLs together. Changing only the kiosk URL can falsely report
readiness from the Jetson's separate port-8080 fallback service:

```bash
BOOKFORGE_KIOSK_URL='http://127.0.0.1:18081/projector?pack=latest&session=bookforge-live&live=1&present=1' \
BOOKFORGE_READY_URL=http://127.0.0.1:18081/readyz \
  ./deploy/jetson/launch-kiosk.sh
```

Then install the user service while logged in as the desktop user:

```bash
install -D -m 0644 deploy/jetson/systemd/bookforge-kiosk.service \
  "$HOME/.config/systemd/user/bookforge-kiosk.service"
install -D -m 0644 deploy/jetson/kiosk.env.example \
  "$HOME/.config/bookforge/kiosk.env"
systemctl --user daemon-reload
systemctl --user enable --now bookforge-kiosk.service
```

Inspect it with:

```bash
systemctl --user status bookforge-kiosk.service --no-pager
journalctl --user -u bookforge-kiosk.service -n 100 --no-pager
```

The kiosk must run in the actual graphical user's session. Do not run the browser as root and do
not add `--no-sandbox` to work around Chromium session or permission problems. The launcher waits
for `/readyz` before opening the browser and restarts if the graphical session begins before the
API is ready. Its default URL joins the canonical `bookforge-live` session, enables progressive
live-scene updates, and retains the latest device-stored Story Pack as a fallback.

While the browser process exists, `systemd-inhibit --what=sleep` prevents system suspend. It does
not inhibit `idle`, disable the lock screen, or change a login policy. The visible projector page
also requests the browser Screen Wake Lock API; browsers release that lock when the page becomes
hidden or the workstation locks, then request it again only after the authenticated session is
visible. Stopping the kiosk process releases both protections automatically.

## 6. Capture hardware acceptance evidence

After the service is running and a prepared Story Pack is installed, collect exact JetPack/CUDA/
TensorRT versions, API/model readiness, the actual Bookforge NVMe mount, browser, thermals,
camera, microphone, display, latest content, and process privacy into one timestamped JSON artifact:

```bash
./deploy/jetson/collect-evidence.sh --strict
```

The wrapper resolves the running `bookforge@USER.service` PID and passes it to the collector.
Process privacy evidence is mandatory: neither the Python collector nor this wrapper can report
`ready: true` when the service is stopped, its PID is unavailable, or the socket audit fails. Set
`BOOKFORGE_SERVICE_USER` only when collecting evidence for a service instance owned by a different
user.

After configuring WhisperTRT, run the real three-second microphone capture, one-frame camera
capture, local transcription, and privacy boundary:

```bash
./deploy/jetson/collect-evidence.sh --strict --exercise-io --require-asr
```

The ASR gate defaults to 4,000 ms for the three-second capture. Override it only when documenting a
deliberate acceptance-budget change with `--max-asr-ms`.

Evidence is written under `/var/lib/bookforge/evidence/` by default. Override the exact camera or
ALSA device when enumeration shows a different target:

```bash
./deploy/jetson/collect-evidence.sh --strict --exercise-io --require-asr \
  --camera-device /dev/video2 --audio-device plughw:2,0
```

During a complete offline reading rehearsal, run the dedicated audit from another terminal:

```bash
./deploy/jetson/check-privacy.sh
```

It resolves the supervised API PID, maps that process's descriptors through `/proc`, and fails
closed if socket tables are unreadable, if TCP/UDP activity reaches a non-loopback peer, or if the
process listens beyond loopback. This proves the API process boundary, not all traffic on the
machine. When Chromium is used, the kiosk disables its background networking. The final offline
claim still requires a network-disabled full rehearsal or an independent whole-device packet
capture; the Firefox fallback does not establish that boundary by itself.

## Device caveats

- A reported NVMe device does not prove that Bookforge is using it. Confirm the mount point and free
  space in the diagnostic output before moving models or caches.
- Camera and microphone enumeration proves presence, not capture quality. Test the exact USB camera,
  microphone, resolution, frame rate, room lighting, and projector interference used for the demo.
- `nvpmodel -q` reports the current profile. Only the explicit `run-power-mode-ab.sh` acceptance
  selects a profile; it persists its phase across any required reboot and refuses completion until
  mode 1 (`25W`) is restored. The measured 25W→MAXN transition required one reboot; MAXN→25W
  applied immediately. No script runs maximum-clock commands.
- Thermal-zone readings are a snapshot. Run a full-length rehearsal and record sustained latency and
  temperature; do not infer thermal stability from an idle check.
- The system service does not join the user to `docker`, `video`, or `audio` groups. Granting device
  or daemon access is a separate security decision.
- The Chromium user profile is persistent under the user's state directory. It contains browser
  state, so treat it as user data when imaging or sharing the SSD.
- The kiosk user service starts only after the graphical user session exists. If the demo must
  recover unattended after power loss, configure the supported desktop auto-login policy and run a
  recorded cold-boot acceptance rehearsal; this repository does not silently change login policy.
- WhisperTRT's published Orin Nano benchmark is useful for selecting the first adapter, but it is
  not evidence for this JetPack 7.2.1 installation. Only the generated hardware evidence file and
  recorded rehearsal count as Bookforge results.
