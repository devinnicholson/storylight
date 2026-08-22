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
  camera, microphone, display, Chromium, and thermal zones.
- `bootstrap.sh`: diagnostic-first setup; mutation requires an explicit option.
- `bookforge.env.example`: conservative API environment with ASR disabled by default.
- `launch-kiosk.sh`: Chromium projector launcher without `--no-sandbox` or privilege escalation.
- `collect-evidence.sh`: one-command JSON acceptance artifact, with optional real I/O exercises.
- `check-privacy.sh`: fail-closed process socket audit for listeners plus active TCP/UDP traffic.
- `warm-asr.sh`: supervised-service-safe Whisper checkpoint and TensorRT engine warmup.
- `systemd/bookforge@.service`: system API service parameterized by the Linux user.
- `systemd/bookforge-kiosk.service`: graphical-session user service for Chromium.
- `kiosk.env.example`: kiosk URL/browser overrides.

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
.venv/bin/python -m uvicorn bookforge.api:app --host 127.0.0.1 --port 8080
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
promotes the Story Pack to `latest`. Chromium never receives a raw filesystem path.

## 5. Install the Chromium kiosk

First use `check-device.sh` to confirm that either `chromium` or `chromium-browser` is available.
If it is missing, install Chromium through the supported software installation path on the JetPack
desktop image; packaging differs between Ubuntu images, so this repository does not guess a package
or silently install a browser.

Test the launcher inside the logged-in graphical desktop session:

```bash
BOOKFORGE_KIOSK_URL=http://127.0.0.1:8080/projector ./deploy/jetson/launch-kiosk.sh
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

The kiosk must run in the actual graphical user's session. Do not run Chromium as root and do not
add `--no-sandbox` to work around session or permission problems. The launcher waits for `/readyz`
before opening Chromium and restarts if the graphical session begins before the API is ready. Its
default URL loads the latest device-stored Story Pack and retains the deterministic fixture/manual
fallback.

## 6. Capture hardware acceptance evidence

After the service is running and a prepared Story Pack is installed, collect exact JetPack/CUDA/
TensorRT versions, API/model readiness, the actual Bookforge NVMe mount, Chromium, thermals,
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
machine. The kiosk disables Chromium background networking, but the final offline claim still
requires a network-disabled full rehearsal or an independent whole-device packet capture.

## Device caveats

- A reported NVMe device does not prove that Bookforge is using it. Confirm the mount point and free
  space in the diagnostic output before moving models or caches.
- Camera and microphone enumeration proves presence, not capture quality. Test the exact USB camera,
  microphone, resolution, frame rate, room lighting, and projector interference used for the demo.
- `nvpmodel -q` reports the current profile. These scripts intentionally never select a profile or
  run maximum-clock commands.
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
