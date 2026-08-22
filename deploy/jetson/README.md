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
creates or reuses `.venv` and installs Bookforge in editable mode. Set `BOOKFORGE_VENV_PATH` to use
a different virtualenv location.

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
```

Open `http://127.0.0.1:8080/projector` on the attached display. The environment intentionally uses
the fake model backend for the first boot. Change the model settings only after its endpoint has
been independently tested.

The API now builds every speech engine behind the portable `AsrBackend` contract. It supports
`mlx_whisper` on the development Mac plus deterministic `disabled` and `test` implementations
without Jetson-only imports. A production Jetson ASR engine is not yet implemented. Keep
`BOOKFORGE_ASR_BACKEND=disabled` on the board until that integration is implemented and measured.

## 4. Install the API service

The checked-out application must be available at `/opt/bookforge`, which is the explicit path in
the unit templates. One way to stage the current checkout is:

```bash
sudo install -d /opt/bookforge
sudo cp -a . /opt/bookforge/
sudo chown -R "$USER":"$(id -gn)" /opt/bookforge
cd /opt/bookforge
./deploy/jetson/bootstrap.sh --create-venv --install-app
```

Review and install the environment and service template:

```bash
sudo install -d /etc/bookforge
sudo install -m 0640 deploy/jetson/bookforge.env.example /etc/bookforge/bookforge.env
sudo install -m 0644 deploy/jetson/systemd/bookforge@.service /etc/systemd/system/bookforge@.service
sudo systemctl daemon-reload
sudo systemctl enable --now "bookforge@${USER}.service"
```

Verify it without exposing the API beyond the device:

```bash
systemctl status "bookforge@${USER}.service" --no-pager
journalctl -u "bookforge@${USER}.service" -n 100 --no-pager
curl -fsS http://127.0.0.1:8080/healthz
```

The unit binds only to loopback, reads `/etc/bookforge/bookforge.env`, runs without elevated
privileges, and applies conservative systemd hardening. It does not grant camera, audio, GPIO, or
Docker permissions.

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
add `--no-sandbox` to work around session or permission problems.

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
