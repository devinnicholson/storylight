# Best demo checkpoint — 8 September 2026

The user designated commit `59d13bf46ba93b97e66cdacdf964b1b3f01aa7c0` as the best build so far. Annotated tag **`checkpoints/best-demo-2026-09-08`** points to that commit and is published on origin. Do not move or replace the tag; create another checkpoint for a later accepted build.

This preserves the accurate watercolor voice demo with background Vertex connection preparation, overlapped local scene checks, complete-only presentation and recording timing. The last measured recording took 3.962 seconds from Finish recording to browser preview, including a 3.496-second provider request. This is one recording, not a latency distribution or a physical projector measurement. The baseline passed 1,157 Python tests and seven JavaScript suites.

## What is preserved

- The Git tag preserves all tracked Mac frontend, gateway, tests, documentation and implementation code.
- `runtime.tar.gz` separately preserves 92 deployed Jetson source/configuration files plus a manifest. The running Jetson API differs from the repository API; the archive preserves its exact compatibility-patched version rather than assuming a checkout restores it.
- The archive includes the active `20260908-v5` parser source, package dependency versions, the parser socket override, and the noncredential kiosk URL/runtime override. Every payload hash was verified against its manifest; `receipt.json` records the archive SHA-256 and installed API/router/provider hashes.
- No service was restarted to capture this checkpoint. Credentials, model weights, jobs, audio, generated assets and non-source package data are excluded. Preserve existing model/credential installations during rollback. This is a code/configuration checkpoint, not a full disk or Python-environment backup.

## Roll back frontend or gateway changes

Create an isolated checkout without resetting main or touching untracked user work:

```sh
git fetch origin refs/tags/checkpoints/best-demo-2026-09-08
git worktree add --detach /tmp/bookforge-best-demo-20260908 checkpoints/best-demo-2026-09-08
```

Stop only the verified voice-gateway process currently listening on port 18767. Start its replacement from the checkpoint source, using the existing Mac environment:

```sh
PYTHONPATH=/tmp/bookforge-best-demo-20260908/src \
  /Users/operator/Documents/ChatGPT/golden-ticket/.venv/bin/python \
  -m uvicorn bookforge.voice_gateway:create_app_from_env --factory \
  --host 127.0.0.1 --port 18767 --no-proxy-headers --timeout-graceful-shutdown 3
```

Keep the existing ASR service on 18766 and SSH forwarding on 18768. Reload the demo once to replace the loaded JavaScript. Running from the isolated checkout is a temporary runtime rollback; use a reviewed forward commit if main should permanently adopt the old behavior. Never force-push main to roll back.

## If later experiments change the Jetson

Verify the archive SHA-256 against `receipt.json`, unpack into a new staging directory, and verify each file against `manifest.json` before copying. `package/` contains source for `/opt/bookforge/.venv/lib/python3.12/site-packages/bookforge`; `parser/` contains the source from the manifest's parser release path. Restore through a reviewed installer that backs up the then-current files, copies only manifest-listed source files, and uses the existing restricted `bookforge-admin restart-api` command. Do not overwrite credentials or install the newer repository API over the archived deployed API.

The parser selection was `/opt/bookforge/voice-language-current` → `/opt/bookforge/voice-language-releases/20260908-v5`. The API override belongs at `/etc/systemd/system/bookforge@operator.service.d/50-voice-language.conf`. The kiosk override belongs at `/run/user/1000/systemd/user/bookforge-kiosk.service.d/50-voice-demo.conf`, and its environment file at `/run/user/1000/bookforge-voice-demo.env`. These kiosk files are temporary and must be recreated after a full reboot. Unit changes require daemon reload and the appropriate service restart.

Dependency/model changes need a separate compatible environment restoration; the manifest records package versions but does not bundle their wheels or model files. Existing host environments, model weights and service units are required. API restart resets in-memory scene jobs. Recheck readiness, the served frontend and a user-recorded scene before calling a future rollback verified.

`rollback-smoke.json` records an actual isolated startup of the tagged gateway source using the existing Mac environment on port 18769. Its HTML and both voice scripts matched the Git tag exactly; the temporary gateway was stopped afterward. Production services were not restarted and no image was requested. This verifies the frontend/gateway rollback path, not a from-scratch Jetson restoration or an inference-latency benchmark.
