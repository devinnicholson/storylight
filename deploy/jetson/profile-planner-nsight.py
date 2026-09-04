"""Bounded, synthetic Nsight capture. Run as the Jetson user, never as root.

Temporarily stops the planner and kiosk to fit an instrumented copy in 8 GB.
Restores both services; does not change engines, power mode, or API configuration.
Use a transient user unit with an ExecStopPost restore hook (see docs).
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

PLANNER = "bookforge-tensorrt-planner.service"
KIOSK = "bookforge-kiosk.service"
FALLBACK = "bookforge-gemma.service"
PORT = 11439
ROOT = Path.home() / ".local/share/bookforge/tensorrt-edgellm-v0.10.0"
ENGINE = ROOT / "models/gemma4-e2b-it-int4-awq-v010/engines/llm"
RUNNER = "/opt/bookforge/deploy/jetson/run-tensorrt-edge-server.sh"
NSYS = "/opt/nvidia/nsight-systems/2026.3.1/bin/nsys"
PASSAGES = [
    "A silver fox carries a brass lantern through a moonlit forest.",
    "Exactly two small blue boats drift on a quiet lake beneath a yellow moon.",
    "A child on a stone bridge holds an open green book with both hands. No other people.",
]
SYSTEM = (
    "Extract the visible scene. Return exactly four short lines: "
    "SETTING: place; ACTOR: visible subjects; ACTION: what they do; MAGIC: magical detail or none. "
    "Preserve colors, counts, objects, and negation. Do not invent details."
)


def systemctl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["systemctl", "--user", *args], check=check, timeout=95)


def ready(port: int, timeout: float = 75) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2):
                return
        except (OSError, urllib.error.URLError):
            time.sleep(0.5)
    raise TimeoutError(f"Planner on loopback port {port} did not become ready")


def request(passage: str) -> dict:
    body = {
        "model": "llm",
        "stream": False,
        "temperature": 0,
        "max_tokens": 64,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": passage}],
    }
    started = time.perf_counter()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=25) as response:
        result = json.load(response)
    return {"http_ms": (time.perf_counter() - started) * 1000, "response": result}


def terminate(process: subprocess.Popen, profile: bool = False) -> None:
    if process.poll() is not None:
        return
    if profile:
        process.send_signal(signal.SIGINT)
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def run_window(output: Path, profile: bool) -> dict:
    label = "profiled" if profile else "baseline"
    env = {
        key: os.environ[key]
        for key in ("HOME", "USER", "PATH", "XDG_RUNTIME_DIR")
        if key in os.environ
    }
    env.update(BOOKFORGE_EDGELLM_SERVER_PORT=str(PORT), EDGELLM_GEMMA4_PLE_STORAGE_BACKED="1")
    command = [RUNNER, str(ENGINE)]
    if profile:
        command = [
            NSYS,
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--sample=none",
            "--cpuctxsw=none",
            "--cuda-graph-trace=graph",
            "--duration=100",
            "--kill=sigterm",
            "--force-overwrite=false",
            "--output=" + str(output / "planner"),
            *command,
        ]
    started = time.perf_counter()
    with (output / f"{label}.log").open("w") as log:
        process = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            ready(PORT)
            load_seconds = time.perf_counter() - started
            warmup = request(PASSAGES[0])
            rows = [request(passage) for passage in PASSAGES]
        finally:
            terminate(process, profile)
    return {"load_seconds": load_seconds, "warmup": warmup, "requests": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.geteuid() == 0:
        raise SystemExit("Run as the Jetson user, not root")
    if not Path(NSYS).is_file() or not (ENGINE / "llm.engine").is_file():
        raise SystemExit("Pinned profiler or accepted engine is missing")
    if systemctl("is-active", "--quiet", PLANNER, check=False).returncode:
        raise SystemExit("Resident planner must be healthy before starting a capture")
    # No port takeover and no concurrent profiling copy.
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", PORT))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    kiosk_active = systemctl("is-active", "--quiet", KIOSK, check=False).returncode == 0
    report = {
        "boundary": "synthetic direct HTTP, not production prompt or end-to-end scene latency",
        "passages": PASSAGES,
    }

    def interrupted(*_):
        raise InterruptedError("capture terminated")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        if kiosk_active:
            systemctl("stop", KIOSK)
        systemctl("stop", PLANNER)
        systemctl("stop", FALLBACK)
        report["baseline"] = run_window(args.output, False)
        report["profiled"] = run_window(args.output, True)
    finally:
        (args.output / "measurements.json").write_text(json.dumps(report, indent=2) + "\n")
        try:
            systemctl("start", PLANNER)
            ready(11435)
        finally:
            if kiosk_active:
                systemctl("start", KIOSK)
    traces = list(args.output.glob("*.nsys-rep"))
    if not traces:
        raise SystemExit("No Nsight report produced; do not claim a successful GPU capture")
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in traces}
    (args.output / "trace-checksums.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"restored": True, "traces": manifest, "output": str(args.output)}))


if __name__ == "__main__":
    main()
