"""Compare isolated HTTP service startup with and without ASR preparation."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PRIMER = ROOT / "experiments/voice-asr-startup/abba/primer.webm"
TARGET = ROOT / "benchmarks/voice-scheduling-2026-09-08/browser/short-candidate/asr-01.webm"
URL = "http://127.0.0.1:18769"


def request(path, audio=None):
    req = urllib.request.Request(URL + path, data=audio, headers={"Content-Type": "audio/webm"})
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.load(response)


def main():
    output = Path(sys.argv[1])
    output.mkdir(exist_ok=False)
    records = []
    for index, prepared in enumerate((False, True, True, False)):
        with tempfile.TemporaryDirectory(prefix="storylight-asr-service-") as directory:
            env = {k: v for k, v in os.environ.items() if not k.startswith("STORYLIGHT_")}
            env.update(
                STORYLIGHT_MODEL_BACKEND="fake",
                STORYLIGHT_ASSET_BACKEND="disabled",
                STORYLIGHT_LIVE_SCENE_BACKEND="disabled",
                STORYLIGHT_ASR_BACKEND="mlx_whisper",
                STORYLIGHT_ASR_MODEL=str(ROOT / ".models/whisper-small.en"),
                STORYLIGHT_ASR_STARTUP_AUDIO=str(PRIMER) if prepared else "",
                STORYLIGHT_DATA_DIR=directory,
                STORYLIGHT_CACHE_DIR=directory + "/cache",
                PYTHONPATH=str(ROOT / "src"),
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
            )
            started = time.perf_counter()
            with (output / f"{index}.log").open("w") as log:
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "storylight.api:app",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        "18769",
                    ],
                    cwd=directory,
                    env=env,
                    stdout=log,
                    stderr=log,
                )
                try:
                    deadline = started + 30
                    while True:
                        if child.poll() is not None:
                            raise RuntimeError("Service exited during startup")
                        try:
                            status = request("/v1/runtime:status")
                            break
                        except urllib.error.URLError:
                            if time.perf_counter() > deadline:
                                raise TimeoutError("Service startup deadline") from None
                            time.sleep(0.05)
                    ready_ms = (time.perf_counter() - started) * 1000
                    before = time.perf_counter()
                    result = request("/v1/audio:transcribe", TARGET.read_bytes())
                    elapsed_ms = (time.perf_counter() - before) * 1000
                    assert result["text"] == "A cat chasing a mouse.", result
                    assert ("prepared in" in status["asr"]["detail"]) == prepared
                    records.append(
                        dict(
                            prepared=prepared,
                            startup_ready_ms=ready_ms,
                            request_ms=elapsed_ms,
                            asr=result,
                            status=status["asr"],
                        )
                    )
                finally:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=5)
    pins = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (
            Path(__file__),
            PRIMER,
            TARGET,
            ROOT / "src/storylight/asr.py",
            ROOT / "src/storylight/api.py",
            ROOT / "src/storylight/config.py",
        )
    }
    (output / "results.json").write_text(json.dumps(dict(pins=pins, runs=records), indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
