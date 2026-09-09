"""Four fresh, sequential offline ASR processes; no running-service changes."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = ROOT / ".models/whisper-small.en"
TARGET = ROOT / "benchmarks/voice-scheduling-2026-09-08/browser/short-candidate/asr-01.webm"
PRIMER_SOURCE = ROOT / "benchmarks/local-asr-2026-09-07/chasing/corpus/samantha-2-clean.wav"
EXPECTED = "A cat chasing a mouse."
PINS = {
    MODEL / "weights.npz": "1bb29b030aca711a035f7a084a0eefac6251ecc2bdd356fa748858fbad082f5a",
    MODEL / "config.json": "40028e43687458e79ba89cc16bf14184c5414a5cbc00fd511f8f78d138647afd",
    TARGET: "54c6702bdac9ec9dd5e65aac00a8bcb8217c7dca15233635d9322fd4db31eed5",
    PRIMER_SOURCE: "f13a81e0611d2526a992096f3150e4338d14d09a324dd792a41de07018da659d",
    ROOT / "src/bookforge/asr.py":
        "318b9f0c89b0ad0b6231df2bc32486a63e1bd7984646b04ace97ec0a54d4765d",
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


async def worker(arm, directory):
    def deny_network(event, _args):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise RuntimeError("network forbidden in offline ASR experiment")

    sys.addaudithook(deny_network)
    started = time.perf_counter()
    from bookforge.asr import LocalTranscriber
    from bookforge.config import Settings

    backend = LocalTranscriber(Settings(
        _env_file=None, asr_backend="mlx_whisper", asr_model=str(MODEL),
    ))
    setup_seconds = time.perf_counter() - started
    records = []
    calls = [("target", TARGET)] if arm == "A" else [
        ("primer", directory / "primer.webm"), ("target", TARGET),
    ]
    for role, path in calls:
        audio = path.read_bytes()
        module = sys.modules.get("mlx_whisper.transcribe")
        loaded = module is not None and module.ModelHolder.model is not None
        before = time.perf_counter()
        response = await backend.transcribe(audio, "audio/webm;codecs=opus")
        records.append({
            "role": role, "audio_sha256": sha(path), "audio_bytes": len(audio),
            "model_loaded_before": loaded, "total_ms": response.total_ms,
            "call_wall_seconds": time.perf_counter() - before,
            "text": response.text, "language": response.language,
        })
    return {
        "arm": arm, "pid": os.getpid(), "records": records,
        "setup_seconds": setup_seconds,
        "worker_wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "queue_contenders": 0, "network_forbidden": True,
    }


def reap(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2)


def run(directory):
    if platform.system() != "Darwin":
        raise ValueError("This fixed experiment requires the original Mac runtime")
    for path, expected in PINS.items():
        if sha(path) != expected:
            raise ValueError(f"input pin mismatch: {path.name}")
    directory.mkdir(parents=True, exist_ok=False)
    ffmpeg = "/opt/homebrew/bin/ffmpeg"
    conversion = [ffmpeg, "-nostdin", "-v", "error", "-i", str(PRIMER_SOURCE),
                  "-ac", "1", "-ar", "48000", "-c:a", "libopus", "-b:a", "64k",
                  str(directory / "primer.webm")]
    subprocess.run(conversion, check=True, timeout=10, capture_output=True)
    package = Path(importlib.metadata.distribution("mlx-whisper").locate_file("mlx_whisper"))
    proof = {
        "schema_version": 1, "order": ["A", "B", "B", "A"],
        "A": "first target in fresh process", "B": "one synthetic primer then exact target",
        "harness_sha256": sha(Path(__file__)),
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-whisper")},
        "inputs": {str(path.relative_to(ROOT)): expected for path, expected in PINS.items()},
        "mlx_whisper_sources": {name: sha(package / name) for name in
                                ("transcribe.py", "load_models.py", "audio.py", "decoding.py")},
        "primer_conversion": conversion, "ffmpeg_sha256": sha(Path(ffmpeg)),
        "primer_sha256": sha(directory / "primer.webm"),
        "primer_reference": "The quick brown fox jumps over the lazy dog.",
        "target_reference": EXPECTED,
        "decode_settings": {
            "language": "en", "condition_on_previous_text": False,
            "logprob_threshold": None, "verbose": None,
            "all_other_options": "unchanged production/package defaults",
        },
        "limits": {"overall_seconds": 60, "per_process_seconds": 15,
                   "termination_grace_seconds": 2, "kill_join_seconds": 2},
        "no_http_or_live_service_calls": True, "downloads_forbidden": True,
        "fresh_process_not_cold_host_or_metal_cache": True,
    }
    write(directory / "protocol.json", proof)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "HF_DATASETS_OFFLINE": "1", "PYTHONPATH": str(ROOT / "src")}
    started = time.monotonic()
    results = []
    for index, arm in enumerate(proof["order"]):
        timeout = min(15, 60 - (time.monotonic() - started))
        if timeout <= 0:
            raise TimeoutError("overall experiment deadline")
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", arm,
                   "--output", str(directory)]
        before = time.perf_counter()
        process = subprocess.Popen(command, env=env, start_new_session=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        failure = None
        stdout, stderr = b"", b""
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except BaseException as error:
            failure = type(error).__name__
            raise
        finally:
            reap(process)
            write(directory / f"{index + 1}-{arm}-process.json", {
                "pid": process.pid, "exit_code": process.returncode,
                "process_wall_seconds": time.perf_counter() - before,
                "reaped": process.poll() is not None, "failure_type": failure,
                "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            })
            (directory / f"{index + 1}-{arm}.stderr").write_bytes(stderr)
        if process.returncode:
            raise RuntimeError("ASR worker failed; no further runs")
        row = json.loads(stdout)
        write(directory / f"{index + 1}-{arm}.json", row)
        results.append(row)
        print(json.dumps({"arm": arm, "records": row["records"]}), flush=True)
    summary = {"protocol_sha256": sha(directory / "protocol.json"),
               "overall_wall_seconds": time.monotonic() - started, "runs": results}
    for arm in ("A", "B"):
        selected = [row for row in results if row["arm"] == arm]
        targets = [row["records"][-1] for row in selected]
        summary[arm] = {
            "target_median_ms": statistics.median(row["total_ms"] for row in targets),
            "all_call_median_ms": statistics.median(
                sum(call["total_ms"] for call in row["records"]) for row in selected),
            "target_exact_count": sum(row["text"] == EXPECTED for row in targets),
        }
    write(directory / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("A", "B"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(asyncio.run(worker(args.worker, args.output.resolve()))))
    else:
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        run(args.output.resolve())
