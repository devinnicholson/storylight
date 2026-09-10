from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
SIGNAL_PATTERN = re.compile(r"lavfi\.signalstats\.([A-Z]+)=([0-9.]+)")
SSIM_PATTERN = re.compile(r"All:([0-9.]+)")


class VisualEvaluationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ProjectionMetrics:
    luma_average: float
    luma_low: float
    luma_high: float
    luma_range: float
    saturation_average: float
    projection_legibility: float


@dataclass(frozen=True, slots=True)
class MotionMetrics:
    duration_seconds: float
    width: int
    height: int
    frames: int
    fps: float
    endpoint_ssim: float
    temporal_change: float
    motion_stability: float


@dataclass(frozen=True, slots=True)
class MediaEvaluation:
    path: str
    kind: str
    sha256: str
    projection: ProjectionMetrics
    motion: MotionMetrics | None


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _require_success(result: subprocess.CompletedProcess[str], tool: str) -> str:
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise VisualEvaluationError(f"{tool} failed: {detail or 'unknown error'}")
    return f"{result.stdout}\n{result.stderr}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _projection_metrics(path: Path, runner: CommandRunner) -> ProjectionMetrics:
    result = runner(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            "signalstats,metadata=print:file=-",
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ]
    )
    output = _require_success(result, "ffmpeg signal analysis")
    values = {name: float(value) for name, value in SIGNAL_PATTERN.findall(output)}
    required = {"YAVG", "YLOW", "YHIGH", "SATAVG"}
    if not required.issubset(values):
        raise VisualEvaluationError("ffmpeg signal analysis omitted required metrics")
    luma_range = values["YHIGH"] - values["YLOW"]
    brightness = 1 - abs(values["YAVG"] - 112) / 112
    contrast = (luma_range - 35) / 150
    legibility = _clamp(0.45 * brightness + 0.55 * contrast)
    return ProjectionMetrics(
        luma_average=round(values["YAVG"], 6),
        luma_low=round(values["YLOW"], 6),
        luma_high=round(values["YHIGH"], 6),
        luma_range=round(luma_range, 6),
        saturation_average=round(values["SATAVG"], 6),
        projection_legibility=round(legibility, 6),
    )


def _probe_video(path: Path, runner: CommandRunner) -> tuple[float, int, int, int, float]:
    result = runner(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames:format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    output = _require_success(result, "ffprobe")
    try:
        payload = json.loads(output)
        stream = payload["streams"][0]
        numerator, denominator = str(stream["avg_frame_rate"]).split("/", 1)
        fps = float(numerator) / float(denominator)
        duration = float(payload["format"]["duration"])
        frames = int(stream.get("nb_frames") or round(duration * fps))
        return duration, int(stream["width"]), int(stream["height"]), frames, fps
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as error:
        raise VisualEvaluationError("ffprobe returned invalid video metadata") from error


def _endpoint_ssim(path: Path, frames: int, runner: CommandRunner) -> float:
    final_index = max(0, frames - 1)
    graph = (
        f"[0:v]trim=start_frame=0:end_frame=1,setpts=PTS-STARTPTS[first];"
        f"[0:v]trim=start_frame={final_index}:end_frame={final_index + 1},"
        "setpts=PTS-STARTPTS[last];[first][last]ssim"
    )
    result = runner(
        ["ffmpeg", "-v", "info", "-i", str(path), "-filter_complex", graph, "-f", "null", "-"]
    )
    output = _require_success(result, "ffmpeg endpoint SSIM")
    matches = SSIM_PATTERN.findall(output)
    if not matches:
        raise VisualEvaluationError("ffmpeg endpoint SSIM omitted its score")
    return _clamp(float(matches[-1]))


def _temporal_change(path: Path, runner: CommandRunner) -> float:
    result = runner(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            "tblend=all_mode=difference,signalstats,metadata=print:file=-",
            "-f",
            "null",
            "-",
        ]
    )
    output = _require_success(result, "ffmpeg temporal analysis")
    differences = [float(value) for name, value in SIGNAL_PATTERN.findall(output) if name == "YAVG"]
    if not differences:
        raise VisualEvaluationError("ffmpeg temporal analysis omitted frame differences")
    return sum(differences) / len(differences) / 255


def evaluate_media(path: Path, *, runner: CommandRunner = _run) -> MediaEvaluation:
    display_path = str(path)
    source = path.resolve()
    if not source.is_file():
        raise VisualEvaluationError(f"media does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm"}:
        raise VisualEvaluationError(f"unsupported media type: {suffix}")
    projection = _projection_metrics(source, runner)
    motion = None
    kind = "image"
    if suffix in {".mp4", ".webm"}:
        kind = "video_loop"
        duration, width, height, frames, fps = _probe_video(source, runner)
        endpoint = _endpoint_ssim(source, frames, runner)
        temporal = _temporal_change(source, runner)
        # Ambient motion should remain visible without becoming flickery. A mean
        # frame delta around 1.5% of luma range is the center of this broad band.
        motion_band = 1 - min(1.0, abs(temporal - 0.015) / 0.04)
        stability = _clamp(0.7 * endpoint + 0.3 * motion_band)
        motion = MotionMetrics(
            duration_seconds=round(duration, 6),
            width=width,
            height=height,
            frames=frames,
            fps=round(fps, 6),
            endpoint_ssim=round(endpoint, 6),
            temporal_change=round(temporal, 6),
            motion_stability=round(stability, 6),
        )
    return MediaEvaluation(
        path=display_path,
        kind=kind,
        sha256=_sha256(source),
        projection=projection,
        motion=motion,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure projection and motion quality")
    parser.add_argument("media", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    evaluations = [asdict(evaluate_media(path)) for path in arguments.media]
    payload = {"schema_version": "1.0", "evaluations": evaluations}
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized)
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
