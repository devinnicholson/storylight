import json
import subprocess
from pathlib import Path

import pytest

from bookforge.visual_evaluation import VisualEvaluationError, evaluate_media


class FakeRunner:
    def __call__(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        tool = command[0]
        if tool == "ffprobe":
            payload = {
                "streams": [
                    {
                        "width": 768,
                        "height": 512,
                        "avg_frame_rate": "24/1",
                        "nb_frames": "96",
                    }
                ],
                "format": {"duration": "4.0"},
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        if "ssim" in " ".join(command):
            return subprocess.CompletedProcess(command, 0, "", "SSIM All:0.98231")
        if "tblend" in " ".join(command):
            rows = "\n".join(
                [
                    "lavfi.signalstats.YAVG=3.2",
                    "lavfi.signalstats.YAVG=4.1",
                    "lavfi.signalstats.YAVG=3.7",
                ]
            )
            return subprocess.CompletedProcess(command, 0, rows, "")
        rows = "\n".join(
            [
                "lavfi.signalstats.YAVG=110",
                "lavfi.signalstats.YLOW=20",
                "lavfi.signalstats.YHIGH=220",
                "lavfi.signalstats.SATAVG=72",
            ]
        )
        return subprocess.CompletedProcess(command, 0, rows, "")


def test_evaluates_motion_loop_metadata_and_stability(tmp_path: Path) -> None:
    video = tmp_path / "scene.mp4"
    video.write_bytes(b"candidate-video")

    evaluation = evaluate_media(video, runner=FakeRunner())

    assert evaluation.kind == "video_loop"
    assert evaluation.motion is not None
    assert evaluation.motion.duration_seconds == 4
    assert evaluation.motion.frames == 96
    assert evaluation.motion.fps == 24
    assert evaluation.motion.endpoint_ssim == pytest.approx(0.98231)
    assert evaluation.motion.motion_stability > 0.95


def test_fails_closed_when_tool_output_is_incomplete(tmp_path: Path) -> None:
    image = tmp_path / "scene.png"
    image.write_bytes(b"candidate")

    def incomplete(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, "lavfi.signalstats.YAVG=100", "")

    with pytest.raises(VisualEvaluationError, match="omitted required"):
        evaluate_media(image, runner=incomplete)
