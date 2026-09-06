import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
import klein_hardware_probe as probe  # noqa: E402


def test_fixed_hardware_schedule_stops_and_retains_failure_evidence(capsys):
    images = (b"master", b"depth")
    cases = [
        {
            "prompt": f"private synthetic passage {index}",
            "seed": index,
            "expected_bucket": bucket,
            **{f"{name}_sha256": hashlib.sha256(data).hexdigest()
               for name, data in zip(("master", "depth"), images, strict=True)},
        }
        for index, bucket in enumerate((128, 256))
    ]
    calls, failure = [], None

    def render(prompt, seed):
        calls.append((prompt, seed))
        if failure == "render" and len(calls) == 4:
            raise RuntimeError("private runtime failure details")
        data = (b"changed master", b"changed depth") if failure == "historical" else images
        return ({
            "seed": seed,
            "sequence_bucket": cases[seed]["expected_bucket"],
            **{f"{name}_sha256": ("0" * 64 if failure == "hash"
                                  else hashlib.sha256(content).hexdigest())
               for name, content in zip(("master", "depth"), data, strict=True)},
        }, *data)

    runtime = SimpleNamespace(render=render)
    result = probe.run_comparison(runtime, cases)
    assert result["status"] == "complete" and result["failure_stage"] is None
    assert calls == [(cases[i]["prompt"], i) for i in (0, 1) * 5]
    assert [(r["phase"], r["case_index"]) for r in result["records"]] == list(probe.SCHEDULE)
    assert [r["ordinal"] for r in result["records"]] == list(range(10))
    assert all((r["master"], r["depth"]) == images for r in result["records"])
    assert all(r["historical_images_exact"] for r in result["records"])
    failure = "historical"
    calls.clear()
    result = probe.run_comparison(runtime, cases)
    assert result["status"] == "complete" and len(calls) == len(result["records"]) == 10
    assert all(not r["historical_images_exact"] for r in result["records"])
    assert all((r["master"], r["depth"]) == (b"changed master", b"changed depth")
               for r in result["records"])
    for failure, stage, count, retained in (
        ("render", "render", 4, 3), ("hash", "verification", 1, 1)
    ):
        calls.clear()
        result = probe.run_comparison(runtime, cases)
        assert result["status"] == "failed" and result["failure_stage"] == stage
        assert len(calls) == count and len(result["records"]) == retained
        if failure == "hash":
            assert result["records"][0]["master"] == images[0]
    calls.clear()
    assert probe.run_comparison(runtime, cases[:1])["failure_stage"] == "setup"
    assert calls == []
    output = capsys.readouterr().out
    assert "private" not in output
    events = [json.loads(line)["hardware_progress"] for line in output.splitlines()]
    assert {row["exception_type"] for row in events if row["state"] == "failed"} == {
        "RuntimeError", "ValueError"
    }
