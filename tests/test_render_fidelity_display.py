from __future__ import annotations

import json
import stat
import sys
from types import SimpleNamespace

import pytest

from scripts import render_fidelity_display as render


@pytest.fixture
def batch_args(tmp_path, monkeypatch):
    requests, pages = [], []
    for page in range(1, 7):
        steps = [f"page-{page}-step-{step}" for step in range(2 if page >= 5 else 1)]
        for step in steps:
            requests.append(
                dict(
                    id=f"candidate-{step}",
                    source_page_index=page,
                    variant="candidate",
                    display_step_id=step,
                    seed=90400 + page,
                    prompt="Watercolor scene.",
                    negative_prompt="text",
                    prompt_sha256=render.assembly.digest("Watercolor scene."),
                    graph_sha256="a" * 64,
                )
            )
        pages.append(
            dict(
                source_page_index=page,
                display_step_ids=steps,
                baseline_available=page in {3, 5, 6},
                baseline_request_id=f"accepted-page-{page:02}" if page in {3, 5, 6} else None,
            )
        )
    for page in (3, 5, 6):
        requests.append(
            dict(
                id=f"accepted-page-{page:02}",
                source_page_index=page,
                variant="accepted",
                display_step_id=None,
                seed=90400 + page,
                prompt="Watercolor scene.",
                negative_prompt="text",
                prompt_sha256=render.assembly.digest("Watercolor scene."),
                graph_sha256=None,
            )
        )
    batch = dict(
        schema_version=1,
        kind="story-fidelity-visual-batch",
        story_sha256=render.assembly.smoke.MANIFEST_SHA256,
        baseline_summary_sha256=render.assembly.BASELINE_SUMMARY_SHA256,
        requests=requests,
        pages=pages,
        assets_generated=False,
        visual_acceptance_measured=False,
        **{
            field: "b" * 64
            for field in (
                "prompt_sha256",
                "implementation_sha256",
                "story_context_sha256",
                "story_evidence_sha256",
                "story_private_sha256",
                "baseline_context_sha256",
            )
        },
    )
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(batch))
    ledger = tmp_path / "ledger.json"
    ledger.write_text("{}")

    class Tokenizer:
        chat_template = "frozen template"
        backend_tokenizer = SimpleNamespace(to_str=lambda: "frozen vocabulary")

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
            return messages[0]["content"]

        def __call__(self, text):
            return {"input_ids": list(range(256 if text == "Watercolor scene." else 257))}

    def from_pretrained(model, **kwargs):
        assert model == render.MODEL
        assert kwargs == dict(
            revision=render.REVISION,
            subfolder="tokenizer",
            local_files_only=True,
            trust_remote_code=False,
        )
        return Tokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=from_pretrained)),
    )
    args = [
        "--mode",
        "preflight",
        "--batch",
        str(path),
        "--proof-batch-sha256",
        render.assembly.file_hash(path),
        "--ledger",
        str(ledger),
        "--output",
        str(tmp_path / "run"),
    ]
    return args, batch, path


def test_preflight_is_offline_and_all_contract_failures_precede_billing(
    batch_args, monkeypatch, tmp_path
):
    args, batch, path = batch_args
    monkeypatch.setattr(
        render, "KleinSceneProvider", lambda **kwargs: pytest.fail("preflight allocated provider")
    )
    assert render.main(args) == 0
    journal = tmp_path / "run/journal.jsonl"
    assert stat.S_IMODE(journal.stat().st_mode) == 0o600
    assert stat.S_IMODE(journal.parent.stat().st_mode) == 0o700
    args[-1] = str(tmp_path / "invalid")
    original = json.loads(path.read_text())
    for mutation in ("hash", "seed", "baseline", "duplicate", "oversize"):
        batch = json.loads(json.dumps(original))
        if mutation == "seed":
            batch["requests"][0]["seed"] += 1
        elif mutation == "baseline":
            batch["baseline_summary_sha256"] = "0" * 64
        elif mutation == "duplicate":
            batch["requests"][1]["id"] = batch["requests"][0]["id"]
        elif mutation == "oversize":
            batch["requests"][-1]["prompt"] = "Oversized token sequence"
            batch["requests"][-1]["prompt_sha256"] = render.assembly.digest(
                batch["requests"][-1]["prompt"]
            )
        path.write_text(json.dumps(batch))
        args[5] = "0" * 64 if mutation == "hash" else render.assembly.file_hash(path)
        assert render.main(args) == 1
        assert not (tmp_path / "invalid").exists()
    path.write_text(json.dumps(original))
    args[5] = render.assembly.file_hash(path)
    monkeypatch.setattr(
        render, "load_tokenizer", lambda: (_ for _ in ()).throw(OSError("private cache path"))
    )
    assert render.main(args) == 1
    assert not (tmp_path / "invalid").exists()


def install_provider(monkeypatch, fail_at=None, wrong_identity=False):
    calls = []

    class Provider:
        def __init__(self, **kwargs):
            assert kwargs["session_gpu_cap_usd"] == 2.75
            assert kwargs["plan_file"].name == "overnight-modal-plan.json"
            assert kwargs["ledger_path"].name == "ledger.json"

        async def generate_fast(self, request, *, output_dir):
            calls.append(request)
            assert (request.width, request.height, request.steps, request.guidance_scale) == (
                1024,
                576,
                4,
                1.0,
            )
            assert request.fidelity_label == ""
            if len(calls) == fail_at:
                raise TimeoutError("private remote failure detail")
            output_dir.mkdir()
            manifest_path = output_dir / "scene.manifest.json"
            manifest_path.write_text("{}")
            identity = {} if wrong_identity else render.expected_identity()
            return SimpleNamespace(
                manifest_path=manifest_path,
                manifest={
                    "identity": identity,
                    "policy": {"reservation_id": "local-reservation"},
                    "stages": {
                        "fast": {
                            **{
                                key: 1.0
                                for key in (
                                    "remote_seconds",
                                    "image_seconds",
                                    "depth_seconds",
                                    "packaging_seconds",
                                    "model_load_seconds",
                                    "startup_seconds",
                                    "cache_setup_seconds",
                                )
                            },
                            "warm_state": "cold" if len(calls) == 1 else "warm",
                            "sequence_bucket": 256,
                            "token_count": 256,
                        }
                    },
                },
                artifacts={role: SimpleNamespace(sha256="c" * 64) for role in ("master", "depth")},
            )

    monkeypatch.setattr(render, "KleinSceneProvider", Provider)
    return calls


def test_execution_has_exactly_eleven_attempts_no_prewarm_and_no_restart(
    batch_args, monkeypatch, tmp_path
):
    args, _, _ = batch_args
    args[1] = "execute"
    calls = install_provider(monkeypatch)
    assert render.main(args) == 0
    assert len(calls) == len({call.scene_id for call in calls}) == 11
    assert [call.seed for call in calls] == [
        90401,
        90402,
        90404,
        90403,
        90403,
        90405,
        90405,
        90405,
        90406,
        90406,
        90406,
    ]
    journal = (tmp_path / "run/journal.jsonl").read_text()
    assert "Watercolor scene." not in journal and "local-reservation" not in journal
    entries = [json.loads(line) for line in journal.splitlines()]
    assert len([row for row in entries if row["kind"] == "start"]) == 11
    assert entries[-1]["kind"] == "complete"
    assert render.main(args) == 1
    assert len(calls) == 11
    args[-1] = str(tmp_path / "restart-with-new-output")
    assert render.main(args) == 1
    assert len(calls) == 11
    claims = list(tmp_path.glob("ledger.json.display-*.attempt.json"))
    assert len(claims) == 1 and stat.S_IMODE(claims[0].stat().st_mode) == 0o600


def test_ambiguous_call_or_identity_mismatch_stops_without_retry(
    batch_args, monkeypatch, tmp_path, capsys
):
    args, batch, batch_path = batch_args
    args[1] = "execute"
    calls = install_provider(monkeypatch, fail_at=3)
    assert render.main(args) == 1
    assert render.main(args) == 1
    assert len(calls) == 3
    journal = (tmp_path / "run/journal.jsonl").read_text()
    assert json.loads(journal.splitlines()[-1])["status"] == "failed_or_unknown"
    assert "private remote failure detail" not in journal + capsys.readouterr().out
    args[-1] = str(tmp_path / "retry-with-new-output")
    assert render.main(args) == 1 and len(calls) == 3
    calls = install_provider(monkeypatch, wrong_identity=True)
    batch["requests"][0]["negative_prompt"] = "captions"
    batch_path.write_text(json.dumps(batch))
    args[5] = render.assembly.file_hash(batch_path)
    args[-1] = str(tmp_path / "identity")
    assert render.main(args) == 1
    assert len(calls) == 1
