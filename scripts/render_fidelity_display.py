#!/usr/bin/env python3
"""Preflight or render the proved eleven-image display batch once, without prewarming."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from bookforge.finite_modal_provider import FastSceneRequest
from bookforge.klein_scene_provider import MODEL, REVISION, KleinSceneProvider

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import assemble_fidelity_display as assembly  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
Digest = assembly.probe.benchmark.Digest
StrictModel = assembly.probe.benchmark.StrictModel
Identifier = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$")]
RECEIPT = ROOT / "benchmarks/overnight-20260904/renderer-summary.json"
RECEIPT_SHA256 = "7662c5fdd1cad0c79ab2f0b87e66d0251f14d02a8817ec31df78a5a8153b7e9b"
EXPECTED_RUNTIME_SHA256 = "f87ab40c8b6a1ed457a10f1bce6bf92e07075eed40c0fdb224ba3ed18c5c3441"
EXPECTED_DEPLOYMENT_SHA256 = "a4d96ea58f900a677e3ae8bca0817ace34be518c8bd3668e7f3f39426175d1c1"
CACHE_ID = "f305950a0fbb4ecf89acfb80a3990351"


class RenderRow(StrictModel):
    id: Identifier
    source_page_index: Annotated[int, Field(ge=1, le=6)]
    variant: Literal["candidate", "accepted"]
    display_step_id: Identifier | None
    seed: int
    prompt: Annotated[str, Field(min_length=1, max_length=4000)]
    negative_prompt: Annotated[str, Field(min_length=1, max_length=4000)]
    prompt_sha256: Digest
    graph_sha256: Digest | None


class BatchPage(StrictModel):
    source_page_index: Annotated[int, Field(ge=1, le=6)]
    display_step_ids: list[Identifier]
    baseline_available: bool
    baseline_request_id: Identifier | None


class RenderBatch(StrictModel):
    schema_version: Literal[1]
    kind: Literal["story-fidelity-visual-batch"]
    story_sha256: Digest
    prompt_sha256: Digest
    implementation_sha256: Digest
    story_context_sha256: Digest
    story_evidence_sha256: Digest
    story_private_sha256: Digest
    baseline_summary_sha256: Digest
    baseline_context_sha256: Digest
    requests: Annotated[list[RenderRow], Field(min_length=11, max_length=11)]
    pages: Annotated[list[BatchPage], Field(min_length=6, max_length=6)]
    assets_generated: Literal[False]
    visual_acceptance_measured: Literal[False]


def read_batch(path: Path, proof_sha256: str) -> RenderBatch:
    raw = path.read_bytes()
    if len(raw) > 131072 or hashlib.sha256(raw).hexdigest() != proof_sha256:
        raise ValueError("batch proof differs")
    batch = RenderBatch.model_validate_json(raw)
    if (
        batch.story_sha256 != assembly.smoke.MANIFEST_SHA256
        or batch.baseline_summary_sha256 != assembly.BASELINE_SUMMARY_SHA256
        or len({row.id for row in batch.requests}) != 11
    ):
        raise ValueError("batch is not the frozen demonstration")
    _, story = assembly.smoke.load_manifest(assembly.probe.STORY)
    candidates = [row for row in batch.requests if row.variant == "candidate"]
    accepted = [row for row in batch.requests if row.variant == "accepted"]
    if (
        Counter(row.source_page_index for row in candidates)
        != Counter({1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2})
        or sorted(row.source_page_index for row in accepted) != [3, 5, 6]
        or len({row.display_step_id for row in candidates}) != 8
    ):
        raise ValueError("batch selection differs")
    for row in batch.requests:
        if row.seed != story[row.source_page_index - 1][1] or row.prompt_sha256 != assembly.digest(
            row.prompt
        ):
            raise ValueError("request identity differs")
        if row.variant == "candidate":
            if (
                row.display_step_id is None
                or row.graph_sha256 is None
                or row.id != f"candidate-{row.display_step_id}"
            ):
                raise ValueError("candidate lacks graph identity")
        elif (
            row.display_step_id is not None
            or row.graph_sha256 is not None
            or row.id != f"accepted-page-{row.source_page_index:02}"
        ):
            raise ValueError("baseline identity differs")
    for index, page in enumerate(batch.pages, start=1):
        steps = [row.display_step_id for row in candidates if row.source_page_index == index]
        if page.model_dump() != {
            "source_page_index": index,
            "display_step_ids": steps,
            "baseline_available": index in {3, 5, 6},
            "baseline_request_id": f"accepted-page-{index:02}" if index in {3, 5, 6} else None,
        }:
            raise ValueError("display sequence differs")
    return batch


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        MODEL,
        revision=REVISION,
        subfolder="tokenizer",
        local_files_only=True,
        trust_remote_code=False,
    )


def token_preflight(batch: RenderBatch) -> dict:
    tokenizer = load_tokenizer()
    counts = {}
    for row in batch.requests:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": row.prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        count = len(tokenizer(text)["input_ids"])
        if not 0 < count <= 256:
            raise ValueError("request exceeds qualified token profile")
        counts[row.id] = count
    return {
        "tokenizer_sha256": assembly.digest(
            {
                "backend": tokenizer.backend_tokenizer.to_str(),
                "chat_template": tokenizer.chat_template,
            }
        ),
        "token_counts": counts,
    }


def ordered_requests(batch: RenderBatch) -> list[RenderRow]:
    # Unpaired scenes absorb initial startup; paired groups alternate their first variant.
    groups = [
        (1, "candidate"),
        (2, "candidate"),
        (4, "candidate"),
        (3, "candidate"),
        (3, "accepted"),
        (5, "accepted"),
        (5, "candidate"),
        (6, "candidate"),
        (6, "accepted"),
    ]
    return [
        row
        for page, variant in groups
        for row in batch.requests
        if row.source_page_index == page and row.variant == variant
    ]


def append(path: Path, row: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def claim_batch(args, journal: Path) -> str:
    ledger = args.ledger.resolve(strict=True)
    path = ledger.with_name(f"{ledger.name}.display-{args.proof_batch_sha256}.attempt.json")
    claim = {
        "kind": "display_batch_attempt",
        "batch_sha256": args.proof_batch_sha256,
        "ledger_path_sha256": assembly.digest(str(args.ledger.resolve())),
        "journal_path_sha256": assembly.digest(str(journal.resolve())),
        "harness_sha256": assembly.file_hash(Path(__file__)),
    }
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(json.dumps(claim, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return assembly.digest(claim)


def fast_request(row: RenderRow) -> FastSceneRequest:
    return FastSceneRequest(
        scene_id=row.id,
        prompt=row.prompt,
        negative_prompt=row.negative_prompt,
        seed=row.seed,
        width=1024,
        height=576,
        steps=4,
        guidance_scale=1.0,
    )


def expected_identity() -> dict:
    if (
        assembly.file_hash(RECEIPT) != RECEIPT_SHA256
        or assembly.file_hash(ROOT / "deploy/modal_klein_scene.py") != EXPECTED_DEPLOYMENT_SHA256
        or assembly.file_hash(ROOT / "deploy/klein_scene_runtime.py") != EXPECTED_RUNTIME_SHA256
    ):
        raise ValueError("renderer receipt differs")
    return json.loads(RECEIPT.read_text())["batches"]["development"]["identity"]


async def execute(batch: RenderBatch, args, journal: Path, identity: dict, tokens: dict) -> None:
    provider = KleinSceneProvider(
        plan_file=args.plan,
        ledger_path=args.ledger,
        modal_executable=args.modal_executable,
        session_gpu_cap_usd=2.75,
    )
    for ordinal, row in enumerate(ordered_requests(batch)):
        request = fast_request(row)
        append(
            journal,
            {
                "kind": "start",
                "ordinal": ordinal,
                "request_sha256": assembly.digest(row.model_dump()),
            },
        )
        try:
            bundle = await provider.generate_fast(
                request, output_dir=args.output / f"image-{ordinal:02}"
            )
            manifest = bundle.manifest
            if manifest["identity"] != identity:
                raise ValueError("runtime receipt differs")
            stage = manifest["stages"]["fast"]
            timing_fields = (
                "remote_seconds",
                "image_seconds",
                "depth_seconds",
                "packaging_seconds",
                "model_load_seconds",
                "startup_seconds",
                "cache_setup_seconds",
            )
            if (
                any(
                    type(stage[key]) not in {int, float}
                    or not math.isfinite(stage[key])
                    or stage[key] < 0
                    for key in timing_fields
                )
                or stage["warm_state"] not in {"cold", "warm"}
                or stage["token_count"] != tokens["token_counts"][row.id]
                or stage["sequence_bucket"] != (128 if stage["token_count"] <= 128 else 256)
            ):
                raise ValueError("renderer measurements differ")
            append(
                journal,
                {
                    "kind": "result",
                    "ordinal": ordinal,
                    "status": "ok",
                    "manifest_sha256": assembly.file_hash(bundle.manifest_path),
                    "identity_sha256": assembly.digest(manifest["identity"]),
                    "reservation_sha256": assembly.digest(manifest["policy"]["reservation_id"]),
                    "master_sha256": bundle.artifacts["master"].sha256,
                    "depth_sha256": bundle.artifacts["depth"].sha256,
                    "metrics": {
                        key: stage[key]
                        for key in (
                            "remote_seconds",
                            "image_seconds",
                            "depth_seconds",
                            "packaging_seconds",
                            "model_load_seconds",
                            "startup_seconds",
                            "cache_setup_seconds",
                            "warm_state",
                            "sequence_bucket",
                            "token_count",
                        )
                    },
                },
            )
        except BaseException:
            append(journal, {"kind": "result", "ordinal": ordinal, "status": "failed_or_unknown"})
            raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "execute"), required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--proof-batch-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "experiments/renderer-fidelity/overnight-modal-plan.json",
    )
    parser.add_argument(
        "--ledger", type=Path, default=ROOT / ".bookforge/overnight-candidate/modal-ledger.json"
    )
    parser.add_argument("--modal-executable", default=str(ROOT / ".venv/bin/modal"))
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("output already exists; retries are forbidden")
        batch = read_batch(args.batch, args.proof_batch_sha256)
        for row in batch.requests:
            fast_request(row)
        identity = expected_identity()
        tokens = token_preflight(batch)
        header = {
            "kind": "header",
            "schema_version": 1,
            "mode": args.mode,
            "batch_sha256": args.proof_batch_sha256,
            "harness_sha256": assembly.file_hash(Path(__file__)),
            "provider_sha256": assembly.file_hash(ROOT / "src/bookforge/klein_scene_provider.py"),
            "plan_sha256": assembly.file_hash(args.plan),
            "ledger_before_sha256": assembly.file_hash(args.ledger),
            "ledger_path_sha256": assembly.digest(str(args.ledger.resolve())),
            "runtime_sha256": EXPECTED_RUNTIME_SHA256,
            "deployment_source_sha256": EXPECTED_DEPLOYMENT_SHA256,
            "renderer_receipt_sha256": RECEIPT_SHA256,
            "expected_identity_sha256": assembly.digest(identity),
            "cache_id_sha256": assembly.digest(CACHE_ID),
            "maximum_requests": 11,
            "call_ceiling_usd": 0.25,
            "batch_ceiling_usd": 2.75,
            "automatic_retries": 0,
            "prewarm_requests": 0,
            "negative_prompt_supported": False,
            "visual_acceptance_measured": False,
            "tokens": tokens,
        }
        args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
        journal = args.output / "journal.jsonl"
        descriptor = os.open(journal, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        if args.mode == "execute":
            header["attempt_claim_sha256"] = claim_batch(args, journal)
        append(journal, header)
        if args.mode == "execute":
            asyncio.run(execute(batch, args, journal, identity, tokens))
        append(
            journal,
            {
                "kind": "complete",
                "mode": args.mode,
                "ledger_after_sha256": assembly.file_hash(args.ledger),
            },
        )
        return 0
    except Exception:
        print("display render stopped; inspect the local sanitized journal")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
