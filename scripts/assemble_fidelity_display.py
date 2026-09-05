#!/usr/bin/env python3
"""Assemble the frozen demonstration from bound local captures, without inference or rendering."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

from bookforge.live_scene_planner import validate_live_scene_plan_privacy
from bookforge.scene_playback import DisplaySourcePage, build_display_story_pack
from bookforge.tensorrt_slot_client import tensor_accepted_graph_wire_plan, tensor_slot_wire_plan

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_scene_prompt_probe as probe  # noqa: E402
from scripts import benchmark_story_fidelity_smoke as smoke  # noqa: E402

BASELINE_SUMMARY_SHA256 = "cf08ff06d1836a6493efceeb8013c1f9c2859e3aef6608e2b6eb9a54656d092c"
BASELINE_REVISION = "c2104698b52240a398335f3a4cd5668df231e6a4"
BASELINE_INDICES = {2, 4, 5}
digest = probe.benchmark.digest


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_private(path: Path) -> list[dict]:
    smoke.private_path(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size > 131072
        ):
            raise ValueError("invalid private capture")
        return [json.loads(line) for line in stream.read().splitlines()]


def story_graphs(args, manifest: dict, cases):
    header = json.loads(args.story_evidence.read_text().splitlines()[0])
    provenance = probe.benchmark.Provenance.model_validate_json(args.provenance.read_text())
    expected = probe.context(
        argparse.Namespace(
            model=args.model,
            endpoint=args.endpoint,
            memory_pid=header.get("memory_pid"),
            timeout=header.get("timeout_seconds"),
        ),
        provenance,
        "story",
    )
    started, results = probe.load_evidence(args.story_evidence, expected)
    if probe.aggregate(expected, started, results)["decision"] != "construction_passed":
        raise ValueError("story capture did not pass all seven constructions")
    entries = read_private(args.story_private)
    if not entries or entries[0] != {"kind": "private_header", "context": expected}:
        raise ValueError("story capture context differs")
    rows = {row.index: row for row in results}
    graphs, seen = {}, set()
    for entry in entries[1:]:
        if set(entry) != {
            "kind",
            "index",
            "source_sha256",
            "raw",
            "raw_sha256",
            "generation_complete",
        }:
            raise ValueError("invalid story capture fields")
        index = entry["index"]
        if type(index) is not int or index not in rows or index in seen:
            raise ValueError("invalid story capture index")
        seen.add(index)
        row, raw = rows[index], entry["raw"]
        source, seed = cases[index]
        if (
            entry["kind"] != "response"
            or not isinstance(raw, str)
            or len(raw) > 16384
            or entry["generation_complete"] is not True
            or not row.generation_complete
            or row.status != "ok"
            or entry["source_sha256"] != digest(source)
            or entry["raw_sha256"] != digest(raw)
            or digest(raw) != row.raw_sha256
            or not (row.candidate_valid and row.graph_valid and row.compiler_proof)
            or row.candidate_privacy_pass is not True
        ):
            raise ValueError("story capture differs from proved response")
        wire = tensor_accepted_graph_wire_plan(raw, source_text=source, scope="scene")
        facts = wire.scene_facts
        plan = wire.to_live_scene_plan(context_text=source)
        validate_live_scene_plan_privacy(plan, source_text=source)
        page = plan.to_page(source_text=source, visual_style=manifest["visual_style"], seed=seed)
        if (
            facts is None
            or digest(facts.model_dump(mode="json")) != row.graph_sha256
            or page.scene_spec.master_prompt
            != facts.to_renderer_prompt(source_text=source, visual_style=manifest["visual_style"])
            or digest(page.scene_spec.master_prompt) != row.candidate_sha256
        ):
            raise ValueError("reconstructed graph differs from captured proof")
        graphs[index] = facts
    if seen != set(range(7)):
        raise ValueError("story capture requires all seven responses")
    return header, graphs


def phase_selection(index: int, facts):
    if index == 4:
        actors = [
            actor
            for actor in facts.subjects
            if actor.label == "fox"
            and actor.color == "silver"
            and actor.actions == ("lifts white feather",)
        ]
        if len(actors) != 1 or facts.transformation is None:
            raise ValueError("frozen transformation action differs")
        return {"before": [f"action:{actors[0].ref}:0"], "after": []}
    if index == 5:
        birds = [
            actor
            for actor in facts.subjects
            if actor.label == "birds"
            and actor.color == "golden"
            and actor.count == 3
            and actor.actions == ("fly",)
        ]
        if len(birds) != 1 or not facts.temporal_order:
            raise ValueError("frozen ordered flight differs")
        return {"first": [], "then": [f"action:{birds[0].ref}:0"]}
    if facts.transformation is not None or facts.temporal_order:
        raise ValueError("unexpected temporal source page")
    return None


def accepted_pages(args, manifest: dict, cases):
    if file_hash(args.baseline_summary) != BASELINE_SUMMARY_SHA256:
        raise ValueError("accepted summary differs from frozen baseline")
    summary = json.loads(args.baseline_summary.read_text())
    original = summary["context"]
    provenance = probe.benchmark.Provenance.model_validate_json(
        args.baseline_provenance.read_text()
    )
    if (
        provenance.code_revision != BASELINE_REVISION
        or provenance.model_dump() != original["provenance"]
        or summary["context_sha256"] != digest(original)
    ):
        raise ValueError("accepted baseline provenance differs")
    header = smoke.context(
        argparse.Namespace(
            model=args.baseline_model or args.model,
            endpoint=args.baseline_endpoint or args.endpoint,
            planning_scope="focal",
            max_output_tokens=64,
            timeout=original["timeout_seconds"],
            memory_pid=original["memory_pid"],
        ),
        provenance,
    )
    replay_context, entries = smoke.load_private_replay(
        args.baseline_private, args.baseline_evidence, header, cases
    )
    if replay_context["original_context_sha256"] != summary["context_sha256"]:
        raise ValueError("accepted evidence differs from original summary")
    started, rows = smoke.load_evidence(args.baseline_evidence, original)
    if started != set(range(7)) or len(rows) != 7:
        raise ValueError("accepted evidence is incomplete")
    expected_rows = {row["index"]: row for row in summary["results"]}
    if any(row.model_dump(mode="json") != expected_rows.get(row.index) for row in rows):
        raise ValueError("accepted results differ from original summary")
    eligible = {row.index for row in rows if row.index < 6 and row.accepted_valid}
    if eligible != BASELINE_INDICES:
        raise ValueError("accepted baseline page availability differs")
    pages = {}
    for entry in entries:
        index = entry["index"]
        if index not in eligible:
            continue
        source, seed = cases[index]
        plan = tensor_slot_wire_plan(entry["raw"], source_text=source).to_live_scene_plan(
            context_text=source
        )
        validate_live_scene_plan_privacy(plan, source_text=source)
        page = plan.to_page(source_text=source, visual_style=manifest["visual_style"], seed=seed)
        if digest(page.scene_spec.master_prompt) != expected_rows[index]["accepted_sha256"]:
            raise ValueError("accepted renderer differs from original prompt")
        pages[index] = page
    return original, pages


def assemble(args):
    manifest, cases = smoke.load_manifest(probe.STORY)
    header, graphs = story_graphs(args, manifest, cases)
    original, baselines = accepted_pages(args, manifest, cases)
    sources = [
        DisplaySourcePage(
            page_id=f"page-{index + 1:02}",
            source_text=source,
            seed=seed,
            facts=graphs[index],
            phase_selection=phase_selection(index, graphs[index]),
        )
        for index, (source, seed) in enumerate(cases[:6])
    ]
    pack, display = build_display_story_pack(
        sources,
        story_id=manifest["story_id"],
        title=manifest["title"],
        visual_style=manifest["visual_style"],
    )
    if len(pack.pages) != 8:
        raise ValueError("frozen demonstration requires eight display pages")
    requests = []
    pages = []
    for page, step in zip(pack.pages, display["steps"], strict=True):
        index = next(
            i for i, source in enumerate(sources) if source.page_id == step["source_page_id"]
        )
        requests.append(
            {
                "id": f"candidate-{page.page_id}",
                "source_page_index": index + 1,
                "variant": "candidate",
                "display_step_id": page.page_id,
                "seed": cases[index][1],
                "prompt": page.scene_spec.master_prompt,
                "negative_prompt": page.scene_spec.negative_prompt,
                "prompt_sha256": digest(page.scene_spec.master_prompt),
                "graph_sha256": step["graph_sha256"],
            }
        )
    for index, page in sorted(baselines.items()):
        requests.append(
            {
                "id": f"accepted-page-{index + 1:02}",
                "source_page_index": index + 1,
                "variant": "accepted",
                "display_step_id": None,
                "seed": cases[index][1],
                "prompt": page.scene_spec.master_prompt,
                "negative_prompt": page.scene_spec.negative_prompt,
                "prompt_sha256": digest(page.scene_spec.master_prompt),
                "graph_sha256": None,
            }
        )
    for index, source in enumerate(sources):
        pages.append(
            {
                "source_page_index": index + 1,
                "display_step_ids": [
                    step["step_id"]
                    for step in display["steps"]
                    if step["source_page_id"] == source.page_id
                ],
                "baseline_available": index in baselines,
                "baseline_request_id": f"accepted-page-{index + 1:02}"
                if index in baselines
                else None,
            }
        )
    if len(requests) != 11 or any(len(row["prompt"]) > 4000 for row in requests):
        raise ValueError("frozen visual batch exceeds its request contract")
    return pack, {
        "schema_version": 1,
        "kind": "story-fidelity-visual-batch",
        "story_sha256": smoke.MANIFEST_SHA256,
        "prompt_sha256": header["request_sha256"]["candidate"],
        "implementation_sha256": digest(
            {"assembler": file_hash(Path(__file__)), "probe": header["implementation_sha256"]}
        ),
        "story_context_sha256": digest(header),
        "story_evidence_sha256": file_hash(args.story_evidence),
        "story_private_sha256": file_hash(args.story_private),
        "baseline_summary_sha256": BASELINE_SUMMARY_SHA256,
        "baseline_context_sha256": digest(original),
        "requests": requests,
        "pages": pages,
        "assets_generated": False,
        "visual_acceptance_measured": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "story-evidence",
        "story-private",
        "provenance",
        "baseline-evidence",
        "baseline-private",
        "baseline-summary",
        "baseline-provenance",
        "private-pack",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11435")
    parser.add_argument("--baseline-model")
    parser.add_argument("--baseline-endpoint")
    args = parser.parse_args(argv)
    try:
        args.endpoint = probe.benchmark.validate_endpoint(args.endpoint)
        if args.baseline_endpoint:
            args.baseline_endpoint = probe.benchmark.validate_endpoint(args.baseline_endpoint)
        smoke.private_path(args.private_pack)
        if args.private_pack.exists() or args.output.exists() or args.output.is_symlink():
            raise ValueError("assembly outputs require fresh paths")
        if args.output.resolve() == args.private_pack.resolve():
            raise ValueError("private and export paths overlap")
        pack, batch = assemble(args)
        descriptor = os.open(
            args.private_pack, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        with os.fdopen(descriptor, "w") as stream:
            stream.write(pack.model_dump_json(indent=2) + "\n")
        descriptor = os.open(
            args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps(batch, indent=2, sort_keys=True) + "\n")
        return 0
    except Exception:
        print("display assembly failed; inspect local configuration and bound evidence")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
