import argparse
import asyncio
import copy
import json
import sys
import time
from pathlib import Path

import pytest
from test_simple_scene_benchmark import inputs, response

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import review_simple_scenes as review_tool  # noqa: E402


def test_review_binds_all_artifacts_and_shared_base_without_promoting_unknowns(
    tmp_path, monkeypatch
):
    benchmark = review_tool.benchmark
    manifest, path = inputs(tmp_path, monkeypatch)
    manifest.update(status="authorized", expires_at=int(time.time()) + 1200)
    path.write_bytes(benchmark.preparation.encoded(manifest))
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    auth_path = private / "authorization.json"
    authorization = {
        "schema_version": 1,
        "manifest_sha256": benchmark.legacy.file_hash(path),
        "reservation_id": "synthetic-review",
        "reserved_usd": 1.75,
        "maximum_operations": 12,
        "ledger_sha256": "b" * 64,
    }
    benchmark.legacy.write_exclusive(auth_path, benchmark.preparation.encoded(authorization))
    args = argparse.Namespace(
        manifest=path,
        authorization=auth_path,
        authorization_sha256=benchmark.legacy.file_hash(auth_path),
        output=tmp_path / "results",
        gallery=tmp_path / "gallery",
        key=private / "key.json",
        review=tmp_path / "review.json",
        deadline_unix=time.time() + 500,
    )
    args.output.mkdir()

    class OfflineClient:
        last_failure = last_timings = None

        async def render(self, operation, reference, reference_sha):
            return response(manifest, operation, reference_sha), dict.fromkeys(
                benchmark.cold.TIMINGS, 0.0
            )

        async def cleanup(self):
            return True

    asyncio.run(
        benchmark.execute(
            args, manifest, benchmark.header(args, manifest, authorization), OfflineClient()
        )
    )
    exposed = tmp_path / "exposed"
    exposed.mkdir(mode=0o755)
    unsafe = copy.copy(args)
    unsafe.key = exposed / "key.json"
    with pytest.raises(ValueError):
        review_tool.build(unsafe)
    alias = tmp_path / "private-alias"
    alias.symlink_to(private, target_is_directory=True)
    unsafe.key = alias / "key.json"
    with pytest.raises(ValueError):
        review_tool.build(unsafe)
    assert not args.gallery.exists() and not args.key.exists()
    review_tool.build(args)
    public = json.loads((args.gallery / "review-data.json").read_text())
    assert len(list(args.gallery.glob("frame-*.jpg"))) == 12
    assert all(len(option["frames"]) == 2 for page in public["pages"] for option in page["options"])
    assert all(
        page["options"][0]["frames"][0] == page["options"][1]["frames"][0]
        for page in public["pages"]
    )
    assert not any(
        word in json.dumps(public) for word in ("text_next", "reference_next", "prompt_sha256")
    )
    ratings = {
        "schema_version": 1,
        "review_id": public["review_id"],
        "pages": [
            {
                "page": page["page"],
                "options": [
                    {
                        "label": option["label"],
                        "correct": "correct",
                        "legible": "clear",
                        "continuity": "consistent",
                        "rating": "5",
                        "forbidden": [],
                        "required": ["visible"] * len(page["required"]),
                    }
                    for option in page["options"]
                ],
            }
            for page in public["pages"]
        ],
    }

    def score(value):
        args.review.write_text(json.dumps(value))
        return review_tool.score(args)

    passing = score(ratings)
    assert all(row["correct"] == 4 for row in passing["variants"].values())
    assert passing["controls"] == [
        {
            "control_id": control["id"],
            "variants": {"text_next": "correct", "reference_next": "correct"},
        }
        for control in benchmark.fixture()["controls"]
    ]
    assert passing["human_review_complete"] and passing["verdicts_complete"]
    for state, outcome in (("missing", "incorrect"), ("", "unrated"), ("unsure", "unrated")):
        changed = copy.deepcopy(ratings)
        changed["pages"][0]["options"][0]["required"][0] = state
        result = score(changed)
        assert all(row[outcome] == 1 and row["correct"] == 3 for row in result["variants"].values())
        assert not result["production_promotion_authorized"]
    changed = copy.deepcopy(ratings)
    changed["pages"][0]["options"][0]["required"][0] = "missing"
    changed["pages"][0]["options"][1]["required"][0] = ""
    resolved = score(changed)
    assert resolved["verdicts_complete"] and not resolved["human_review_complete"]
    changed = copy.deepcopy(ratings)
    changed["pages"][0]["options"][0]["required"][0] = "unsure"
    changed["pages"][0]["options"][1]["correct"] = "incorrect"
    mixed = score(changed)
    assert sorted(
        (row["incorrect"], row["unrated"]) for row in mixed["variants"].values()
    ) == [(0, 1), (1, 0)]
    hidden = json.loads(args.key.read_text())["pages"][0]["options"]
    assert mixed["controls"][0]["variants"] == {
        hidden[0]["variant"]: "unrated",
        hidden[1]["variant"]: "incorrect",
    }
    for mutation in ("wrong_review", "swapped_options", "boolean_fact"):
        changed = copy.deepcopy(ratings)
        if mutation == "wrong_review":
            changed["review_id"] = "f" * 32
        elif mutation == "swapped_options":
            changed["pages"][0]["options"].reverse()
        else:
            changed["pages"][0]["options"][0]["required"][0] = False
        with pytest.raises(ValueError):
            score(changed)
    score(ratings)
    frame = args.gallery / public["pages"][0]["options"][0]["frames"][1]
    original = frame.read_bytes()
    frame.write_bytes((args.gallery / public["pages"][0]["options"][1]["frames"][1]).read_bytes())
    with pytest.raises(ValueError):
        review_tool.score(args)
    frame.write_bytes(original)
    # Even a non-gallery depth asset is part of the bound twelve-image result.
    benchmark.artifact(args.output, 11, "depth").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        review_tool.score(args)
