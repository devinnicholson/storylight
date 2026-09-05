from __future__ import annotations

import copy
import json

import pytest
from test_install_fidelity_display import captured_story, completed_render  # noqa: F401

from scripts import build_fidelity_review as gallery
from scripts import summarize_fidelity_review as summary


@pytest.fixture
def reviewed(completed_render, tmp_path):  # noqa: F811
    _, inputs, sources = completed_render
    key_path = inputs.private_pack.parent / "review-key.json"
    public_path = tmp_path / "gallery/review-data.json"
    assert (
        gallery.main(
            [
                "--batch",
                str(inputs.output),
                "--proof-batch-sha256",
                summary.assembly.file_hash(inputs.output),
                "--render-dir",
                str(tmp_path / "renders"),
                "--output",
                str(public_path.parent),
                "--key-file",
                str(key_path),
            ]
        )
        == 0
    )
    key = json.loads(key_path.read_text())
    public = json.loads(public_path.read_text())
    review = {"schema_version": 1, "review_id": key["review_id"], "pages": []}
    for page, mapping in zip(public["pages"], key["pages"], strict=True):
        options = []
        for option in mapping["options"]:
            candidate = option["variant"] == "candidate"
            rating = ([5, 5, 3, 2, 1, 2] if candidate else [0, 0, 2, 0, 3, 1])[page["page"] - 1]
            options.append(
                {
                    "label": option["label"],
                    "correct": "correct" if page["page"] < 3 else "incorrect",
                    "legible": "clear",
                    "continuity": "consistent" if page["page"] < 3 else "inconsistent",
                    "rating": str(rating),
                    "required": [""] * len(page["required"]),
                    "forbidden": [""] * len(page["forbidden"]),
                }
            )
        review["pages"].append({"page": page["page"], "options": options})
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review))
    output = tmp_path / "summary.json"
    argv = [
        "--review",
        str(review_path),
        "--gallery-data",
        str(public_path),
        "--key-file",
        str(key_path),
        "--batch",
        str(inputs.output),
        "--render-dir",
        str(tmp_path / "renders"),
        "--output",
        str(output),
    ]
    return argv, review_path, public_path, key_path, output, sources


def test_bound_ratings_keep_blank_facts_and_separate_correctness(reviewed):
    argv, review_path, _, key_path, output, sources = reviewed
    assert summary.main(argv) == 0
    result = json.loads(output.read_text())
    assert result["decision"] == "reject" and result["candidate_rejected_pages"] == [3, 4, 5, 6]
    assert (
        result["counts"]["candidate"]["correct"],
        result["counts"]["candidate"]["available"],
    ) == (2, 6)
    assert (result["counts"]["accepted"]["correct"], result["counts"]["accepted"]["available"]) == (
        0,
        3,
    )
    assert result["paired_overall_ratings"] == {
        "candidate_wins": 2,
        "accepted_wins": 1,
        "ties": 0,
        "unrated": 0,
    }
    assert result["paired_correctness"]["both_incorrect"] == 3
    assert (
        result["counts"]["candidate"]["clear"] == 6 and result["counts"]["accepted"]["clear"] == 3
    )
    assert result["fact_annotations"]["rated"] == 0 and result["fact_annotations"]["unrated"] == 58
    assert not result["fact_annotations_complete"]
    assert all(
        value is None
        for page in result["pages"]
        for option in page["options"].values()
        for value in [*option["required"], *option["forbidden"]]
    )
    assert all(source not in output.read_text() for source, _ in sources)
    assert all(word not in output.read_text() for word in ('"label"', '"requests"', '"frames"'))
    assert summary.main(argv) == 1
    output.unlink()
    review = json.loads(review_path.read_text())
    for page in review["pages"]:
        for option in page["options"]:
            option.update(correct="correct", continuity="consistent")
    review_path.write_text(json.dumps(review))
    assert summary.main(argv) == 0
    assert json.loads(output.read_text())["decision"] == "incomplete"
    output.unlink()
    for page, mapping in zip(
        review["pages"], json.loads(key_path.read_text())["pages"], strict=True
    ):
        candidate_label = next(
            option["label"] for option in mapping["options"] if option["variant"] == "candidate"
        )
        candidate = next(option for option in page["options"] if option["label"] == candidate_label)
        candidate["required"] = ["visible"] * len(candidate["required"])
        candidate["forbidden"] = ["absent"] * len(candidate["forbidden"])
    review_path.write_text(json.dumps(review))
    assert summary.main(argv) == 0
    result = json.loads(output.read_text())
    assert result["decision"] == "incomplete" and not result["fact_annotations_complete"]


def test_unknown_missing_swapped_or_corrupt_inputs_refuse(reviewed, tmp_path, capsys):
    argv, review_path, public_path, key_path, output, _ = reviewed
    original = json.loads(review_path.read_text())
    mutations = [
        lambda value: value.update(review_id="f" * 32),
        lambda value: value["pages"].pop(),
        lambda value: value["pages"].__setitem__(1, copy.deepcopy(value["pages"][0])),
        lambda value: value["pages"][2]["options"].pop(),
        lambda value: value["pages"][2]["options"].__setitem__(
            1, copy.deepcopy(value["pages"][2]["options"][0])
        ),
        lambda value: value["pages"][0]["options"][0].update(label="C"),
        lambda value: value["pages"][0]["options"][0].update(correct="private invalid rating"),
        lambda value: value["pages"][0]["options"][0]["required"].pop(),
    ]
    for mutate in mutations:
        value = copy.deepcopy(original)
        mutate(value)
        review_path.write_text(json.dumps(value))
        assert summary.main(argv) == 1 and not output.exists()
    review_path.write_text(json.dumps(original))
    public_original, key_original = public_path.read_text(), key_path.read_text()
    public, key = json.loads(public_original), json.loads(key_original)
    public["pages"][0]["required"][0] = "private altered checklist"
    key["review_data_sha256"] = summary.assembly.digest(public)
    public_path.write_text(json.dumps(public))
    key_path.write_text(json.dumps(key))
    assert summary.main(argv) == 1 and not output.exists()
    public_path.write_text(public_original)
    key = json.loads(key_original)
    key["pages"][0]["options"][0]["requests"] = key["pages"][1]["options"][0]["requests"]
    key_path.write_text(json.dumps(key))
    assert summary.main(argv) == 1 and not output.exists()
    key_path.write_text(key_original)
    first, second = (
        tmp_path / "renders/image-00/master.jpg",
        tmp_path / "renders/image-01/master.jpg",
    )
    first.write_bytes(second.read_bytes())
    assert summary.main(argv) == 1 and not output.exists()
    assert "private" not in capsys.readouterr().out
