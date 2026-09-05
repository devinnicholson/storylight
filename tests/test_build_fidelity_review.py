from __future__ import annotations

import json
import re
import stat

from test_install_fidelity_display import captured_story, completed_render  # noqa: F401

from scripts import build_fidelity_review as gallery


def arguments(inputs, tmp_path):
    return [
        "--batch",
        str(inputs.output),
        "--proof-batch-sha256",
        gallery.installer.assembly.file_hash(inputs.output),
        "--render-dir",
        str(tmp_path / "renders"),
        "--output",
        str(tmp_path / "gallery"),
        "--key-file",
        str(inputs.private_pack.parent / "review-key.json"),
    ]


def test_gallery_hides_variant_mapping_and_keeps_ordered_states(completed_render, tmp_path):  # noqa: F811
    _, inputs, sources = completed_render
    args = arguments(inputs, tmp_path)
    assert gallery.main(args) == 0
    output = tmp_path / "gallery"
    key_file = inputs.private_pack.parent / "review-key.json"
    key = json.loads(key_file.read_text())
    public = json.loads((output / "review-data.json").read_text())
    exposed = (output / "index.html").read_text() + (output / "review-data.json").read_text()
    assert not any(
        word in exposed for word in ("candidate", "accepted", "prompt", "request_sha256")
    )
    assert all(source not in exposed for source, _ in sources)
    assert not key_file.is_relative_to(output) and stat.S_IMODE(key_file.stat().st_mode) == 0o600
    assert key["review_data_sha256"] == gallery.installer.assembly.digest(public)
    assert [len(page["options"]) for page in public["pages"]] == [1, 1, 2, 1, 2, 2]
    assert [page["states"] for page in public["pages"]] == [1, 1, 1, 1, 2, 2]
    batch = gallery.render.read_batch(inputs.output, args[3])
    paths = {
        row.id: tmp_path / f"renders/image-{i:02}/master.jpg"
        for i, row in enumerate(gallery.render.ordered_requests(batch))
    }
    for page, mapping in zip(public["pages"], key["pages"], strict=True):
        for option, private in zip(page["options"], mapping["options"], strict=True):
            assert option["label"] == private["label"]
            expected = [paths[request].read_bytes() for request in private["requests"]]
            if len(expected) == 1:
                expected *= page["states"]
            assert [output.joinpath(frame).read_bytes() for frame in option["frames"]] == expected
            assert all(
                re.fullmatch(r"frame-[0-9a-f]{64}\.jpg", frame) for frame in option["frames"]
            )
    snapshot = {path.name: path.read_bytes() for path in output.iterdir()}
    assert gallery.main(args) == 1
    assert {path.name: path.read_bytes() for path in output.iterdir()} == snapshot


def test_gallery_rejects_swapped_assets_before_writing(completed_render, tmp_path):  # noqa: F811
    _, inputs, _ = completed_render
    args = arguments(inputs, tmp_path)
    first = tmp_path / "renders/image-00/master.jpg"
    other = tmp_path / "renders/image-01/master.jpg"
    first.write_bytes(other.read_bytes())
    assert gallery.main(args) == 1
    assert not (tmp_path / "gallery").exists()
    assert not (inputs.private_pack.parent / "review-key.json").exists()
