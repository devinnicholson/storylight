from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import stat
import sys
from pathlib import Path

import pytest
from PIL import Image

from bookforge.asset_cache import AssetCache
from bookforge.klein_scene_provider import write_bundle
from bookforge.story_store import StoryPackStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_assemble_fidelity_display import captured_story, cli  # noqa: E402,F401

from scripts import install_fidelity_display as installer  # noqa: E402


@pytest.fixture
def completed_render(captured_story, tmp_path):  # noqa: F811
    args, cases = captured_story
    assembly, render = installer.assembly, installer.render
    assert assembly.main(cli(args)) == 0
    proof = assembly.file_hash(args.output)
    batch = render.read_batch(args.output, proof)
    directory = tmp_path / "renders"
    directory.mkdir(mode=0o700)
    identity = render.expected_identity()
    header = {
        "kind": "header",
        "schema_version": 1,
        "mode": "execute",
        "batch_sha256": proof,
        "harness_sha256": assembly.file_hash(Path(render.__file__)),
        "provider_sha256": assembly.file_hash(
            render.ROOT / "src/bookforge/klein_scene_provider.py"
        ),
        "runtime_sha256": render.EXPECTED_RUNTIME_SHA256,
        "deployment_source_sha256": render.EXPECTED_DEPLOYMENT_SHA256,
        "renderer_receipt_sha256": render.RECEIPT_SHA256,
        "expected_identity_sha256": assembly.digest(identity),
        "cache_id_sha256": assembly.digest(render.CACHE_ID),
        "maximum_requests": 11,
        "call_ceiling_usd": 0.25,
        "batch_ceiling_usd": 2.75,
        "automatic_retries": 0,
        "prewarm_requests": 0,
        "negative_prompt_supported": False,
        "visual_acceptance_measured": False,
        **{
            key: "e" * 64
            for key in (
                "plan_sha256",
                "ledger_before_sha256",
                "ledger_path_sha256",
                "attempt_claim_sha256",
            )
        },
        "tokens": {
            "tokenizer_sha256": "f" * 64,
            "token_counts": {row.id: 64 for row in batch.requests},
        },
    }
    journal = directory / "journal.jsonl"
    render.append(journal, header)
    for ordinal, row in enumerate(render.ordered_requests(batch)):
        image = io.BytesIO()
        Image.new("RGB", (1024, 576), (ordinal * 20, 80, 140)).save(image, format="JPEG")
        content = image.getvalue()
        payload = {
            "identity": identity,
            "master": content,
            "depth": content,
            "metrics": {
                "seed": row.seed,
                "sequence_bucket": 128,
                "token_count": 64,
                "image_seconds": 0.1,
                "depth_seconds": 0.1,
                "encoding_seconds": 0.1,
                "total_seconds": 0.3,
                "master_sha256": hashlib.sha256(content).hexdigest(),
                "depth_sha256": hashlib.sha256(content).hexdigest(),
            },
            "model_load_seconds": 0.1,
            "startup_seconds": 0.1,
            "cache_setup_seconds": 0.1,
            "warm_state": "warm",
        }
        bundle = write_bundle(
            render.fast_request(row),
            directory / f"image-{ordinal:02}",
            payload,
            0.5,
            "private-reservation",
        )
        render.append(
            journal,
            {
                "kind": "start",
                "ordinal": ordinal,
                "request_sha256": assembly.digest(row.model_dump()),
            },
        )
        render.append(
            journal,
            {
                "kind": "result",
                "ordinal": ordinal,
                "status": "ok",
                "manifest_sha256": assembly.file_hash(bundle.manifest_path),
                "identity_sha256": assembly.digest(identity),
                "reservation_sha256": assembly.digest("private-reservation"),
                "master_sha256": bundle.master.sha256,
                "depth_sha256": bundle.depth.sha256,
                "metrics": {
                    key: bundle.manifest["stages"]["fast"][key] for key in installer.METRICS
                },
            },
        )
    render.append(journal, {"kind": "complete", "mode": "execute", "ledger_after_sha256": "a" * 64})
    install_args = [
        "--batch",
        str(args.output),
        "--proof-batch-sha256",
        proof,
        "--private-pack",
        str(args.private_pack),
        "--proof-private-pack-sha256",
        assembly.file_hash(args.private_pack),
        "--render-dir",
        str(directory),
        "--data-dir",
        str(tmp_path / "installed"),
        "--output",
        str(tmp_path / "installation.json"),
    ]
    return install_args, args, cases


def test_install_verified_candidate_pages_and_cache_replay(completed_render, tmp_path):
    args, inputs, cases = completed_render
    original_pack = inputs.private_pack.read_bytes()
    previous_umask = os.umask(0o002)
    try:
        assert installer.main(args) == 0
    finally:
        os.umask(previous_umask)
    data = tmp_path / "installed"
    pack = asyncio.run(StoryPackStore(data / "story-packs").latest())
    assert len(pack.pages) == 8 and len(pack.assets) == 16
    assert {asset.page_id for asset in pack.assets} == {page.page_id for page in pack.pages}
    assert all(
        asset.state.value == "ready" and asset.local_uri.startswith("/v1/assets/")
        for asset in pack.assets
    )
    assert all(
        sum(asset.page_id == page.page_id for asset in pack.assets) == 2 for page in pack.pages
    )
    assert asyncio.run(AssetCache(data / "cache/assets").install_pack(pack, data)) == pack
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o700
        for path in (data, *data.rglob("*"))
        if path.is_dir()
    )
    assert inputs.private_pack.read_bytes() == original_pack
    receipt = (tmp_path / "installation.json").read_text()
    assert json.loads(receipt)["cached_replay_checksums_verified"] is True
    assert all(source not in receipt for source, _ in cases)
    assert "private-reservation" not in receipt
    assert installer.main(args) == 1


def test_install_refuses_swapped_corrupt_and_incomplete_inputs_before_target(
    completed_render, tmp_path, capsys
):
    args, inputs, _ = completed_render
    directory = tmp_path / "renders"
    journal = directory / "journal.jsonl"
    original = journal.read_text()
    entries = [json.loads(line) for line in original.splitlines()]
    manifest = directory / "image-00/scene.manifest.json"
    original_manifest = manifest.read_text()
    value = json.loads(original_manifest)
    value["request"]["prompt"] = "private wrong-page prompt"
    manifest.write_text(json.dumps(value))
    entries[2]["manifest_sha256"] = installer.assembly.file_hash(manifest)
    journal.write_text("\n".join(json.dumps(row) for row in entries) + "\n")
    assert installer.main(args) == 1
    assert not (tmp_path / "installed").exists()
    manifest.write_text(original_manifest)
    journal.write_text(original)
    asset = directory / "image-01/master.jpg"
    content = asset.read_bytes()
    asset.write_bytes(content + b"corrupt")
    assert installer.main(args) == 1
    asset.write_bytes(content)
    journal.write_text("\n".join(original.splitlines()[:-1]) + "\n")
    assert installer.main(args) == 1
    journal.write_text(original)
    original_pack = inputs.private_pack.read_text()
    changed_pack = json.loads(original_pack)
    changed_pack["title"] = "private changed title"
    inputs.private_pack.write_text(json.dumps(changed_pack))
    assert installer.main(args) == 1
    inputs.private_pack.write_text(original_pack)
    checkout = tmp_path / "another-checkout"
    checkout.mkdir()
    (checkout / ".git").mkdir()
    for parent in (installer.render.ROOT, checkout):
        target = parent / "private-install-rejection-control"
        changed_args = [*args]
        changed_args[changed_args.index("--data-dir") + 1] = str(target)
        assert installer.main(changed_args) == 1
        assert not target.exists()
    inputs.private_pack.chmod(0o644)
    assert installer.main(args) == 1
    assert not (tmp_path / "installed").exists() and not (tmp_path / "installation.json").exists()
    assert "private wrong-page prompt" not in capsys.readouterr().out
