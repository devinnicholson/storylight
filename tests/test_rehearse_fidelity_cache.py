from __future__ import annotations

import json
from contextlib import contextmanager

import httpx
import pytest
from test_install_fidelity_display import captured_story, completed_render  # noqa: F401

from scripts import install_fidelity_display as installer
from scripts import rehearse_fidelity_cache as rehearsal

HTTP_CLIENT = httpx.Client


@pytest.fixture
def installed(completed_render, tmp_path, monkeypatch):  # noqa: F811
    args, _, cases = completed_render
    assert installer.main(args) == 0
    receipt_path = tmp_path / "installation.json"
    receipt = json.loads(receipt_path.read_bytes())
    data = tmp_path / "installed"
    output = tmp_path / "rehearsal.json"
    argv = [
        "--data-dir",
        str(data),
        "--installation-receipt",
        str(receipt_path),
        "--proof-installation-sha256",
        rehearsal.digest(receipt_path.read_bytes()),
        "--output",
        str(output),
    ]
    monkeypatch.setattr(rehearsal, "on_jetson", lambda: True)
    return argv, data, receipt, output, cases


def test_three_cached_passes_and_restart_keep_source_private(installed, monkeypatch):
    argv, data, receipt, output, cases = installed
    pack, path = rehearsal.installed_pack(data, receipt)
    original = path.read_bytes()
    requests, starts, stops = [], [], []

    def respond(request):
        requests.append(request.url.path)
        if request.url.path == "/v1/story-packs/latest":
            return httpx.Response(200, content=pack.model_dump_json().encode())
        asset = next(asset for asset in pack.assets if asset.local_uri == request.url.path)
        content = (data / "cache/assets" / asset.local_uri.removeprefix("/v1/assets/")).read_bytes()
        return httpx.Response(200, content=content, headers={"ETag": f'"{asset.checksum_sha256}"'})

    @contextmanager
    def server(directory, port):
        assert directory == data and port == 18089
        starts.append(port)
        try:
            with HTTP_CLIENT(
                base_url="http://127.0.0.1:18089", transport=httpx.MockTransport(respond)
            ) as client:
                yield client
        finally:
            stops.append(port)

    monkeypatch.setattr(rehearsal, "isolated_api", server)
    assert rehearsal.main(argv) == 0
    result = json.loads(output.read_bytes())
    assert len(starts) == len(stops) == result["api_starts"] == 2
    assert result["cached_passes_before_restart"] == result["cached_passes_after_restart"] == 3
    assert len(result["passes"]) == 6 and len(requests) == 102
    assert all(requests.count(asset.local_uri) == 6 for asset in pack.assets)
    assert result["stored_pack_sha256"] == receipt["stored_pack_sha256"]
    assert not result["visual_acceptance_measured"] and not result["physical_playback_verified"]
    assert path.read_bytes() == original
    assert all(source not in output.read_text() for source, _ in cases)
    assert all(asset.prompt not in output.read_text() for asset in pack.assets)
    assert rehearsal.main(argv) == 1 and len(starts) == 2


def test_corrupt_cache_wrong_api_pack_and_reserved_port_refuse(installed, monkeypatch, capsys):
    argv, data, receipt, output, _ = installed
    pack, _ = rehearsal.installed_pack(data, receipt)
    asset = pack.assets[0]
    path = data / "cache/assets" / asset.local_uri.removeprefix("/v1/assets/")
    original = path.read_bytes()
    monkeypatch.setattr(
        rehearsal, "isolated_api", lambda *args: pytest.fail("started API before preflight")
    )
    assert rehearsal.main([*argv, "--port", "8080"]) == 1
    path.write_bytes(b"private corrupt cache contents")
    assert rehearsal.main(argv) == 1 and not output.exists()
    path.write_bytes(original)
    starts, stops = [], []

    @contextmanager
    def wrong_server(*args):
        starts.append(True)
        altered = pack.model_copy(update={"title": "private wrong pack title"})
        try:
            with HTTP_CLIENT(
                base_url="http://127.0.0.1:18089",
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, content=altered.model_dump_json().encode())
                ),
            ) as client:
                yield client
        finally:
            stops.append(True)

    monkeypatch.setattr(rehearsal, "isolated_api", wrong_server)
    assert rehearsal.main(argv) == 1 and not output.exists()
    assert len(starts) == len(stops) == 1
    assert "private" not in capsys.readouterr().out
