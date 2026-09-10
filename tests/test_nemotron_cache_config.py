import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "infra/gcp/gke/configure-nemotron-cache.sh"


def test_configuration_is_guarded_before_any_cluster_access():
    result = subprocess.run(
        ["bash", str(SCRIPT)], capture_output=True, text=True, env={"PATH": os.environ["PATH"]}
    )
    assert result.returncode == 2
    assert "Dry guard" in result.stdout


@pytest.mark.parametrize(
    "running,changed_image,expected", [(False, False, 0), (True, False, 1)]
)
def test_cache_configuration_requires_stopped_runtime(tmp_path, running, changed_image, expected):
    deployment = {
        "spec": {
            "replicas": int(running),
            "template": {
                "spec": {
                    "nodeSelector": {"cloud.google.com/gke-accelerator": "nvidia-l4"},
                    "volumes": [{"persistentVolumeClaim": {"claimName": "storylight-nim-cache"}}],
                    "containers": [
                        {
                            "name": "nemotron-nim",
                            "image": "changed"
                            if changed_image
                            else (
                                "nvcr.io/nim/nvidia/llama-3.1-nemotron-nano-vl-8b-v1@sha256:"
                                "f4f0ef214fc448af0b9e7a9de84f8d163b21b698d2ad9e4a5d6d8afa8f03b065"
                            ),
                            "env": [
                                {"name": key, "value": value}
                                for key, value in {
                                    "NIM_MAX_MODEL_LEN": "2048",
                                    "NIM_MAX_BATCH_SIZE": "1",
                                    "NIM_LOW_MEMORY_MODE": "1",
                                    "NIM_CACHE_PATH": "/opt/nim/.cache",
                                }.items()
                            ],
                            "resources": {"limits": {"nvidia.com/gpu": "1"}},
                        },
                        {"name": "gpu-watchdog"},
                    ],
                }
            },
        },
    }
    fixture = tmp_path / "deployment.json"
    fixture.write_text(json.dumps(deployment))
    stub = tmp_path / "kubectl"
    stub.write_text(
        "#!/usr/bin/env python3\nimport os, pathlib, sys\n"
        "if 'get' in sys.argv:\n    print(pathlib.Path(os.environ['FIXTURE']).read_text())\n"
        "else:\n    pathlib.Path(os.environ['MUTATION']).write_text(' '.join(sys.argv))\n"
    )
    stub.chmod(0o755)
    mutation = tmp_path / "mutation"
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "FIXTURE": str(fixture),
            "MUTATION": str(mutation),
            "STORYLIGHT_NIM_CACHE_APPLY": "I_UNDERSTAND_THIS_CONFIGURES_THE_STOPPED_NIM",
        },
    )
    assert result.returncode == expected, result.stderr
    assert mutation.exists() is (expected == 0)
    if mutation.exists():
        command = mutation.read_text()
        assert "set env" in command
        assert "NIM_ENABLE_KV_CACHE_REUSE-" in command
        assert "NIM_SERVED_MODEL_NAME=nvidia/" in command
        assert "NIM_MODEL_PROFILE=308eb448" in command
        assert "scale" not in command
