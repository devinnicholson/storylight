from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_klein_cold_start as benchmark  # noqa: E402


def test_isolated_deployment_bounds_resources_and_claims_before_model_import(tmp_path):
    root = tmp_path / "container"
    root.mkdir()
    for name in ("modal_klein_cold_start.py", "klein_scene_runtime.py"):
        shutil.copyfile(benchmark.ROOT / "deploy" / name, root / name)
    identity, cases = benchmark.frozen_cases()
    script = r"""
import builtins, hashlib, json, os, runpy, sys, time
from pathlib import Path
from types import SimpleNamespace

root = Path(sys.argv[1])
identity, cases = json.loads(sys.stdin.read())
registrations, mounts, claimed = {}, {}, []
permit = False
class Image:
    @staticmethod
    def from_id(value):
        assert value == 'im-WtXer8GjRPdgMqWAAUSMwJ'
        return Image()
    def add_local_file(self, source, destination, **kwargs):
        assert not kwargs.get('copy', False)
        mounts[str(destination)] = str(source)
        return self
class App:
    def __init__(self, name):
        assert name == 'bookforge-klein-cold-start'
    def function(self, **options):
        def register(function):
            registrations[function.__name__] = options
            return function
        return register
def claim(key, value, *, skip_if_exists):
    assert value is True and skip_if_exists is True
    claimed.append(key)
    return permit
sys.modules['modal'] = SimpleNamespace(
    Image=Image, App=App, is_local=lambda: True,
    Dict=SimpleNamespace(from_name=lambda *a, **k: SimpleNamespace(put=claim)),
    Volume=SimpleNamespace(from_name=lambda *a, **k: object()),
)
module = runpy.run_path(str(root / 'modal_klein_cold_start.py'))
assert set(mounts) == {'/root/klein_scene_runtime.py', '/root/cold-start-manifest.json'}
assert Path(mounts['/root/klein_scene_runtime.py']).read_bytes() == (
    root / 'klein_scene_runtime.py'
).read_bytes()
baseline, candidate = registrations['baseline_cycle'], registrations['candidate_cycle']
assert baseline['memory'] == (65536, 65536)
assert candidate['memory'] == (16384, 65536)
assert {k: v for k, v in baseline.items() if k != 'memory'} == {
    k: v for k, v in candidate.items() if k != 'memory'
}
assert baseline['cpu'] == (8, 8) and baseline['gpu'] == 'L4'
assert baseline['single_use_containers'] and baseline['max_containers'] == 1
assert baseline['min_containers'] == baseline['retries'] == 0
assert tuple(baseline[k] for k in ('timeout', 'startup_timeout', 'scaledown_window')) == (
    180, 120, 2
)
assert tuple(baseline[k] for k in ('cloud', 'region', 'routing_region')) == (
    'aws', 'us-east', 'us-east'
)

manifest = {
    'status': 'authorized', 'experiment_id': module['EXPERIMENT'],
    'image_id': module['IMAGE_ID'], 'cache_id': module['CACHE_ID'],
    'expires_at': int(time.time()) + 1200,
    'runtime_sha256': hashlib.sha256((root / 'klein_scene_runtime.py').read_bytes()).hexdigest(),
    'deployment_sha256': hashlib.sha256(
        (root / 'modal_klein_cold_start.py').read_bytes()
    ).hexdigest(),
    'expected_identity': identity, 'cases': cases,
    'operations': [{'request_id': f'{index:032x}', 'variant': 'baseline'} for index in range(6)],
}
path = root / 'manifest.json'
module['run_cycle'].__globals__['MANIFEST'] = path
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'diffusers', 'transformers', 'triton'}:
        raise AssertionError('model import boundary')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
os.environ['MODAL_CLOUD_PROVIDER'] = 'CLOUD_PROVIDER_AWS'
os.environ['MODAL_REGION'] = 'us-east-1'
def refused(value, request='0' * 32, reason=''):
    path.write_text(json.dumps(value))
    try:
        module['baseline_cycle'](request)
    except ValueError as error:
        assert reason in str(error)
    else:
        raise AssertionError('unexpected authorization')
refused(manifest, 'f' * 32, 'not authorized')
refused({**manifest, 'cases': cases + cases[:1]}, reason='cases differ')
refused({**manifest, 'expires_at': 1}, reason='authorization differs')
os.environ['MODAL_REGION'] = 'us-west-2'
refused(manifest, reason='placement differs')
assert not claimed
os.environ['MODAL_REGION'] = 'us-east-1'
refused(manifest, reason='already claimed')
assert claimed == ['0' * 32]
permit = True
try:
    module['baseline_cycle']('0' * 32)
except AssertionError as error:
    assert str(error) == 'model import boundary'
else:
    raise AssertionError('valid request did not reach model boundary')
assert claimed == ['0' * 32, '0' * 32]
assert not {'torch', 'diffusers', 'transformers', 'triton'} & set(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(root)],
        input=json.dumps([identity, cases]),
        cwd=root,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
