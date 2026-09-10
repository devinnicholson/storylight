"""A resealed receipt cannot leave the pilot selection file unauthenticated."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v5_selector_test', HERE / 'select-recipe.py')
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)


def test_omitted_selection_receipt_refused_after_resealing_completion(tmp_path):
    trainer = HERE / 'train.py'
    trainer_sha = selector.digest(trainer)
    args = SimpleNamespace(trainer=trainer, trainer_sha256=trainer_sha,
                           v2_trainer=HERE / "support/training-support.py",
                           output=tmp_path / 'chosen.json')
    for recipe, attr in [('qv', 'qv'), ('text-linear', 'text_linear')]:
        directory = tmp_path / recipe
        directory.mkdir()
        protocol = dict(runner_sha256=trainer_sha, recipe=recipe, max_steps=1200,
                        model_manifest_sha256='same', data_manifest_sha256='same',
                        development_manifest_sha256='same', fewshot_prompt_sha256='same',
                        seed=7, learning_rate=.0001, rank=16, alpha=32)
        selection = dict(selected_step=400, selected_adapter_reloaded_and_verified=True,
                         screen_generation_started=False,
                         development=[dict(step=n, rows=256, supervised_tokens=1000,
                                           token_weighted_loss=1.) for n in (400, 800, 1200)])
        for name, data in [('protocol.json', protocol), ('selection.json', selection),
                           ('loaded.json', dict(initial_adapter_sha256='same'))]:
            (directory / name).write_text(json.dumps(data))
        complete = dict(completed_steps=1200, recipe=recipe, selected_step=400,
                        files={p.name: selector.digest(p) for p in directory.iterdir()})
        (directory / 'completed.json').write_text(json.dumps(complete))
        setattr(args, attr, directory)
        setattr(args, attr + '_sha256', selector.digest(directory / 'completed.json'))
    selector.select(args)
    assert json.loads(args.output.read_text())['selected']['recipe'] == 'qv'
    completion = args.qv / 'completed.json'
    value = json.loads(completion.read_text())
    del value['files']['selection.json']
    completion.write_text(json.dumps(value))
    args.qv_sha256 = selector.digest(completion)
    args.output = tmp_path / 'must-not-exist.json'
    with pytest.raises(ValueError, match='pilot_proof_missing'):
        selector.select(args)
    assert not args.output.exists()
