import importlib.util
from pathlib import Path


def test_word_error_scoring_preserves_scene_meaning():
    source = Path(__file__).parents[1] / "scripts/benchmark_local_asr.py"
    spec = importlib.util.spec_from_file_location("local_asr_benchmark", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    words, distance = module.words, module.distance
    assert distance(words("The pink fox."), words("THE PINK FOX!")) == 0
    assert distance(words("two foxes"), words("three foxes")) == 1
    assert distance(words("is not chasing"), words("is chasing")) == 1
    assert distance(words("dog"), words("the lazy dog")) == 2
    assert len(module.PHRASES) * len(module.VOICES) == 12
