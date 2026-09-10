"""Only exposed V4 cases exercise the new count/receipt contract before test release."""

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v5_evaluation_under_test', HERE / 'evaluate.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def fixture():
    previous = HERE / "support/v4-screen-gold.jsonl"
    gold = [{**json.loads(line), 'id': f'exposed-{copy}-{json.loads(line)["id"]}'}
            for copy in range(2) for line in previous.read_text().splitlines()]
    records = [dict(id=r['id'], arm=arm, repetition=rep, output=r['target'], latency_ms=2.)
               for r in gold for arm in evaluation.ARMS for rep in range(2)]
    return gold, records


def test_fixed128_cases_two_repetitions_and_unknown_error_not_refusal():
    gold, records = fixture()
    result = evaluation.score_records(gold, records)
    assert result['arms']['new']['positive_case_exact'] == 96
    assert result['arms']['new']['correct_refusal_cases'] == 32
    assert result['adapter_comparison_gate_passed'] is False  # Equal arms are not improvement.
    refusal = next(r for r in gold if r['expected'] == 'REFUSE')['id']
    next(r for r in records if r['id'] == refusal and r['arm'] == 'new')['output'] = (
        'ERROR:rejection_cause_unknown')
    result = evaluation.score_records(gold, records)
    assert result['arms']['new']['correct_refusal_cases'] == 31
    assert result['demo_preservation_review_required'] is True


@pytest.mark.parametrize('mutation', ['duplicate', 'nan', 'missing'])
def test_incomplete_duplicate_or_nonfinite_receipts_refused(mutation):
    gold, records = fixture()
    if mutation == 'duplicate':
        records[-1] = records[0]
    elif mutation == 'nan':
        records[-1]['latency_ms'] = float('nan')
    else:
        records.pop()
    with pytest.raises(ValueError):
        evaluation.score_records(gold, records)
