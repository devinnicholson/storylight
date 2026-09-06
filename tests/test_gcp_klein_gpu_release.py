"""Missing, stale or partial Monitoring data must never become GPU-release proof."""

import asyncio
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import wait_gcp_klein_gpu_release as guard  # noqa: E402


def test_release_requires_complete_fresh_zero_revisions_and_bounded_readonly_wait(
    tmp_path, monkeypatch
):
    service = "bookforge-klein-qualification-20260906-m"
    after = guard.timestamp("2026-09-06T22:30:00Z")

    def response(now, revisions=(service + "-trial", "revision-b")):
        return {
            "timeSeries": [
                {
                    "metric": {"type": guard.METRIC, "labels": {"state": state}},
                    "resource": {
                        "type": "cloud_run_revision",
                        "labels": {
                            "project_id": guard.PROJECT,
                            "location": guard.REGION,
                            "service_name": service,
                            "revision_name": revision,
                        },
                    },
                    "metricKind": "GAUGE",
                    "valueType": "INT64",
                    "points": [
                        {
                            "interval": {"endTime": guard.utc(now - age)},
                            "value": {"int64Value": "0"},
                        }
                        for age in (60, 120)
                    ],
                }
                for revision in revisions
                for state in ("active", "idle")
            ]
        }

    fresh = response(after + 660)
    valid = guard.evaluate(fresh, service, after, after + 660, set())
    assert valid["proven_zero"] and valid["summed_instance_counts"] == [0, 0]
    assert not guard.evaluate(fresh, service, after, after + 900, set())["proven_zero"]
    assert not guard.evaluate(response(after + 599), service, after, after + 599, set())[
        "proven_zero"
    ]
    assert not guard.evaluate(response(after + 60), service, after, after + 660, set())[
        "proven_zero"
    ]
    mutations = []
    nonzero = copy.deepcopy(fresh)
    nonzero["timeSeries"][2]["points"][0]["value"]["int64Value"] = "1"
    mutations.append(nonzero)
    incomplete = copy.deepcopy(fresh)
    incomplete["timeSeries"][1]["points"].pop(0)
    mutations.append(incomplete)
    newer = copy.deepcopy(fresh)
    newer["timeSeries"][0]["points"].insert(
        0,
        {
            "interval": {"endTime": guard.utc(after + 630)},
            "value": {"int64Value": "0"},
        },
    )
    mutations.extend(
        [
            newer,
            {"timeSeries": fresh["timeSeries"][:-1]},
            {},
            {**fresh, "nextPageToken": "more"},
            {**fresh, "executionErrors": [{}]},
            response(after + 660, ("revision-b",)),
        ]
    )
    wrong = copy.deepcopy(fresh)
    wrong["timeSeries"][0]["resource"]["labels"]["service_name"] = "another-service"
    mutations.append(wrong)
    for data in mutations:
        assert not guard.evaluate(data, service, after, after + 660, set())["proven_zero"]
    seen = set()
    assert guard.evaluate(fresh, service, after, after + 660, seen)["proven_zero"]
    assert not guard.evaluate(
        response(after + 720, (service + "-trial",)), service, after, after + 720, seen
    )["proven_zero"]
    with pytest.raises(ValueError):
        guard.decode(b'{"timeSeries": [], "timeSeries": []}')

    class Clock:
        elapsed = 540.0

        def now(self):
            return after + self.elapsed

        async def sleep(self, delay):
            self.elapsed += delay

    clock, queries = Clock(), []

    async def forbidden(*args, **kwargs):
        raise AssertionError("test must not make network or GPU calls")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden)
    with pytest.raises(ValueError):
        asyncio.run(
            guard.wait_for_release("qualification20260906m", guard.utc(after), tmp_path / "bad")
        )
    assert not (tmp_path / "bad").exists()

    async def reader(params, timeout):
        assert 0 < timeout <= 30
        assert not any("aggregation" in key for key in params)
        assert f'resource.labels.service_name="{service}"' in params["filter"]
        assert "state" not in params["filter"] and "revision_name" not in params["filter"]
        queries.append(params)
        clock.elapsed += 2
        return json.dumps(response(clock.now() - 2)).encode()

    def run(output, selected_reader):
        return asyncio.run(
            guard.wait_for_release(
                service,
                guard.utc(after),
                output,
                reader=selected_reader,
                now=clock.now,
                monotonic=lambda: clock.elapsed,
                sleep=clock.sleep,
            )
        )

    output = tmp_path / "success"
    summary = run(output, reader)
    assert summary["decision"] == "observed_zero" and summary["reads"] == 2
    assert clock.elapsed == 602 and not summary["quota_reserved"]
    assert len(list(output.glob("*.response.json"))) == 2
    assert json.loads((output / "summary.json").read_text()) == summary
    with pytest.raises(FileExistsError):
        run(output, reader)

    async def missing(params, timeout):
        return b"{}"

    started = clock.elapsed
    summary = run(tmp_path / "missing", missing)
    assert summary["decision"] == "unknown" and summary["reads"] == 15
    assert clock.elapsed - started == 900
