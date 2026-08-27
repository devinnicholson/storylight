import json
from pathlib import Path

DASHBOARD = Path("infra/gcp/monitoring/live-scene-dashboard.json")


def test_live_scene_dashboard_tracks_latency_gpu_startup_and_cost() -> None:
    payload = json.loads(DASHBOARD.read_text())
    serialized = json.dumps(payload)

    assert payload["displayName"] == "Bookforge live-scene GPU performance"
    assert "run.googleapis.com/request_latency/e2e_latencies" in serialized
    assert "run.googleapis.com/container/startup_latencies" in serialized
    assert "run.googleapis.com/container/gpu/utilizations" in serialized
    assert "run.googleapis.com/container/gpu/memory_usages" in serialized
    assert "run.googleapis.com/container/instance_count" in serialized
    assert "run.googleapis.com/container/billable_instance_time" in serialized
    assert 'resource.label.\\"service_name\\"=\\"bookforge-scene-rtx\\"' in serialized


def test_live_scene_dashboard_contains_no_story_or_user_dimensions() -> None:
    serialized = DASHBOARD.read_text().casefold()

    for forbidden in ("prompt", "passage", "reader", "audio", "camera", "session_id"):
        assert forbidden not in serialized


def test_live_scene_dashboard_tiles_do_not_overlap() -> None:
    dashboard = json.loads(DASHBOARD.read_text())
    tiles = dashboard["mosaicLayout"]["tiles"]
    occupied: set[tuple[int, int]] = set()

    for tile in tiles:
        assert tile["xPos"] >= 0
        assert tile["yPos"] >= 0
        assert tile["xPos"] + tile["width"] <= dashboard["mosaicLayout"]["columns"]
        cells = {
            (x, y)
            for x in range(tile["xPos"], tile["xPos"] + tile["width"])
            for y in range(tile["yPos"], tile["yPos"] + tile["height"])
        }
        assert occupied.isdisjoint(cells)
        occupied.update(cells)
