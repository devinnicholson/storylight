from pathlib import Path

DASHBOARD = Path("infra/gcp/monitoring/live-scene-dashboard.json")


def test_live_scene_dashboard_contains_no_story_or_user_dimensions() -> None:
    serialized = DASHBOARD.read_text().casefold()

    for forbidden in ("prompt", "passage", "reader", "audio", "camera", "session_id"):
        assert forbidden not in serialized
