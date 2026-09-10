"""Local browser-check server; synthetic fixtures only, never cloud credentials.

Run: PYTHONPATH=src:tests .venv/bin/python tests/serve_prepared_projection_fixture.py
"""

from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import uvicorn
from test_anticipatory_playback import Rig

import storylight.api as api
from storylight.config import Settings


def main():
    with TemporaryDirectory(prefix="storylight-prepared-browser-") as directory:
        settings = Settings(
            _env_file=None,
            model_backend="fake",
            model_name="fake",
            asset_backend="fake",
            live_scene_backend="fake",
            live_scene_planner="deterministic",
            live_scene_planner_auto_warmup=False,
            asr_backend="disabled",
            anticipatory_backend="disabled",
            data_dir=Path(directory) / "data",
            cache_dir=Path(directory) / "cache",
        )
        api.get_settings = lambda: settings
        original_lifespan = api.app.router.lifespan_context

        @asynccontextmanager
        async def fixture_lifespan(app):
            async with original_lifespan(app):
                rig = Rig(Path(directory) / "fixtures")
                await rig.initialize()
                rig.playback.registry = app.state.live_scenes
                rig.playback.cache = app.state.asset_cache
                app.state.anticipatory_playback = rig.playback
                try:
                    yield
                finally:
                    await rig.close()

        api.app.router.lifespan_context = fixture_lifespan
        uvicorn.run(api.app, host="127.0.0.1", port=18085)


if __name__ == "__main__":
    main()
