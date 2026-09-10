"""Explicit, local-only playback of pre-generated demo scenes."""

from pathlib import Path
from typing import Annotated

from pydantic import Field

from storylight.domain import FrozenStrictModel, StoryPack
from storylight.event_hub import SessionId
from storylight.live_scene import LiveSceneCreateRequest, LiveSceneServerInstanceId


class CueScene(FrozenStrictModel):
    scene_id: str
    title: str
    cues: list[str]
    provider: str
    request: LiveSceneCreateRequest
    pack: StoryPack


class CueCatalog(FrozenStrictModel):
    scenes: Annotated[list[CueScene], Field(min_length=1, max_length=16)]


def load_catalog(path: Path) -> CueCatalog:
    catalog = CueCatalog.model_validate_json(path.read_bytes())
    ids = [scene.scene_id for scene in catalog.scenes]
    if len(ids) != len(set(ids)):
        raise ValueError("Demo scene IDs must be unique")
    if any(not scene.cues or any(not cue.strip() for cue in scene.cues)
           for scene in catalog.scenes):
        raise ValueError("Each demo scene needs nonempty spoken cues")
    return catalog


class CueActivation(FrozenStrictModel):
    scene_id: Annotated[str, Field(min_length=1, max_length=80)]
    session_id: SessionId
    activation_id: Annotated[str, Field(pattern=r"^[a-f0-9-]{36}$")]
    server_instance_id: LiveSceneServerInstanceId
    session_revision: Annotated[int, Field(ge=0)]
