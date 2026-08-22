from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FrozenStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SupportAction(StrEnum):
    WAIT = "wait"
    HIGHLIGHT_GRAPHEME = "highlight_grapheme"
    SHOW_MOUTH = "show_mouth"
    SPEAK_SOUND = "speak_sound"
    DEFINE_WORD = "define_word"
    CELEBRATE = "celebrate"


class InterventionRequest(StrictModel):
    session_id: NonEmpty
    page_id: NonEmpty
    expected_word: NonEmpty
    expected_grapheme: str = ""
    heard_text: str = ""
    pause_ms: Annotated[int, Field(ge=0, le=30_000)] = 0
    attempt_count: Annotated[int, Field(ge=0, le=10)] = 0
    reader_level: Annotated[int, Field(ge=0, le=12)] = 2
    previous_help: list[SupportAction] = Field(default_factory=list)
    allowed_actions: list[SupportAction] = Field(
        default_factory=lambda: [
            SupportAction.WAIT,
            SupportAction.HIGHLIGHT_GRAPHEME,
            SupportAction.SHOW_MOUTH,
            SupportAction.SPEAK_SOUND,
        ]
    )

    @model_validator(mode="after")
    def require_actions(self) -> "InterventionRequest":
        if not self.allowed_actions:
            raise ValueError("allowed_actions must contain at least one action")
        return self


class InterventionDecision(StrictModel):
    action: SupportAction
    target: str = ""
    display_text: str = ""
    rationale_code: Literal[
        "not_needed",
        "initial_pause",
        "repeated_pause",
        "pronunciation_support",
        "meaning_support",
        "confirmed",
    ]
    confidence: Annotated[float, Field(ge=0, le=1)]


class BookPageInput(StrictModel):
    page_id: NonEmpty
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=4000)]
    art_direction: Annotated[str, StringConstraints(strip_whitespace=True, max_length=1000)] = ""


class StoryCompileRequest(StrictModel):
    story_id: NonEmpty
    title: NonEmpty
    reading_level: Annotated[int, Field(ge=0, le=12)] = 2
    visual_style: NonEmpty = "luminous paper theater"
    pages: Annotated[list[BookPageInput], Field(min_length=1, max_length=12)]


class VisualLayer(StrictModel):
    layer_id: NonEmpty
    kind: Literal["background", "character", "prop", "effect", "typography"]
    prompt: NonEmpty
    z_index: Annotated[int, Field(ge=0, le=20)]
    motion: NonEmpty


class AssetState(StrEnum):
    PENDING = "pending"
    GENERATING = "generating"
    READY = "ready"
    FAILED = "failed"


class AssetKind(StrEnum):
    IMAGE = "image"
    VIDEO_LOOP = "video_loop"
    SPRITE = "sprite"
    PROCEDURAL = "procedural"


class AssetRecord(FrozenStrictModel):
    asset_id: NonEmpty
    page_id: NonEmpty
    layer_id: NonEmpty
    kind: AssetKind
    provider: NonEmpty
    prompt: NonEmpty
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)]
    width: Annotated[int, Field(ge=1, le=8192)]
    height: Annotated[int, Field(ge=1, le=8192)]
    duration_ms: Annotated[int, Field(ge=0, le=300_000)] = 0
    checksum_sha256: Annotated[
        str,
        StringConstraints(pattern=r"^$|^[a-f0-9]{64}$"),
    ] = ""
    storage_uri: str = ""
    local_uri: str = ""
    state: AssetState = AssetState.PENDING
    generation_ms: Annotated[float, Field(ge=0)] = 0

    @model_validator(mode="after")
    def require_ready_asset_location(self) -> "AssetRecord":
        if self.state is AssetState.READY:
            if not self.checksum_sha256:
                raise ValueError("ready assets require checksum_sha256")
            if not self.storage_uri and not self.local_uri:
                raise ValueError("ready assets require storage_uri or local_uri")
        return self


class StoryTrigger(StrictModel):
    trigger_id: NonEmpty
    word: NonEmpty
    occurrence: Annotated[int, Field(ge=1, le=20)] = 1
    action: Literal["reveal", "move", "transform", "open", "glow", "fade"]
    target_layer_id: NonEmpty
    duration_ms: Annotated[int, Field(ge=50, le=10_000)]


class LiteracySupport(StrictModel):
    word: NonEmpty
    grapheme: NonEmpty
    hint_ladder: Annotated[list[NonEmpty], Field(min_length=1, max_length=4)]


class ComprehensionPrompt(StrictModel):
    question: NonEmpty
    expected_concepts: Annotated[list[NonEmpty], Field(min_length=1, max_length=5)]


class GeneratedPagePlan(StrictModel):
    page_id: NonEmpty
    source_text: str = ""
    scene_summary: NonEmpty
    layers: Annotated[list[VisualLayer], Field(min_length=1, max_length=12)]
    triggers: Annotated[list[StoryTrigger], Field(max_length=20)]
    literacy_support: Annotated[list[LiteracySupport], Field(max_length=12)]
    comprehension: Annotated[list[ComprehensionPrompt], Field(max_length=4)]


class GeneratedStoryPlan(StrictModel):
    pages: list[GeneratedPagePlan]


class StoryPack(StrictModel):
    schema_version: Literal["1.1"] = "1.1"
    story_id: NonEmpty
    title: NonEmpty
    reading_level: int
    visual_style: NonEmpty
    compiler_model: NonEmpty
    pages: list[GeneratedPagePlan]
    assets: list[AssetRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "StoryPack":
        page_layers = {
            page.page_id: {layer.layer_id for layer in page.layers} for page in self.pages
        }
        for page in self.pages:
            if not page.source_text.strip():
                raise ValueError(f"Story Pack page {page.page_id!r} requires source_text")
            for trigger in page.triggers:
                if trigger.target_layer_id not in page_layers[page.page_id]:
                    raise ValueError(
                        f"Trigger {trigger.trigger_id!r} references missing layer"
                    )
        seen_asset_ids: set[str] = set()
        for asset in self.assets:
            if asset.asset_id in seen_asset_ids:
                raise ValueError(f"Duplicate asset_id {asset.asset_id!r}")
            seen_asset_ids.add(asset.asset_id)
            if asset.page_id not in page_layers:
                raise ValueError(f"Asset {asset.asset_id!r} references missing page")
            if asset.layer_id not in page_layers[asset.page_id]:
                raise ValueError(f"Asset {asset.asset_id!r} references missing layer")
        return self


class ModelMetrics(StrictModel):
    backend: NonEmpty
    model: NonEmpty
    total_ms: Annotated[float, Field(ge=0)]
    load_ms: Annotated[float, Field(ge=0)] = 0
    input_tokens: Annotated[int, Field(ge=0)] = 0
    output_tokens: Annotated[int, Field(ge=0)] = 0


class CompileResponse(StrictModel):
    story_pack: StoryPack
    metrics: ModelMetrics


class InterventionResponse(StrictModel):
    decision: InterventionDecision
    source: Literal["fast_path", "model"]
    metrics: ModelMetrics | None = None


class ModelProbe(StrictModel):
    ready: bool
    backend: NonEmpty
    model: NonEmpty
    detail: NonEmpty


class TranscriptionResponse(StrictModel):
    text: NonEmpty
    language: NonEmpty
    model: NonEmpty
    total_ms: Annotated[float, Field(ge=0)]
    audio_bytes: Annotated[int, Field(gt=0)]
