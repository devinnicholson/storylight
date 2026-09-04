from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="BOOKFORGE_",
        extra="ignore",
    )

    environment: str = "development"
    model_backend: Literal["ollama", "openai", "fake"] = "ollama"
    model_name: str = "gemma4:e2b-it-qat"
    model_base_url: str = "http://127.0.0.1:11434"
    model_api_key: str = ""
    model_timeout_seconds: float = 180.0
    model_keep_alive: str = "10m"
    model_context_tokens: Annotated[int, Field(ge=2_048, le=32_768)] = 8_192
    model_max_output_tokens: Annotated[int, Field(ge=64, le=8_192)] = 4_096
    model_require_gpu: bool = False
    asset_backend: Literal["disabled", "modal", "mflux", "fake"] = "disabled"
    asset_model: str = "z-image-turbo"
    asset_command: str = "mflux-generate-z-image-turbo"
    asset_depth_command: str = "mflux-save-depth"
    asset_modal_command: str = "modal"
    asset_modal_app: Path = Path("deploy/modal_scene_foundry.py")
    asset_modal_steps: int = 4
    asset_width: int = 1024
    asset_height: int = 576
    asset_steps: int = 9
    asset_quantize: Literal[3, 4, 5, 6, 8] = 4
    asset_timeout_seconds: float = 1_800.0
    asset_low_ram: bool = True
    live_scene_backend: Literal[
        "auto",
        "disabled",
        "fake",
        "modal",
        "modal_warm",
        "modal_klein",
        "gcp_cloud_run",
        "gcp_resilient",
    ] = "auto"
    live_scene_planner: Literal["deterministic", "model"] = "deterministic"
    live_scene_planner_backend: Literal[
        "configured",
        "tensorrt_slots",
        "tensorrt_hybrid",
    ] = "configured"
    live_scene_planner_base_url: str = "http://127.0.0.1:11435"
    live_scene_planner_model_name: str = "llm"
    live_scene_planner_max_output_tokens: Annotated[int, Field(ge=1, le=128)] = 64
    live_scene_planner_fallback_ready_seconds: Annotated[
        float,
        Field(ge=0, le=30),
    ] = 5.0
    live_scene_planner_timeout_seconds: Annotated[float, Field(gt=0, le=60)] = 12.0
    live_scene_planner_model_revision: Annotated[
        str,
        Field(min_length=1, max_length=200),
    ] = "configured-local-model"
    live_scene_planner_cache_entries: Annotated[int, Field(ge=0, le=256)] = 32
    live_scene_planner_auto_warmup: bool = False
    # Positional short-key output was faster on Gemma 3 1B but repeatedly
    # dropped verbs, settings, and supporting objects. Keep the faithful named
    # schema as the production default; compact remains an explicit research A/B.
    live_scene_planner_compact_wire: bool = False
    live_scene_enable_motion: bool = False
    live_scene_enable_preview: bool = True
    live_scene_output_dir: Path = Path("artifacts/live-scenes/generated")
    live_scene_master_width: int = 1024
    live_scene_master_height: int = 576
    # One-step SANA-Sprint is technically supported but failed the showcase
    # duplicate-subject gate for only a 5.5% end-to-end gain. Production stays
    # on the accepted two-to-four-step range; the finite CLI remains available
    # for explicitly budgeted research.
    live_scene_master_steps: Annotated[int, Field(ge=2, le=4)] = 2
    live_scene_master_guidance_scale: Annotated[float, Field(ge=0, le=12)] = 4.5
    # Inline preserves the fail-closed Grounding DINO gate. Deferred returns the
    # first generated plate immediately so an optional critic can evaluate it
    # after projection without extending time-to-first-image.
    live_scene_fidelity_mode: Literal["inline", "deferred"] = "inline"
    live_scene_auto_prewarm_on_submit: bool = False
    live_scene_modal_session_gpu_cap_usd: Annotated[float, Field(gt=0, le=10)] = 1.0
    live_scene_modal_plan_file: Path = Path("experiments/live-scenes/modal-plan.json")
    live_scene_modal_ledger_path: Path = Path("artifacts/live-scenes/modal-ledger.json")
    live_scene_gcp_url: str = ""
    live_scene_gcp_audience: str = ""
    live_scene_gcp_impersonate_service_account: str = ""
    live_scene_gcp_gpu: Literal["L4", "RTX_PRO_6000"] = "L4"
    live_scene_gcp_timeout_seconds: Annotated[float, Field(gt=0, le=600)] = 180.0
    live_scene_gcp_session_gpu_cap_usd: Annotated[float, Field(gt=0, le=10)] = 0.50
    live_scene_vertex_project_id: str = ""
    live_scene_vertex_location: str = "global"
    live_scene_vertex_model: str = "gemini-3.1-flash-lite-image"
    live_scene_vertex_timeout_seconds: Annotated[float, Field(gt=0, le=300)] = 90.0
    live_scene_vertex_session_cost_cap_usd: Annotated[float, Field(gt=0, le=10)] = 0.50
    live_scene_vertex_estimated_image_usd: Annotated[float, Field(gt=0, le=1)] = 0.034
    live_scene_routing_probe_timeout_seconds: Annotated[float, Field(gt=0, le=60)] = 2.0
    live_scene_routing_failure_cooldown_seconds: Annotated[
        float,
        Field(gt=0, le=3_600),
    ] = 300.0
    live_scene_critic_backend: Literal["disabled", "nemotron"] = "disabled"
    live_scene_critic_url: str = ""
    live_scene_critic_audience: str = ""
    live_scene_critic_api_key: str = ""
    live_scene_critic_model: str = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
    live_scene_critic_timeout_seconds: Annotated[float, Field(gt=0, le=300)] = 90.0
    anticipatory_backend: Literal["disabled", "gke"] = "disabled"
    anticipatory_url: str = ""
    anticipatory_audience: str = ""
    anticipatory_allow_loopback_http: bool = False
    anticipatory_timeout_seconds: Annotated[float, Field(ge=1, le=120)] = 30.0
    anticipatory_edge_gate_revision: Annotated[
        str,
        Field(min_length=1, max_length=120),
    ] = "sanitized-scene-spec-v1"
    live_scene_max_active_jobs: Annotated[int, Field(ge=1, le=16)] = 2
    live_scene_max_retained_jobs: Annotated[int, Field(ge=1, le=256)] = 64
    live_scene_event_queue_size: Annotated[int, Field(ge=1, le=128)] = 8
    asr_backend: Literal["mlx_whisper", "whisper_trt", "disabled", "test"] = "mlx_whisper"
    asr_model: str = ".models/whisper-base.en"
    asr_engine_path: str = ""
    asr_max_audio_mb: int = 20
    data_dir: Path = Path(".bookforge/data")
    cache_dir: Path = Path(".bookforge/cache")
    allowed_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("live_scene_master_width", "live_scene_master_height")
    @classmethod
    def validate_live_scene_master_dimension(cls, value: int) -> int:
        if not 512 <= value <= 1_536 or value % 32:
            raise ValueError("live-scene master dimensions must be 512..1536 and divisible by 32")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
