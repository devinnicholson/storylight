from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import signal
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from bookforge.asset_cache import AssetCache
from bookforge.domain import AssetKind, AssetRecord, AssetRole, AssetState, StoryPack
from bookforge.live_scene import (
    DETERMINISTIC_LIVE_SCENE_COMPILER_MODEL,
    LiveSceneArtifact,
    LiveSceneArtifactKind,
    LiveSceneCostSource,
    LiveSceneCreateRequest,
    LiveSceneMetrics,
    LiveSceneModelProvenance,
    LiveScenePlanningStatus,
    LiveSceneProviderUnavailableError,
    LiveSceneStage,
    LiveSceneUpdate,
    LiveSceneWarmState,
    build_live_scene_story_pack,
    build_planned_live_scene_story_pack,
    live_scene_request_seed,
)
from bookforge.live_scene_planner import (
    LiveScenePlanner,
    LiveScenePlannerError,
)
from bookforge.modal_budget import (
    authorize_and_reserve_modal_budget,
    release_modal_budget_reservation,
    settle_modal_budget,
)
from bookforge.visual_evaluation import MediaEvaluation, evaluate_media
from bookforge.visual_lab import GPU_USD_PER_SECOND, GenerationRecord

FAST_MODEL = "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers"
FAST_MODEL_REVISION = "19683c58b7ea290e55cedd8950ae1d86ada7ef96"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_MODEL_REVISION = "b4769fd619394250528294b658587285526fab1c"
DEPTH_DTYPE = "float16"
FIDELITY_MODEL = "IDEA-Research/grounding-dino-tiny"
FIDELITY_MODEL_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
MOTION_MODEL = "Lightricks/LTX-Video"
MOTION_MODEL_REVISION = "a6d59ee37c13c58261aa79027d3e41cd41960925"
PROVIDER_NAME = "modal-finite"


class FiniteModalProviderError(RuntimeError):
    pass


class FiniteModalUnavailableError(FiniteModalProviderError):
    pass


class FiniteModalBudgetError(FiniteModalProviderError):
    pass


@dataclass(frozen=True, slots=True)
class ModalStagePolicy:
    gpu: str
    remote_timeout_seconds: int
    command_timeout_seconds: int
    maximum_gpu_usd: float

    def __post_init__(self) -> None:
        if self.gpu not in GPU_USD_PER_SECOND:
            raise ValueError(f"unknown Modal GPU price: {self.gpu}")
        if self.remote_timeout_seconds < 1:
            raise ValueError("remote_timeout_seconds must be positive")
        if self.command_timeout_seconds < self.remote_timeout_seconds:
            raise ValueError("command timeout cannot be shorter than the remote timeout")
        if not math.isfinite(self.maximum_gpu_usd) or self.maximum_gpu_usd <= 0:
            raise ValueError("maximum_gpu_usd must be finite and positive")
        if self.worst_case_gpu_usd > self.maximum_gpu_usd + 1e-9:
            raise ValueError(
                f"{self.gpu} worst-case cost ${self.worst_case_gpu_usd:.6f} exceeds "
                f"the ${self.maximum_gpu_usd:.6f} stage cap"
            )

    @property
    def worst_case_gpu_usd(self) -> float:
        # Use the complete local command lifetime, not merely Modal's function
        # timeout, so cold model loading is covered by the reservation.
        return GPU_USD_PER_SECOND[self.gpu] * self.command_timeout_seconds


FAST_GPU = "L4"
WARM_FAST_GPU = "L40S"
MOTION_GPU = "L4"

FAST_STAGE_POLICY = ModalStagePolicy(
    gpu=FAST_GPU,
    remote_timeout_seconds=180,
    command_timeout_seconds=360,
    maximum_gpu_usd=0.08,
)
WARM_FAST_STAGE_POLICY = ModalStagePolicy(
    gpu=WARM_FAST_GPU,
    remote_timeout_seconds=180,
    command_timeout_seconds=360,
    maximum_gpu_usd=0.22,
)
MOTION_STAGE_POLICY = ModalStagePolicy(
    gpu=MOTION_GPU,
    remote_timeout_seconds=300,
    command_timeout_seconds=480,
    maximum_gpu_usd=0.12,
)
WARM_FAST_SESSION_CEILING_USD = 0.30
WARM_FAST_PRESENTATION_SESSION_CEILING_USD = 0.70
WARM_FULL_SESSION_CEILING_USD = 0.60
WARM_SCALEDOWN_WINDOW_SECONDS = 90
WARM_MAX_SCALEDOWN_WINDOW_SECONDS = 900


@dataclass(frozen=True, slots=True)
class FastSceneRequest:
    scene_id: str
    prompt: str
    negative_prompt: str = (
        "words, letters, captions, logo, watermark, interface, border, split screen, "
        "collage, photorealistic, plastic 3d render, distorted anatomy, duplicate character"
    )
    seed: int = 42
    width: int = 1024
    height: int = 576
    steps: int = 2
    guidance_scale: float = 4.5
    fidelity_label: str = ""
    fidelity_object_label: str = ""
    require_subject_object_overlap: bool = False
    expected_subject_count: int = 1

    def __post_init__(self) -> None:
        _validate_identifier(self.scene_id)
        _validate_prompt(self.prompt, name="prompt")
        _validate_prompt(self.negative_prompt, name="negative_prompt")
        _validate_seed(self.seed)
        _validate_dimensions(self.width, self.height, minimum=512)
        if not 1 <= self.steps <= 4:
            raise ValueError("fast-scene steps must be between 1 and 4")
        if not math.isfinite(self.guidance_scale) or not 0 <= self.guidance_scale <= 12:
            raise ValueError("guidance_scale must be between 0 and 12")
        if len(self.fidelity_label) > 80:
            raise ValueError("fidelity_label cannot exceed 80 characters")
        if self.fidelity_label and not self.fidelity_label.strip():
            raise ValueError("fidelity_label cannot be whitespace")
        if len(self.fidelity_object_label) > 80:
            raise ValueError("fidelity_object_label cannot exceed 80 characters")
        if self.fidelity_object_label and not self.fidelity_object_label.strip():
            raise ValueError("fidelity_object_label cannot be whitespace")
        if self.require_subject_object_overlap and not self.fidelity_object_label:
            raise ValueError("subject/object overlap requires an object label")
        if not 1 <= self.expected_subject_count <= 4:
            raise ValueError("expected_subject_count must be between 1 and 4")


@dataclass(frozen=True, slots=True)
class FastPreviewRequest:
    scene_id: str
    prompt: str
    seed: int = 42
    width: int = 512
    height: int = 288
    guidance_scale: float = 4.5

    def __post_init__(self) -> None:
        _validate_identifier(self.scene_id)
        _validate_prompt(self.prompt, name="prompt")
        _validate_seed(self.seed)
        _validate_dimensions(self.width, self.height, minimum=256)
        if self.width > 640 or self.height > 384:
            raise ValueError("preview dimensions cannot exceed 640x384")
        if not math.isfinite(self.guidance_scale) or not 0 <= self.guidance_scale <= 12:
            raise ValueError("guidance_scale must be between 0 and 12")


@dataclass(frozen=True, slots=True)
class MotionUpgradeRequest:
    prompt: str = (
        "Locked storybook camera. Preserve every character, object, shape, and color. "
        "Add gentle breathing, blinking, drifting paper motes, and warm light. Subtle readable "
        "ambient motion only; no scene transition or new content."
    )
    negative_prompt: str = (
        "camera shake, fast motion, hard cut, scene change, zoom, morphing, melting, flicker, "
        "jitter, inconsistent character, blurry, distorted, text, watermark"
    )
    seed: int = 43
    width: int = 800
    height: int = 448
    generated_frames: int = 41
    fps: int = 24
    steps: int = 24

    def __post_init__(self) -> None:
        _validate_prompt(self.prompt, name="prompt")
        _validate_prompt(self.negative_prompt, name="negative_prompt")
        _validate_seed(self.seed)
        _validate_dimensions(self.width, self.height, minimum=384)
        if not 9 <= self.generated_frames <= 65 or (self.generated_frames - 1) % 8:
            raise ValueError("generated_frames must be 8n+1 and between 9 and 65")
        if not 12 <= self.fps <= 30:
            raise ValueError("fps must be between 12 and 30")
        if not 8 <= self.steps <= 40:
            raise ValueError("motion steps must be between 8 and 40")


@dataclass(frozen=True, slots=True)
class SceneArtifact:
    role: str
    path: Path
    sha256: str
    mime_type: str
    width: int
    height: int
    duration_ms: int = 0
    frames: int = 1
    fps: int = 0


@dataclass(frozen=True, slots=True)
class FiniteSceneBundle:
    manifest_path: Path
    scene_id: str
    artifacts: Mapping[str, SceneArtifact]
    manifest: Mapping[str, Any]

    @property
    def master(self) -> SceneArtifact:
        return self.artifacts["master"]

    @property
    def depth(self) -> SceneArtifact:
        return self.artifacts["depth"]

    @property
    def motion(self) -> SceneArtifact | None:
        return self.artifacts.get("motion")

    @property
    def estimated_gpu_usd(self) -> float:
        stages = self.manifest["stages"]
        return sum(float(stage["estimated_gpu_usd"]) for stage in stages.values())


Runner = Callable[[list[str], float], Awaitable[None]]
BillingReader = Callable[[str, float], Awaitable[float]]


class WarmModalInvoker(Protocol):
    async def probe(self) -> tuple[bool, str]: ...

    async def invoke(
        self,
        class_name: str,
        method_name: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]: ...

    async def configure_scaledown_window(
        self,
        class_name: str,
        seconds: int,
    ) -> None: ...


class ModalSdkWarmInvoker:
    """Authenticated SDK lookup for deployed, scale-to-zero Modal classes."""

    def __init__(
        self,
        *,
        app_name: str = "bookforge-fast-scene",
        environment_name: str | None = None,
        fast_gpu: str = WARM_FAST_GPU,
    ) -> None:
        if fast_gpu not in GPU_USD_PER_SECOND:
            raise ValueError(f"unknown warm Modal GPU price: {fast_gpu}")
        self.app_name = app_name
        self.environment_name = environment_name
        self.fast_gpu = fast_gpu
        self._classes: dict[str, Any] = {}
        self._objects: dict[str, Any] = {}

    async def probe(self) -> tuple[bool, str]:
        try:
            importlib.import_module("modal")
        except ImportError:
            return False, "Modal SDK is not installed; install bookforge[modal-authoring]"
        try:
            await self._deployed_class("FastSceneStudio")
            await self._deployed_class("MotionUpgradeStudio")
        except Exception as error:
            return False, f"deployed Modal classes are unreachable: {error}"
        return True, "Authenticated deployed Modal classes are reachable"

    async def _deployed_class(self, class_name: str) -> Any:
        deployed_class = self._classes.get(class_name)
        if deployed_class is None:
            modal = importlib.import_module("modal")
            base_class = modal.Cls.from_name(
                self.app_name,
                class_name,
                environment_name=self.environment_name,
            )
            # Hydration performs an authenticated metadata lookup. It verifies
            # deployment without entering a container or allocating a GPU.
            await base_class.hydrate.aio()
            deployed_class = base_class
            if class_name == "FastSceneStudio" and self.fast_gpu != FAST_GPU:
                deployed_class = base_class.with_options(
                    gpu=self.fast_gpu,
                    max_containers=1,
                    scaledown_window=WARM_SCALEDOWN_WINDOW_SECONDS,
                    timeout=FAST_STAGE_POLICY.remote_timeout_seconds,
                )
                await deployed_class.hydrate.aio()
            self._classes[class_name] = deployed_class
        return deployed_class

    async def invoke(
        self,
        class_name: str,
        method_name: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        instance = await self._deployed_object(class_name)
        remote_method = getattr(instance, method_name)
        result = await remote_method.remote.aio(**dict(arguments))
        if not isinstance(result, dict):
            raise FiniteModalProviderError(
                f"deployed Modal class {class_name}.{method_name} returned invalid data"
            )
        return result

    async def _deployed_object(self, class_name: str) -> Any:
        instance = self._objects.get(class_name)
        if instance is None:
            deployed_class = await self._deployed_class(class_name)
            instance = deployed_class()
            self._objects[class_name] = instance
        return instance

    async def configure_scaledown_window(self, class_name: str, seconds: int) -> None:
        instance = await self._deployed_object(class_name)
        await asyncio.to_thread(instance.update_autoscaler, scaledown_window=seconds)


@dataclass(frozen=True, slots=True)
class WarmPrewarmReport:
    prewarm_id: str
    reservation_id: str
    include_motion: bool
    fast_remote_seconds: float
    motion_remote_seconds: float
    full_session_ceiling_usd: float
    fast_model_load_seconds: float
    motion_model_load_seconds: float
    expires_in_seconds: float
    scaledown_window_seconds: int = WARM_SCALEDOWN_WINDOW_SECONDS
    fast_inference_warmup_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class WarmProviderStatus:
    ready: bool
    detail: str
    state: str
    prewarm_id: str | None = None
    include_motion: bool = False
    expires_in_seconds: float = 0.0
    scaledown_window_seconds: int = WARM_SCALEDOWN_WINDOW_SECONDS


@dataclass(slots=True)
class _ActiveWarmSession:
    prewarm_id: str
    experiment_id: str
    reservation_id: str
    include_motion: bool
    full_session_ceiling_usd: float
    fast_prewarm_seconds: float
    motion_prewarm_seconds: float
    fast_model_load_seconds: float
    motion_model_load_seconds: float
    receipt_path: Path
    deadline_monotonic: float
    scaledown_window_seconds: int = WARM_SCALEDOWN_WINDOW_SECONDS
    fast_scene_seconds: float = 0
    preview_scene_seconds: float = 0
    preview_scene_id: str | None = None
    scene_id: str | None = None
    scene_master: SceneArtifact | None = None


@dataclass(frozen=True, slots=True)
class _PreparedFastAuthorization:
    scene_id: str
    reservation_id: str


@dataclass(frozen=True, slots=True)
class MotionTechnicalEvidence:
    sha256: str
    duration_seconds: float
    fps: float
    endpoint_ssim: float
    motion_stability: float
    projection_legibility: float


@dataclass(frozen=True, slots=True)
class MotionTechnicalGate:
    """Technical promotion gate; semantic identity still requires human review."""

    minimum_duration_seconds: float = 3.0
    maximum_duration_seconds: float = 8.0
    minimum_fps: float = 20.0
    minimum_endpoint_ssim: float = 0.94
    minimum_motion_stability: float = 0.80
    minimum_projection_legibility: float = 0.20

    def rejection_reasons(self, evidence: MotionTechnicalEvidence) -> list[str]:
        reasons: list[str] = []
        if (
            not self.minimum_duration_seconds
            <= evidence.duration_seconds
            <= (self.maximum_duration_seconds)
        ):
            reasons.append(
                f"duration {evidence.duration_seconds:.3f}s is outside "
                f"{self.minimum_duration_seconds:g}-{self.maximum_duration_seconds:g}s"
            )
        if evidence.fps < self.minimum_fps:
            reasons.append(f"fps {evidence.fps:.3f} is below {self.minimum_fps:g}")
        if evidence.endpoint_ssim < self.minimum_endpoint_ssim:
            reasons.append(
                f"endpoint SSIM {evidence.endpoint_ssim:.6f} is below "
                f"{self.minimum_endpoint_ssim:g}"
            )
        if evidence.motion_stability < self.minimum_motion_stability:
            reasons.append(
                f"motion stability {evidence.motion_stability:.6f} is below "
                f"{self.minimum_motion_stability:g}"
            )
        if evidence.projection_legibility < self.minimum_projection_legibility:
            reasons.append(
                f"projection legibility {evidence.projection_legibility:.6f} is below "
                f"{self.minimum_projection_legibility:g}"
            )
        return reasons


MotionEvaluator = Callable[[Path], MotionTechnicalEvidence]


class FiniteModalSceneProvider:
    """Finite ``modal run`` adapter for staged, on-demand projection scenes.

    The first call returns a master and depth map. The optional second call upgrades
    that exact master to a seamless LTX-Video loop. The adapter never deploys or
    addresses a persistent web endpoint.
    """

    def __init__(
        self,
        *,
        modal_executable: str = "modal",
        modal_app: Path = Path("deploy/modal_fast_scene.py"),
        plan_file: Path = Path("experiments/live-scenes/modal-plan.json"),
        ledger_path: Path = Path("artifacts/live-scenes/modal-ledger.json"),
        fast_policy: ModalStagePolicy = FAST_STAGE_POLICY,
        motion_policy: ModalStagePolicy = MOTION_STAGE_POLICY,
        session_gpu_cap_usd: float = 15.0,
        runner: Runner | None = None,
        billing_reader: BillingReader | None = None,
        billing_timeout_seconds: float = 60.0,
    ) -> None:
        if not math.isfinite(session_gpu_cap_usd) or session_gpu_cap_usd <= 0:
            raise ValueError("session_gpu_cap_usd must be finite and positive")
        if not math.isfinite(billing_timeout_seconds) or billing_timeout_seconds <= 0:
            raise ValueError("billing_timeout_seconds must be finite and positive")
        self.modal_executable = modal_executable
        self.modal_app = modal_app
        self.plan_file = plan_file
        self.ledger_path = ledger_path
        self.fast_policy = fast_policy
        self.motion_policy = motion_policy
        self.session_gpu_cap_usd = session_gpu_cap_usd
        self._runner = runner or _run_command
        self._billing_reader = billing_reader or _read_authoritative_modal_total
        self.billing_timeout_seconds = billing_timeout_seconds
        self.last_authoritative_workspace_usd: float | None = None
        self._session_estimated_gpu_usd = 0.0
        self._active_reservations_gpu_usd = 0.0
        self._failed_reservations_gpu_usd = 0.0

    async def probe(self) -> tuple[bool, str]:
        if shutil.which(self.modal_executable) is None:
            return False, f"Modal command was not found: {self.modal_executable}"
        if not self.modal_app.is_file():
            return False, f"finite Modal app was not found: {self.modal_app}"
        if not self.plan_file.is_file():
            return False, f"Modal budget plan was not found: {self.plan_file}"
        return True, "Finite Modal scene provider is ready"

    async def generate_fast(
        self,
        request: FastSceneRequest,
        *,
        output_dir: Path,
    ) -> FiniteSceneBundle:
        await self._require_ready()
        destination = output_dir.resolve()
        manifest_path = destination / "scene.manifest.json"
        self._require_new_output(manifest_path)
        self._reserve(self.fast_policy)
        experiment_id = _fast_experiment_id(request)
        try:
            reservation_id = await self._authorize_with_current_billing(
                self.fast_policy,
                experiment_id=experiment_id,
            )
        except BaseException:
            self._active_reservations_gpu_usd -= self.fast_policy.worst_case_gpu_usd
            raise
        command = [
            self.modal_executable,
            "run",
            "--quiet",
            f"{self.modal_app}::fast_scene_cli",
            "--scene-id",
            request.scene_id,
            "--prompt",
            request.prompt,
            "--negative-prompt",
            request.negative_prompt,
            "--seed",
            str(request.seed),
            "--width",
            str(request.width),
            "--height",
            str(request.height),
            "--steps",
            str(request.steps),
            "--guidance-scale",
            str(request.guidance_scale),
            "--fidelity-label",
            request.fidelity_label,
            "--fidelity-object-label",
            request.fidelity_object_label,
            *(
                ["--require-subject-object-overlap"]
                if request.require_subject_object_overlap
                else []
            ),
            "--expected-subject-count",
            str(request.expected_subject_count),
            "--output-dir",
            str(destination),
            "--plan-file",
            str(self.plan_file),
            "--ledger-path",
            str(self.ledger_path),
            "--maximum-gpu-usd",
            str(self.fast_policy.maximum_gpu_usd),
            "--experiment-id",
            experiment_id,
            "--reservation-id",
            reservation_id,
        ]
        await self._execute(command, self.fast_policy)
        bundle = load_finite_scene_bundle(manifest_path)
        if bundle.scene_id != request.scene_id or bundle.motion is not None:
            raise FiniteModalProviderError("fast scene manifest does not match the request")
        self._settle(bundle, stage="fast")
        return bundle

    async def upgrade_motion(
        self,
        bundle: FiniteSceneBundle | Path,
        request: MotionUpgradeRequest,
    ) -> FiniteSceneBundle:
        await self._require_ready()
        source = load_finite_scene_bundle(bundle) if isinstance(bundle, Path) else bundle
        if source.motion is not None:
            raise FiniteModalProviderError("scene already contains a motion upgrade")
        self._reserve(self.motion_policy)
        experiment_id = _motion_experiment_id(source, request)
        try:
            reservation_id = await self._authorize_with_current_billing(
                self.motion_policy,
                experiment_id=experiment_id,
            )
        except BaseException:
            self._active_reservations_gpu_usd -= self.motion_policy.worst_case_gpu_usd
            raise
        command = [
            self.modal_executable,
            "run",
            "--quiet",
            f"{self.modal_app}::motion_upgrade_cli",
            "--scene-manifest-path",
            str(source.manifest_path),
            "--prompt",
            request.prompt,
            "--negative-prompt",
            request.negative_prompt,
            "--seed",
            str(request.seed),
            "--width",
            str(request.width),
            "--height",
            str(request.height),
            "--generated-frames",
            str(request.generated_frames),
            "--fps",
            str(request.fps),
            "--steps",
            str(request.steps),
            "--plan-file",
            str(self.plan_file),
            "--ledger-path",
            str(self.ledger_path),
            "--maximum-gpu-usd",
            str(self.motion_policy.maximum_gpu_usd),
            "--experiment-id",
            experiment_id,
            "--reservation-id",
            reservation_id,
        ]
        await self._execute(command, self.motion_policy)
        upgraded = load_finite_scene_bundle(source.manifest_path)
        if upgraded.master.sha256 != source.master.sha256 or upgraded.motion is None:
            raise FiniteModalProviderError("motion upgrade did not preserve its source master")
        self._settle(upgraded, stage="motion")
        return upgraded

    async def _require_ready(self) -> None:
        ready, detail = await self.probe()
        if not ready:
            raise FiniteModalUnavailableError(detail)

    async def _authorize_with_current_billing(
        self,
        policy: ModalStagePolicy,
        *,
        experiment_id: str,
    ) -> str:
        return await self._reserve_against_current_billing(
            experiment_id=experiment_id,
            full_call_ceiling_usd=policy.worst_case_gpu_usd,
        )

    async def _reserve_against_current_billing(
        self,
        *,
        experiment_id: str,
        full_call_ceiling_usd: float,
    ) -> str:
        try:
            workspace_total = await self._billing_reader(
                self.modal_executable,
                self.billing_timeout_seconds,
            )
            reservation_id = await asyncio.to_thread(
                authorize_and_reserve_modal_budget,
                plan_path=self.plan_file,
                ledger_path=self.ledger_path,
                authoritative_workspace_usd=workspace_total,
                experiment_id=experiment_id,
                full_call_ceiling_usd=full_call_ceiling_usd,
            )
        except ValueError as error:
            raise FiniteModalBudgetError(str(error)) from error
        except (OSError, KeyError, json.JSONDecodeError) as error:
            raise FiniteModalBudgetError(
                "authoritative Modal billing could not be verified; paid call rejected"
            ) from error
        if not math.isfinite(workspace_total) or workspace_total < 0:
            raise FiniteModalBudgetError("authoritative Modal billing total is invalid")
        self.last_authoritative_workspace_usd = workspace_total
        return reservation_id

    def _reserve(self, policy: ModalStagePolicy) -> None:
        projected = (
            self._session_estimated_gpu_usd
            + self._active_reservations_gpu_usd
            + self._failed_reservations_gpu_usd
            + policy.worst_case_gpu_usd
        )
        if projected > self.session_gpu_cap_usd + 1e-9:
            raise FiniteModalBudgetError(
                f"finite Modal session cap would be exceeded: ${projected:.6f} projected, "
                f"${self.session_gpu_cap_usd:.6f} allowed"
            )
        self._active_reservations_gpu_usd += policy.worst_case_gpu_usd

    async def _execute(self, command: list[str], policy: ModalStagePolicy) -> None:
        try:
            await self._runner(command, policy.command_timeout_seconds)
        except BaseException:
            # A provider or transport failure has unknown billable duration. Keep
            # the full reservation charged in memory and let the persisted Modal
            # ledger retain its reservation as a second fail-closed guard.
            self._active_reservations_gpu_usd -= policy.worst_case_gpu_usd
            self._failed_reservations_gpu_usd += policy.worst_case_gpu_usd
            raise

    def _settle(self, bundle: FiniteSceneBundle, *, stage: str) -> None:
        record = bundle.manifest["stages"].get(stage)
        if not isinstance(record, dict):
            raise FiniteModalProviderError(f"scene manifest has no {stage} stage")
        amount = float(record["estimated_gpu_usd"])
        policy = self.fast_policy if stage == "fast" else self.motion_policy
        if not math.isfinite(amount) or amount < 0 or amount > policy.maximum_gpu_usd + 1e-9:
            raise FiniteModalBudgetError(f"{stage} manifest reported invalid cost ${amount}")
        self._active_reservations_gpu_usd -= policy.worst_case_gpu_usd
        self._session_estimated_gpu_usd += amount

    @staticmethod
    def _require_new_output(manifest_path: Path) -> None:
        if manifest_path.exists():
            raise FiniteModalProviderError(f"scene output already exists: {manifest_path}")


class WarmModalSceneProvider(FiniteModalSceneProvider):
    """Authenticated deployed-class provider with explicit, bounded prewarming."""

    def __init__(
        self,
        *,
        invoker: WarmModalInvoker | None = None,
        prewarm_fidelity: bool = True,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("fast_policy", WARM_FAST_STAGE_POLICY)
        super().__init__(**kwargs)
        self.invoker = invoker or ModalSdkWarmInvoker()
        self.prewarm_fidelity = prewarm_fidelity
        self._warm_session: _ActiveWarmSession | None = None
        self._prepared_fast_authorizations: dict[str, _PreparedFastAuthorization] = {}
        self._operation_lock = asyncio.Lock()
        self._configured_scaledown_window_seconds = WARM_SCALEDOWN_WINDOW_SECONDS
        self._remote_warm_deadline_monotonic = 0.0

    async def probe(self) -> tuple[bool, str]:
        ready, detail = await super().probe()
        if not ready:
            return ready, detail
        return await self.invoker.probe()

    async def prewarm(
        self,
        *,
        prewarm_id: str,
        include_motion: bool = False,
        scaledown_window_seconds: int = WARM_SCALEDOWN_WINDOW_SECONDS,
    ) -> WarmPrewarmReport:
        _validate_identifier(prewarm_id)
        if not (
            WARM_SCALEDOWN_WINDOW_SECONDS
            <= scaledown_window_seconds
            <= WARM_MAX_SCALEDOWN_WINDOW_SECONDS
        ):
            raise ValueError(
                f"scaledown window must be {WARM_SCALEDOWN_WINDOW_SECONDS}-"
                f"{WARM_MAX_SCALEDOWN_WINDOW_SECONDS} seconds"
            )
        if include_motion and scaledown_window_seconds != WARM_SCALEDOWN_WINDOW_SECONDS:
            raise ValueError("extended presentation warming is supported for master/depth only")
        await self._require_ready()
        async with self._operation_lock:
            await self._expire_warm_session_locked()
            if self._warm_session is not None:
                raise FiniteModalProviderError("a prewarmed scene session is already active")
            experiment_id = f"warm-session:{prewarm_id}"
            session_ceiling = (
                WARM_FULL_SESSION_CEILING_USD
                if include_motion
                else (
                    WARM_FAST_PRESENTATION_SESSION_CEILING_USD
                    if scaledown_window_seconds > WARM_SCALEDOWN_WINDOW_SECONDS
                    else WARM_FAST_SESSION_CEILING_USD
                )
            )
            reservation_id = await self._reserve_against_current_billing(
                experiment_id=experiment_id,
                full_call_ceiling_usd=session_ceiling,
            )

            async def timed(class_name: str) -> tuple[float, dict[str, Any]]:
                started = time.perf_counter()
                arguments = (
                    {"include_fidelity": self.prewarm_fidelity}
                    if class_name == "FastSceneStudio"
                    else {}
                )
                result = await self.invoker.invoke(class_name, "prewarm", arguments)
                return time.perf_counter() - started, result

            # Motion is deliberately opt-in. A master-only session never starts
            # or pays for the LTX container. When requested, the two independently
            # scaling classes load concurrently. Extended presentation warming is
            # master/depth-only and remains bounded by Modal's idle scale-down timer.
            try:
                await self.invoker.configure_scaledown_window(
                    "FastSceneStudio",
                    scaledown_window_seconds,
                )
                self._configured_scaledown_window_seconds = scaledown_window_seconds
                if include_motion:
                    await self.invoker.configure_scaledown_window(
                        "MotionUpgradeStudio",
                        WARM_SCALEDOWN_WINDOW_SECONDS,
                    )
                    (fast_seconds, fast), (motion_seconds, motion) = await asyncio.gather(
                        timed("FastSceneStudio"),
                        timed("MotionUpgradeStudio"),
                    )
                else:
                    fast_seconds, fast = await timed("FastSceneStudio")
                    motion_seconds, motion = 0.0, {}
                if fast.get("gpu") != self.fast_policy.gpu:
                    raise FiniteModalProviderError("warm fast prewarm ran on an unexpected GPU")
                if include_motion and motion.get("gpu") != self.motion_policy.gpu:
                    raise FiniteModalProviderError("warm motion prewarm ran on an unexpected GPU")
            except BaseException:
                # The cross-process reservation deliberately remains live: a
                # cancelled SDK call may still execute and bill remotely.
                self._warm_session = None
                raise
            receipt_path = self.ledger_path.parent / "warm-prewarm" / f"{prewarm_id}.json"
            _atomic_write_json(
                receipt_path,
                {
                    "schema_version": "1.0",
                    "prewarm_id": prewarm_id,
                    "include_motion": include_motion,
                    "fast_remote_seconds": fast_seconds,
                    "motion_remote_seconds": motion_seconds,
                    "scaledown_window_seconds": scaledown_window_seconds,
                    "fast_inference_warmup_seconds": float(fast.get("inference_warmup_seconds", 0)),
                    "created_at": datetime.now(UTC).isoformat(),
                },
            )
            session = _ActiveWarmSession(
                prewarm_id=prewarm_id,
                experiment_id=experiment_id,
                reservation_id=reservation_id,
                include_motion=include_motion,
                full_session_ceiling_usd=session_ceiling,
                fast_prewarm_seconds=fast_seconds,
                motion_prewarm_seconds=motion_seconds,
                fast_model_load_seconds=float(fast.get("model_load_seconds", 0)),
                motion_model_load_seconds=float(motion.get("model_load_seconds", 0)),
                receipt_path=receipt_path,
                deadline_monotonic=time.monotonic() + scaledown_window_seconds,
                scaledown_window_seconds=scaledown_window_seconds,
            )
            self._warm_session = session
            self._remote_warm_deadline_monotonic = session.deadline_monotonic
            return WarmPrewarmReport(
                prewarm_id=prewarm_id,
                reservation_id=reservation_id,
                include_motion=include_motion,
                fast_remote_seconds=fast_seconds,
                motion_remote_seconds=motion_seconds,
                full_session_ceiling_usd=session_ceiling,
                fast_model_load_seconds=session.fast_model_load_seconds,
                motion_model_load_seconds=session.motion_model_load_seconds,
                expires_in_seconds=scaledown_window_seconds,
                scaledown_window_seconds=scaledown_window_seconds,
                fast_inference_warmup_seconds=float(fast.get("inference_warmup_seconds", 0)),
            )

    async def warm_status(self) -> WarmProviderStatus:
        ready, detail = await self.probe()
        async with self._operation_lock:
            await self._expire_warm_session_locked()
            session = self._warm_session
            if session is None:
                return WarmProviderStatus(
                    ready=ready,
                    detail=detail,
                    state="idle",
                    scaledown_window_seconds=self._configured_scaledown_window_seconds,
                )
            return WarmProviderStatus(
                ready=ready,
                detail=detail,
                state="prewarmed",
                prewarm_id=session.prewarm_id,
                include_motion=session.include_motion,
                expires_in_seconds=max(
                    0.0,
                    session.deadline_monotonic - time.monotonic(),
                ),
                scaledown_window_seconds=session.scaledown_window_seconds,
            )

    async def is_prewarmed(self) -> bool:
        """Check local session state without a second deployed-class metadata probe."""

        async with self._operation_lock:
            await self._expire_warm_session_locked()
            return self._warm_session is not None

    async def is_renderer_likely_warm(self) -> bool:
        """Return a same-process hint; authorization is still required per scene."""

        async with self._operation_lock:
            await self._expire_warm_session_locked()
            return self._warm_session is not None or (
                time.monotonic() < self._remote_warm_deadline_monotonic
            )

    async def generate_preview(
        self,
        request: FastPreviewRequest,
        *,
        output_dir: Path,
    ) -> FiniteSceneBundle:
        """Generate one provisional plate under its own or the active session reservation."""

        await self._require_ready()
        destination = output_dir.resolve()
        manifest_path = destination / "preview.manifest.json"
        self._require_new_output(manifest_path)
        async with self._operation_lock:
            if await self._expire_warm_session_locked():
                raise FiniteModalProviderError(
                    "explicit Modal prewarm expired; prewarm again before preview"
                )
            session = self._warm_session
            uses_session = session is not None
            prepared = self._prepared_fast_authorizations.get(request.scene_id)
            if session is not None and prepared is not None:
                raise FiniteModalProviderError(
                    "preview cannot use both a prewarm session and a prepared authorization"
                )
            if session is None and prepared is not None:
                self._prepared_fast_authorizations.pop(request.scene_id, None)
            if session is not None and session.preview_scene_id is not None:
                raise FiniteModalProviderError(
                    "the active prewarm session already produced a preview"
                )
            experiment_id = (
                session.experiment_id
                if session is not None
                else f"fast-authorization:{request.scene_id}"
            )
            reservation_id = (
                session.reservation_id if session else prepared.reservation_id if prepared else ""
            )
            if not uses_session and prepared is None:
                self._reserve(self.fast_policy)
                try:
                    reservation_id = await self._authorize_with_current_billing(
                        self.fast_policy,
                        experiment_id=experiment_id,
                    )
                except BaseException:
                    self._active_reservations_gpu_usd -= self.fast_policy.worst_case_gpu_usd
                    raise
            started = time.perf_counter()
            try:
                result = await self.invoker.invoke(
                    "FastSceneStudio",
                    "generate_preview",
                    {
                        "prompt": request.prompt,
                        "seed": request.seed,
                        "width": request.width,
                        "height": request.height,
                        "guidance_scale": request.guidance_scale,
                    },
                )
            except BaseException:
                if uses_session:
                    self._warm_session = None
                else:
                    self._mark_failed(self.fast_policy)
                raise
            remote_seconds = time.perf_counter() - started
            self._remote_warm_deadline_monotonic = (
                time.monotonic() + self._configured_scaledown_window_seconds
            )
            estimated_gpu_usd = remote_seconds * GPU_USD_PER_SECOND[self.fast_policy.gpu]
            if estimated_gpu_usd > self.fast_policy.maximum_gpu_usd + 1e-9:
                if uses_session:
                    self._warm_session = None
                else:
                    self._mark_failed(self.fast_policy)
                raise FiniteModalBudgetError("warm preview exceeded its stage cap")
            try:
                bundle = _write_warm_preview_bundle(
                    request=request,
                    result=result,
                    destination=destination,
                    remote_seconds=remote_seconds,
                    estimated_gpu_usd=estimated_gpu_usd,
                    warm_state=_deployed_warm_state(
                        result,
                        remote_seconds=remote_seconds,
                        uses_session=uses_session,
                    ),
                    reservation_id=reservation_id,
                )
            except BaseException:
                if uses_session:
                    self._warm_session = None
                else:
                    self._mark_failed(self.fast_policy)
                raise
            if session is not None:
                session.preview_scene_seconds = remote_seconds
                session.preview_scene_id = request.scene_id
            else:
                try:
                    await self._settle_warm_stage(
                        reservation_id=reservation_id,
                        experiment_id=experiment_id,
                        stage="warm-preview-scene",
                        model=FAST_MODEL,
                        revision=FAST_MODEL_REVISION,
                        seed=request.seed,
                        prompt=request.prompt,
                        artifact=bundle.artifacts["preview"],
                        generation_seconds=remote_seconds,
                        estimated_gpu_usd=estimated_gpu_usd,
                        gpu=self.fast_policy.gpu,
                    )
                except BaseException:
                    self._mark_failed(self.fast_policy)
                    raise
                self._active_reservations_gpu_usd -= self.fast_policy.worst_case_gpu_usd
                self._session_estimated_gpu_usd += estimated_gpu_usd
            return bundle

    async def prepare_fast_authorization(self, *, scene_id: str) -> None:
        """Authorize one exact scene while local planning runs; no GPU RPC starts here."""

        _validate_identifier(scene_id)
        await self._require_ready()
        async with self._operation_lock:
            if self._warm_session is not None:
                return
            if scene_id in self._prepared_fast_authorizations:
                return
            self._reserve(self.fast_policy)
            experiment_id = f"fast-authorization:{scene_id}"
            try:
                reservation_id = await self._reserve_against_current_billing(
                    experiment_id=experiment_id,
                    full_call_ceiling_usd=self.fast_policy.worst_case_gpu_usd,
                )
            except BaseException:
                self._active_reservations_gpu_usd -= self.fast_policy.worst_case_gpu_usd
                raise
            self._prepared_fast_authorizations[scene_id] = _PreparedFastAuthorization(
                scene_id=scene_id,
                reservation_id=reservation_id,
            )

    async def abandon_prepared_fast_authorization(self, *, scene_id: str) -> None:
        """Release a prepared reservation only if generation never consumed it."""

        async with self._operation_lock:
            prepared = self._prepared_fast_authorizations.get(scene_id)
            if prepared is None:
                return
            await asyncio.to_thread(
                release_modal_budget_reservation,
                plan_path=self.plan_file,
                ledger_path=self.ledger_path,
                reservation_id=prepared.reservation_id,
            )
            self._prepared_fast_authorizations.pop(scene_id, None)
            self._active_reservations_gpu_usd -= self.fast_policy.worst_case_gpu_usd

    async def abandon_warm_session(
        self,
        *,
        scene_id: str | None = None,
    ) -> None:
        """Forget a possibly billable session while retaining its ledger reservation."""

        async with self._operation_lock:
            session = self._warm_session
            if session is None:
                return
            if scene_id is not None and session.scene_id not in {None, scene_id}:
                return
            self._warm_session = None

    async def _expire_warm_session_locked(self) -> bool:
        session = self._warm_session
        if session is None or time.monotonic() < session.deadline_monotonic:
            return False
        artifact = session.scene_master or _prewarm_receipt_artifact(session.receipt_path)
        fast_seconds = (
            session.fast_prewarm_seconds
            + session.preview_scene_seconds
            + session.fast_scene_seconds
            + session.scaledown_window_seconds
        )
        motion_seconds = (
            session.motion_prewarm_seconds + session.scaledown_window_seconds
            if session.include_motion
            else 0
        )
        generation_seconds = fast_seconds + motion_seconds
        estimate = min(
            session.full_session_ceiling_usd,
            fast_seconds * GPU_USD_PER_SECOND[self.fast_policy.gpu]
            + motion_seconds * GPU_USD_PER_SECOND[self.motion_policy.gpu],
        )
        await self._settle_warm_stage(
            reservation_id=session.reservation_id,
            experiment_id=session.experiment_id,
            stage="warm-prewarm-expired",
            model=FAST_MODEL if not session.include_motion else f"{FAST_MODEL} + {MOTION_MODEL}",
            revision=(
                FAST_MODEL_REVISION
                if not session.include_motion
                else f"{FAST_MODEL_REVISION}+{MOTION_MODEL_REVISION}"
            ),
            seed=0,
            prompt="Expired explicit prewarm session; no warm RPC reuse permitted",
            artifact=artifact,
            generation_seconds=generation_seconds,
            estimated_gpu_usd=estimate,
            gpu=(self.motion_policy.gpu if session.include_motion else self.fast_policy.gpu),
        )
        self._warm_session = None
        return True

    async def generate_fast(
        self,
        request: FastSceneRequest,
        *,
        output_dir: Path,
    ) -> FiniteSceneBundle:
        await self._require_ready()
        destination = output_dir.resolve()
        manifest_path = destination / "scene.manifest.json"
        self._require_new_output(manifest_path)
        async with self._operation_lock:
            if await self._expire_warm_session_locked():
                raise FiniteModalProviderError(
                    "explicit Modal prewarm expired; prewarm again before generation"
                )
            session = self._warm_session
            uses_session = session is not None
            prepared = self._prepared_fast_authorizations.get(request.scene_id)
            if session is not None and prepared is not None:
                raise FiniteModalProviderError(
                    "scene cannot use both a prewarm session and a prepared authorization"
                )
            if session is None and prepared is not None:
                self._prepared_fast_authorizations.pop(request.scene_id, None)
            if session is not None and session.scene_id is not None:
                raise FiniteModalProviderError(
                    "the active prewarm reservation already belongs to another scene"
                )
            if (
                session is not None
                and session.preview_scene_id is not None
                and session.preview_scene_id != f"{request.scene_id}-preview"
            ):
                raise FiniteModalProviderError("preview does not match the final scene request")
            experiment_id = _fast_experiment_id(request)
            reservation_id = (
                session.reservation_id if session else prepared.reservation_id if prepared else ""
            )
            if not uses_session and prepared is None:
                self._reserve(self.fast_policy)
                try:
                    reservation_id = await self._authorize_with_current_billing(
                        self.fast_policy,
                        experiment_id=experiment_id,
                    )
                except BaseException:
                    self._active_reservations_gpu_usd -= self.fast_policy.worst_case_gpu_usd
                    raise
            started = time.perf_counter()
            try:
                result = await self.invoker.invoke(
                    "FastSceneStudio",
                    "generate",
                    {
                        "prompt": request.prompt,
                        "negative_prompt": request.negative_prompt,
                        "seed": request.seed,
                        "width": request.width,
                        "height": request.height,
                        "steps": request.steps,
                        "guidance_scale": request.guidance_scale,
                        "fidelity_label": request.fidelity_label,
                        "fidelity_object_label": request.fidelity_object_label,
                        "require_subject_object_overlap": request.require_subject_object_overlap,
                        "expected_subject_count": request.expected_subject_count,
                    },
                )
            except BaseException:
                if uses_session:
                    # Never reuse a reservation after cancellation: the remote
                    # invocation may keep running even when the local task stops.
                    self._warm_session = None
                else:
                    self._mark_failed(self.fast_policy)
                raise
            remote_seconds = time.perf_counter() - started
            self._remote_warm_deadline_monotonic = (
                time.monotonic() + self._configured_scaledown_window_seconds
            )
            estimated_gpu_usd = remote_seconds * GPU_USD_PER_SECOND[self.fast_policy.gpu]
            if estimated_gpu_usd > self.fast_policy.maximum_gpu_usd + 1e-9:
                if uses_session:
                    self._warm_session = None
                else:
                    self._mark_failed(self.fast_policy)
                raise FiniteModalBudgetError("warm fast scene exceeded its stage cap")
            try:
                bundle = _write_warm_fast_bundle(
                    request=request,
                    result=result,
                    destination=destination,
                    remote_seconds=remote_seconds,
                    estimated_gpu_usd=estimated_gpu_usd,
                    warm_state=_deployed_warm_state(
                        result,
                        remote_seconds=remote_seconds,
                        uses_session=uses_session,
                    ),
                    reservation_id=reservation_id,
                )
            except BaseException:
                if uses_session:
                    self._warm_session = None
                else:
                    self._mark_failed(self.fast_policy)
                raise
            if session is not None:
                session.fast_scene_seconds = remote_seconds
                session.scene_id = request.scene_id
                session.scene_master = bundle.master
                if not session.include_motion:
                    total_seconds = (
                        session.fast_prewarm_seconds
                        + session.preview_scene_seconds
                        + remote_seconds
                        + session.scaledown_window_seconds
                    )
                    session_estimate = min(
                        session.full_session_ceiling_usd,
                        total_seconds * GPU_USD_PER_SECOND[self.fast_policy.gpu],
                    )
                    try:
                        await self._settle_warm_stage(
                            reservation_id=session.reservation_id,
                            experiment_id=session.experiment_id,
                            stage="warm-prewarm-master",
                            model=FAST_MODEL,
                            revision=FAST_MODEL_REVISION,
                            seed=request.seed,
                            prompt=request.prompt,
                            artifact=bundle.master,
                            generation_seconds=total_seconds,
                            estimated_gpu_usd=session_estimate,
                            gpu=self.fast_policy.gpu,
                        )
                    except BaseException:
                        self._warm_session = None
                        raise
                    self._warm_session = None
            else:
                await self._settle_warm_stage(
                    reservation_id=reservation_id,
                    experiment_id=experiment_id,
                    stage="warm-fast-scene",
                    model=FAST_MODEL,
                    revision=FAST_MODEL_REVISION,
                    seed=request.seed,
                    prompt=request.prompt,
                    artifact=bundle.master,
                    generation_seconds=remote_seconds,
                    estimated_gpu_usd=estimated_gpu_usd,
                    gpu=self.fast_policy.gpu,
                )
                self._settle(bundle, stage="fast")
            return bundle

    async def upgrade_motion(
        self,
        bundle: FiniteSceneBundle | Path,
        request: MotionUpgradeRequest,
    ) -> FiniteSceneBundle:
        await self._require_ready()
        source = load_finite_scene_bundle(bundle) if isinstance(bundle, Path) else bundle
        if source.motion is not None:
            raise FiniteModalProviderError("scene already contains a motion upgrade")
        async with self._operation_lock:
            if await self._expire_warm_session_locked():
                raise FiniteModalProviderError(
                    "explicit Modal motion prewarm expired; prewarm again before motion"
                )
            session = self._warm_session
            uses_session = session is not None
            if session is not None and session.scene_id != source.scene_id:
                raise FiniteModalProviderError(
                    "motion scene does not match the active prewarm reservation"
                )
            if session is not None and not session.include_motion:
                raise FiniteModalProviderError(
                    "the active prewarm session did not reserve or load motion"
                )
            experiment_id = _motion_experiment_id(source, request)
            reservation_id = session.reservation_id if session else ""
            if not uses_session:
                self._reserve(self.motion_policy)
                try:
                    reservation_id = await self._authorize_with_current_billing(
                        self.motion_policy,
                        experiment_id=experiment_id,
                    )
                except BaseException:
                    self._active_reservations_gpu_usd -= self.motion_policy.worst_case_gpu_usd
                    raise
            started = time.perf_counter()
            try:
                result = await self.invoker.invoke(
                    "MotionUpgradeStudio",
                    "generate",
                    {
                        "master_bytes": source.master.path.read_bytes(),
                        "prompt": request.prompt,
                        "negative_prompt": request.negative_prompt,
                        "seed": request.seed,
                        "width": request.width,
                        "height": request.height,
                        "generated_frames": request.generated_frames,
                        "fps": request.fps,
                        "steps": request.steps,
                    },
                )
            except BaseException:
                if uses_session:
                    self._warm_session = None
                else:
                    self._mark_failed(self.motion_policy)
                raise
            remote_seconds = time.perf_counter() - started
            estimated_gpu_usd = remote_seconds * GPU_USD_PER_SECOND[self.motion_policy.gpu]
            if estimated_gpu_usd > self.motion_policy.maximum_gpu_usd + 1e-9:
                if uses_session:
                    self._warm_session = None
                else:
                    self._mark_failed(self.motion_policy)
                raise FiniteModalBudgetError("warm motion scene exceeded its stage cap")
            try:
                upgraded = _write_warm_motion_bundle(
                    source=source,
                    request=request,
                    result=result,
                    remote_seconds=remote_seconds,
                    estimated_gpu_usd=estimated_gpu_usd,
                    warm_state=_deployed_warm_state(
                        result,
                        remote_seconds=remote_seconds,
                        uses_session=uses_session,
                    ),
                    reservation_id=reservation_id,
                )
            except BaseException:
                if uses_session:
                    self._warm_session = None
                else:
                    self._mark_failed(self.motion_policy)
                raise
            if session is not None:
                fast_seconds = (
                    session.fast_prewarm_seconds
                    + session.preview_scene_seconds
                    + session.fast_scene_seconds
                    + session.scaledown_window_seconds
                )
                motion_seconds = (
                    session.motion_prewarm_seconds
                    + remote_seconds
                    + session.scaledown_window_seconds
                )
                total_seconds = fast_seconds + motion_seconds
                session_estimate = min(
                    session.full_session_ceiling_usd,
                    fast_seconds * GPU_USD_PER_SECOND[self.fast_policy.gpu]
                    + motion_seconds * GPU_USD_PER_SECOND[self.motion_policy.gpu],
                )
                try:
                    await self._settle_warm_stage(
                        reservation_id=session.reservation_id,
                        experiment_id=session.experiment_id,
                        stage="warm-prewarm-scene",
                        model=f"{FAST_MODEL} + {MOTION_MODEL}",
                        revision=f"{FAST_MODEL_REVISION}+{MOTION_MODEL_REVISION}",
                        seed=request.seed,
                        prompt=request.prompt,
                        artifact=upgraded.motion,
                        generation_seconds=total_seconds,
                        estimated_gpu_usd=session_estimate,
                        gpu=self.motion_policy.gpu,
                    )
                except BaseException:
                    self._warm_session = None
                    raise
                self._warm_session = None
            else:
                assert upgraded.motion is not None
                await self._settle_warm_stage(
                    reservation_id=reservation_id,
                    experiment_id=experiment_id,
                    stage="warm-motion",
                    model=MOTION_MODEL,
                    revision=MOTION_MODEL_REVISION,
                    seed=request.seed,
                    prompt=request.prompt,
                    artifact=upgraded.motion,
                    generation_seconds=remote_seconds,
                    estimated_gpu_usd=estimated_gpu_usd,
                    gpu=self.motion_policy.gpu,
                )
                self._settle(upgraded, stage="motion")
            return upgraded

    def _mark_failed(self, policy: ModalStagePolicy) -> None:
        self._active_reservations_gpu_usd -= policy.worst_case_gpu_usd
        self._failed_reservations_gpu_usd += policy.worst_case_gpu_usd

    async def _settle_warm_stage(
        self,
        *,
        reservation_id: str,
        experiment_id: str,
        stage: str,
        model: str,
        revision: str,
        seed: int,
        prompt: str,
        artifact: SceneArtifact | None,
        generation_seconds: float,
        estimated_gpu_usd: float,
        gpu: str,
    ) -> None:
        if artifact is None:
            raise FiniteModalProviderError("warm settlement requires an artifact")
        record = GenerationRecord(
            experiment_id=experiment_id,
            stage=stage,
            model=model,
            model_revision=revision,
            gpu=gpu,
            seed=seed,
            prompt=prompt,
            artifact_path=str(artifact.path),
            sha256=artifact.sha256,
            generation_seconds=generation_seconds,
            estimated_gpu_usd=estimated_gpu_usd,
            width=artifact.width,
            height=artifact.height,
            frames=artifact.frames,
            fps=artifact.fps,
        )
        await asyncio.to_thread(
            settle_modal_budget,
            plan_path=self.plan_file,
            ledger_path=self.ledger_path,
            reservation_id=reservation_id,
            record=record,
        )


@dataclass(frozen=True, slots=True)
class _ResolvedLiveScenePlan:
    pack: StoryPack
    planning_ms: float
    status: LiveScenePlanningStatus
    provenance: LiveSceneModelProvenance
    fidelity_label: str = ""
    preparation_ms: float = 0
    planning_cache_hit: bool = False


@dataclass(frozen=True, slots=True)
class _GeneratedLivePreview:
    bundle: FiniteSceneBundle
    pack: StoryPack
    artifact: LiveSceneArtifact
    cache_ms: float


class FiniteModalLiveSceneProvider:
    """Adapter from finite Modal artifacts to the progressive live-scene contract."""

    serializes_paid_jobs = True

    def __init__(
        self,
        provider: FiniteModalSceneProvider,
        *,
        cache: AssetCache,
        output_root: Path = Path("artifacts/live-scenes/generated"),
        enable_motion: bool = False,
        enable_preview: bool = True,
        planner: LiveScenePlanner | None = None,
        motion_gate: MotionTechnicalGate | None = None,
        motion_evaluator: MotionEvaluator | None = None,
        master_width: int = 1024,
        master_height: int = 576,
        master_steps: int = 2,
        master_guidance_scale: float = 4.5,
        fidelity_mode: Literal["inline", "deferred"] = "inline",
        auto_prewarm_on_submit: bool = False,
        provider_name: str = PROVIDER_NAME,
    ) -> None:
        self.provider = provider
        self.cache = cache
        self.output_root = output_root
        self.enable_motion = enable_motion
        self.enable_preview = enable_preview
        self.planner = planner
        self.motion_gate = motion_gate or MotionTechnicalGate()
        self.motion_evaluator = motion_evaluator or _evaluate_motion_technical
        if fidelity_mode not in {"inline", "deferred"}:
            raise ValueError("fidelity_mode must be 'inline' or 'deferred'")
        self.fidelity_mode = fidelity_mode
        self.auto_prewarm_on_submit = auto_prewarm_on_submit
        self.provider_name = provider_name
        # Validate the complete render profile once at construction time.
        profile = FastSceneRequest(
            scene_id="render-profile",
            prompt="validated render profile",
            negative_prompt="text",
            seed=0,
            width=master_width,
            height=master_height,
            steps=master_steps,
            guidance_scale=master_guidance_scale,
        )
        self.master_width = profile.width
        self.master_height = profile.height
        self.master_steps = profile.steps
        self.master_guidance_scale = profile.guidance_scale

    @property
    def name(self) -> str:
        return self.provider_name

    async def aclose(self) -> None:
        close = getattr(self.provider, "aclose", None)
        if callable(close):
            await close()

    async def generate(
        self,
        request: LiveSceneCreateRequest,
        *,
        job_id: str,
    ) -> AsyncIterator[LiveSceneUpdate]:
        try:
            async for update in self._generate_unprotected(request, job_id=job_id):
                yield update
        finally:
            if isinstance(self.provider, WarmModalSceneProvider):
                # Async generators are commonly closed while suspended at a
                # yielded MASTER_READY update. Always abandon any assigned warm
                # session; a completed motion settlement makes this a no-op.
                cleanup = asyncio.create_task(self.provider.abandon_warm_session(scene_id=job_id))
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    await cleanup
                    raise
                authorization_cleanup = asyncio.create_task(
                    self.provider.abandon_prepared_fast_authorization(scene_id=job_id)
                )
                try:
                    await asyncio.shield(authorization_cleanup)
                except asyncio.CancelledError:
                    await authorization_cleanup
                    raise
                preview_authorization_cleanup = asyncio.create_task(
                    self.provider.abandon_prepared_fast_authorization(
                        scene_id=f"{job_id}-preview"
                    )
                )
                try:
                    await asyncio.shield(preview_authorization_cleanup)
                except asyncio.CancelledError:
                    await preview_authorization_cleanup
                    raise

    async def _generate_unprotected(
        self,
        request: LiveSceneCreateRequest,
        *,
        job_id: str,
    ) -> AsyncIterator[LiveSceneUpdate]:
        seed = live_scene_request_seed(request)
        deterministic_compiler = DETERMINISTIC_LIVE_SCENE_COMPILER_MODEL
        draft = build_live_scene_story_pack(
            request,
            job_id=job_id,
            seed=seed,
            assets=[],
            compiler_model=deterministic_compiler,
        )
        yield LiveSceneUpdate(
            stage=LiveSceneStage.DRAFT_READY,
            progress=0.35,
            story_pack=draft,
        )

        preview: _GeneratedLivePreview | None = None
        preview_task: asyncio.Task[_GeneratedLivePreview] | None = None
        if await self._should_generate_preview(request):
            preview_task = asyncio.create_task(
                self._generate_preview(request, job_id=job_id, seed=seed),
                name=f"bookforge-live-preview-{job_id}",
            )
        resolved_task = asyncio.create_task(
            self._resolve_plan_while_preparing_renderer(
                request,
                job_id=job_id,
                seed=seed,
                draft=draft,
                prepare_renderer=preview_task is None,
            ),
            name=f"bookforge-live-plan-and-renderer-{job_id}",
        )
        try:
            if preview_task is not None:
                done, _ = await asyncio.wait(
                    {preview_task, resolved_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if preview_task in done:
                    try:
                        preview = preview_task.result()
                    except Exception:
                        # The preview is a latency optimization, never a dependency
                        # of the authoritative Gemma/SANA scene. Fail open to the
                        # final render while preserving cancellation semantics.
                        preview = None
                    if preview is not None:
                        yield LiveSceneUpdate(
                            stage=LiveSceneStage.PREVIEW_READY,
                            progress=0.58,
                            story_pack=preview.pack,
                            artifacts=[preview.artifact],
                            metrics=_preview_scene_metrics(preview),
                        )
                    resolved = await resolved_task
                else:
                    # A provisional preview has no value once the authoritative
                    # plan is ready. Never put it on the final-render critical path.
                    resolved = resolved_task.result()
                    preview_task.cancel()
                    preview_drain = asyncio.ensure_future(
                        asyncio.gather(preview_task, return_exceptions=True)
                    )
                    try:
                        await asyncio.shield(preview_drain)
                    except asyncio.CancelledError:
                        await preview_drain
                        raise
            else:
                resolved = await resolved_task
        finally:
            pending = [
                task
                for task in (preview_task, resolved_task)
                if task is not None and not task.done()
            ]
            for task in pending:
                task.cancel()
            if pending:
                drain = asyncio.ensure_future(asyncio.gather(*pending, return_exceptions=True))
                try:
                    await asyncio.shield(drain)
                except asyncio.CancelledError:
                    await drain
                    raise
        page = resolved.pack.pages[0]
        if page.scene_spec is None:
            raise FiniteModalProviderError("live-scene draft has no SceneSpec")
        background_layer_id = _background_layer_id(resolved.pack)
        fidelity_object_label = _fidelity_action_object(page.layers)
        inline_fidelity = self.fidelity_mode == "inline"
        fast_request = FastSceneRequest(
            scene_id=job_id,
            prompt=_bounded_prompt(
                page.scene_spec.master_prompt,
                "Full-bleed luminous storybook projection, strong foreground/background depth, "
                "clean silhouettes, no border, no interface.",
            ),
            negative_prompt=(
                f"{page.scene_spec.negative_prompt}, words, letters, captions, interface, border, "
                "split screen, collage, photorealism, duplicate character"
            ),
            seed=seed,
            width=self.master_width,
            height=self.master_height,
            steps=self.master_steps,
            guidance_scale=self.master_guidance_scale,
            fidelity_label=resolved.fidelity_label if inline_fidelity else "",
            fidelity_object_label=fidelity_object_label if inline_fidelity else "",
            require_subject_object_overlap=(
                _fidelity_requires_overlap(page.layers) if inline_fidelity else False
            ),
            expected_subject_count=1,
        )
        try:
            fast_bundle = await self.provider.generate_fast(
                fast_request,
                output_dir=self.output_root / job_id,
            )
        except FiniteModalUnavailableError as error:
            raise LiveSceneProviderUnavailableError(str(error)) from error
        try:
            selected_seed = int(
                fast_bundle.manifest["stages"]["fast"].get("selected_seed", seed)
            )
            promotion_started = time.perf_counter()
            master_result, depth_result = await asyncio.gather(
                self._promote_artifact(
                    bundle=fast_bundle,
                    role="master",
                    job_id=job_id,
                    seed=selected_seed,
                    prompt=fast_request.prompt,
                    layer_id=background_layer_id,
                ),
                self._promote_artifact(
                    bundle=fast_bundle,
                    role="depth",
                    job_id=job_id,
                    seed=selected_seed,
                    prompt=f"Depth estimate for {job_id}-master",
                    layer_id=background_layer_id,
                ),
            )
            master_record, master_artifact, _master_cache_ms = master_result
            depth_record, depth_artifact, _depth_cache_ms = depth_result
            fast_cache_ms = (time.perf_counter() - promotion_started) * 1_000
            assets = [master_record, depth_record]
            artifacts = [master_artifact, depth_artifact]
            master_pack = _with_live_scene_assets(resolved.pack, assets)
            yield LiveSceneUpdate(
                stage=LiveSceneStage.MASTER_READY,
                progress=1 if not self.enable_motion else 0.7,
                complete=not self.enable_motion,
                story_pack=master_pack,
                artifacts=artifacts,
                metrics=_live_scene_metrics(
                    fast_bundle,
                    cache_ms=fast_cache_ms + (preview.cache_ms if preview else 0),
                    planning=resolved,
                    preview=preview,
                ),
            )
        except BaseException:
            if isinstance(self.provider, WarmModalSceneProvider):
                cleanup = asyncio.create_task(self.provider.abandon_warm_session(scene_id=job_id))
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    await cleanup
                    raise
            raise
        if not self.enable_motion:
            return

        motion_seed = (seed + 1) % (2**32)
        motion_request = MotionUpgradeRequest(
            prompt=_bounded_prompt(
                page.scene_spec.master_prompt,
                "Locked storybook camera. Preserve exact identity, composition, objects, colors, "
                "and silhouettes. Add only gentle character "
                "breathing, blinking, drifting paper motes, and warm breathing light. No new "
                "content, transition, morph, or readable text.",
            ),
            seed=motion_seed,
        )
        upgraded = await self.provider.upgrade_motion(fast_bundle, motion_request)
        if upgraded.motion is None:
            raise FiniteModalProviderError("motion provider returned no motion artifact")
        evidence = await asyncio.to_thread(self.motion_evaluator, upgraded.motion.path)
        reasons = self.motion_gate.rejection_reasons(evidence)
        if evidence.sha256 != upgraded.motion.sha256:
            reasons.insert(0, "technical evaluation checksum does not match the generated loop")
        await asyncio.to_thread(
            _write_motion_gate_report,
            upgraded.manifest_path.parent / "motion.technical.json",
            evidence,
            reasons,
        )
        if reasons:
            raise FiniteModalProviderError(
                "motion technical promotion gate rejected the loop: " + "; ".join(reasons)
            )
        motion_record, motion_artifact, motion_cache_ms = await self._promote_artifact(
            bundle=upgraded,
            role="motion",
            job_id=job_id,
            seed=motion_seed,
            prompt=motion_request.prompt,
            layer_id=background_layer_id,
        )
        motion_pack = _with_live_scene_assets(resolved.pack, [*assets, motion_record])
        yield LiveSceneUpdate(
            stage=LiveSceneStage.MOTION_READY,
            progress=1,
            complete=True,
            story_pack=motion_pack,
            artifacts=[*artifacts, motion_artifact],
            metrics=_live_scene_metrics(
                upgraded,
                cache_ms=(preview.cache_ms if preview else 0) + fast_cache_ms + motion_cache_ms,
                include_motion=True,
                planning=resolved,
                preview=preview,
            ),
        )

    async def _should_generate_preview(self, request: LiveSceneCreateRequest) -> bool:
        if (
            not self.enable_preview
            or not isinstance(self.provider, WarmModalSceneProvider)
            or self.planner is None
        ):
            return False
        if not await self.provider.is_renderer_likely_warm():
            return False
        cache_probe = getattr(self.planner, "has_cached_plan", None)
        if not callable(cache_probe):
            return False
        try:
            return not await cache_probe(text=request.text)
        except Exception:
            return False

    async def _generate_preview(
        self,
        request: LiveSceneCreateRequest,
        *,
        job_id: str,
        seed: int,
    ) -> _GeneratedLivePreview:
        if not isinstance(self.provider, WarmModalSceneProvider):
            raise FiniteModalProviderError("live preview requires the warm Modal provider")
        preview_id = f"{job_id}-preview"
        await self.provider.prepare_fast_authorization(scene_id=preview_id)
        cloud_safe_pack = build_live_scene_story_pack(
            request,
            job_id=job_id,
            seed=seed,
            assets=[],
            compiler_model=DETERMINISTIC_LIVE_SCENE_COMPILER_MODEL,
            cloud_safe_prompts=True,
        )
        page = cloud_safe_pack.pages[0]
        if page.scene_spec is None:
            raise FiniteModalProviderError("live preview has no privacy-safe SceneSpec")
        background_layer_id = _background_layer_id(cloud_safe_pack)
        preview_request = FastPreviewRequest(
            scene_id=preview_id,
            prompt=_bounded_prompt(
                page.scene_spec.master_prompt,
                "Provisional visual sketch with "
                f"{_safe_preview_subject(request.text)}. One cohesive full-bleed 16:9 scene, "
                "crisp complete silhouette, tactile paper depth, no text, no border.",
            ),
            seed=seed,
        )
        bundle = await self.provider.generate_preview(
            preview_request,
            output_dir=self.output_root / job_id / "preview",
        )
        record, artifact, cache_ms = await self._promote_artifact(
            bundle=bundle,
            role="preview",
            job_id=job_id,
            seed=seed,
            prompt=preview_request.prompt,
            layer_id=background_layer_id,
        )
        return _GeneratedLivePreview(
            bundle=bundle,
            pack=_with_live_scene_assets(cloud_safe_pack, [record]),
            artifact=artifact,
            cache_ms=cache_ms,
        )

    async def _resolve_plan_while_preparing_renderer(
        self,
        request: LiveSceneCreateRequest,
        *,
        job_id: str,
        seed: int,
        draft: StoryPack,
        prepare_renderer: bool = True,
    ) -> _ResolvedLiveScenePlan:
        prepare_task: asyncio.Task[float] | None = None
        if prepare_renderer and (
            isinstance(self.provider, WarmModalSceneProvider)
            or (self.auto_prewarm_on_submit and self._supports_auto_prewarm())
        ):
            prepare_task = asyncio.create_task(
                self._prepare_warm_renderer(job_id=job_id),
                name=f"bookforge-renderer-preparation-{job_id}",
            )
        elif prepare_renderer and callable(getattr(self.provider, "probe", None)):
            # Readiness probes are explicitly non-billable. Overlap credential
            # refresh and DNS/TLS/HTTP setup with private local planning so a
            # fresh process does not put that latency on the render critical path.
            prepare_task = asyncio.create_task(
                self._prepare_safe_provider_route(),
                name=f"bookforge-provider-route-preparation-{job_id}",
            )
        try:
            resolved = await self._resolve_plan(
                request,
                job_id=job_id,
                seed=seed,
                draft=draft,
            )
            if prepare_task is not None:
                resolved = replace(
                    resolved,
                    preparation_ms=await prepare_task,
                )
            return resolved
        finally:
            if prepare_task is not None and not prepare_task.done():
                prepare_task.cancel()
                drain = asyncio.ensure_future(asyncio.gather(prepare_task, return_exceptions=True))
                try:
                    await asyncio.shield(drain)
                except asyncio.CancelledError:
                    await drain
                    raise

    def _supports_auto_prewarm(self) -> bool:
        return all(
            callable(getattr(self.provider, method_name, None))
            for method_name in ("prewarm", "is_prewarmed", "is_renderer_likely_warm")
        )

    async def _prepare_safe_provider_route(self) -> float:
        probe = getattr(self.provider, "probe", None)
        if not callable(probe):
            return 0
        started = time.perf_counter()
        try:
            result = await probe()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise LiveSceneProviderUnavailableError(
                f"Cloud scene route readiness failed before billing: {error}"
            ) from error
        if (
            not isinstance(result, tuple)
            or len(result) != 2
            or not isinstance(result[0], bool)
        ):
            raise LiveSceneProviderUnavailableError(
                "Cloud scene route returned invalid readiness evidence before billing"
            )
        ready, detail = result
        if not ready:
            raise LiveSceneProviderUnavailableError(
                f"Cloud scene route is unavailable before billing: {str(detail)[:300]}"
            )
        return (time.perf_counter() - started) * 1_000

    async def _prepare_warm_renderer(self, *, job_id: str) -> float:
        is_prewarmed = getattr(self.provider, "is_prewarmed", None)
        if not callable(is_prewarmed):
            return 0
        if await is_prewarmed():
            return 0
        started = time.perf_counter()
        is_likely_warm = getattr(self.provider, "is_renderer_likely_warm", None)
        prewarm = getattr(self.provider, "prewarm", None)
        if (
            self.auto_prewarm_on_submit
            and callable(is_likely_warm)
            and callable(prewarm)
            and not await is_likely_warm()
        ):
            await prewarm(prewarm_id=f"auto-{job_id}", include_motion=self.enable_motion)
        elif isinstance(self.provider, WarmModalSceneProvider):
            await self.provider.prepare_fast_authorization(scene_id=job_id)
        return (time.perf_counter() - started) * 1_000

    async def _resolve_plan(
        self,
        request: LiveSceneCreateRequest,
        *,
        job_id: str,
        seed: int,
        draft: StoryPack,
    ) -> _ResolvedLiveScenePlan:
        if self.planner is None:
            cloud_safe_pack = build_live_scene_story_pack(
                request,
                job_id=job_id,
                seed=seed,
                assets=[],
                compiler_model=draft.compiler_model,
                cloud_safe_prompts=True,
            )
            return _ResolvedLiveScenePlan(
                pack=cloud_safe_pack,
                planning_ms=0,
                status=LiveScenePlanningStatus.DETERMINISTIC,
                provenance=LiveSceneModelProvenance(
                    role="scene_plan",
                    model=draft.compiler_model,
                    revision="v1",
                ),
                fidelity_label="",
            )

        try:
            result = await self.planner.plan(
                text=request.text,
                visual_style=request.visual_style,
                seed=seed,
            )
            planned_page = result.plan.to_page(
                source_text=request.text,
                visual_style=request.visual_style,
                seed=seed,
            )
        except LiveScenePlannerError as error:
            # A generic privacy-safe prompt is useful for deterministic fixtures,
            # but it is not a faithful substitute for a requested live scene. In
            # model-planner mode, stop before the paid renderer rather than turn a
            # local timeout into polished, semantically unrelated artwork.
            raise LiveSceneProviderUnavailableError(
                "Local scene planning failed, so no cloud render was started: "
                f"{error}. Warm the edge planner and retry."
            ) from error

        pack = build_planned_live_scene_story_pack(
            request,
            job_id=job_id,
            page=planned_page,
            assets=[],
            compiler_model=result.metrics.model,
        )
        return _ResolvedLiveScenePlan(
            pack=pack,
            planning_ms=result.wall_ms,
            status=LiveScenePlanningStatus.MODEL,
            provenance=LiveSceneModelProvenance(
                role="scene_plan",
                model=result.metrics.model,
                revision=result.model_revision,
            ),
            fidelity_label=result.plan.focus_label,
            planning_cache_hit=result.cache_hit,
        )

    async def _promote_artifact(
        self,
        *,
        bundle: FiniteSceneBundle,
        role: str,
        job_id: str,
        seed: int,
        prompt: str,
        layer_id: str,
    ) -> tuple[AssetRecord, LiveSceneArtifact, float]:
        promotion_started = time.perf_counter()
        source = bundle.artifacts[role]
        settings = {
            "preview": (
                AssetKind.IMAGE,
                AssetRole.PREVIEW,
                LiveSceneArtifactKind.PREVIEW,
                FAST_MODEL,
                FAST_MODEL_REVISION,
            ),
            "master": (
                AssetKind.IMAGE,
                AssetRole.MASTER,
                LiveSceneArtifactKind.MASTER,
                FAST_MODEL,
                FAST_MODEL_REVISION,
            ),
            "depth": (
                AssetKind.DEPTH_MAP,
                AssetRole.DEPTH,
                LiveSceneArtifactKind.DEPTH,
                DEPTH_MODEL,
                DEPTH_MODEL_REVISION,
            ),
            "motion": (
                AssetKind.VIDEO_LOOP,
                AssetRole.MOTION,
                LiveSceneArtifactKind.MOTION,
                MOTION_MODEL,
                MOTION_MODEL_REVISION,
            ),
        }
        asset_kind, asset_role, live_kind, model, revision = settings[role]
        suffix = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "video/mp4": ".mp4",
        }.get(source.mime_type)
        if suffix is None:
            raise FiniteModalProviderError(
                f"unsupported {role} artifact media type: {source.mime_type}"
            )
        artifact_id = f"{job_id}-{role}"
        content = await asyncio.to_thread(source.path.read_bytes)
        digest, uri = await self.cache.store_generated(
            asset_id=artifact_id,
            kind=asset_kind,
            content=content,
            suffix=suffix,
        )
        if digest != source.sha256:
            raise FiniteModalProviderError(f"cached {role} checksum changed during promotion")
        stage_name = role if role in {"preview", "motion"} else "fast"
        stage = bundle.manifest["stages"][stage_name]
        model, revision = _artifact_model_provenance(
            stage,
            role=role,
            default_model=model,
            default_revision=revision,
        )
        if role == "master":
            generation_ms = float(stage.get("image_seconds", 0)) * 1000
        elif role == "depth":
            generation_ms = float(stage.get("depth_seconds", 0)) * 1000
        else:
            generation_ms = float(stage.get("inference_seconds", 0)) * 1000
        artifact_provider = str(bundle.manifest.get("provider", self.provider_name))
        provider_label = f"{artifact_provider}:{model}@{revision}"
        record = AssetRecord(
            asset_id=artifact_id,
            page_id="page-01",
            layer_id=layer_id,
            kind=asset_kind,
            role=asset_role,
            provider=provider_label,
            prompt=prompt,
            seed=seed,
            width=source.width,
            height=source.height,
            duration_ms=source.duration_ms,
            checksum_sha256=digest,
            local_uri=uri,
            state=AssetState.READY,
            generation_ms=generation_ms,
        )
        artifact = LiveSceneArtifact(
            artifact_id=artifact_id,
            kind=live_kind,
            uri=uri,
            checksum_sha256=digest,
            media_type=source.mime_type,
            provider=artifact_provider,
            model=f"{model}@{revision}",
            seed=seed,
            width=source.width,
            height=source.height,
            duration_ms=source.duration_ms,
        )
        return record, artifact, (time.perf_counter() - promotion_started) * 1000


def _live_scene_metrics(
    bundle: FiniteSceneBundle,
    *,
    cache_ms: float,
    include_motion: bool = False,
    planning: _ResolvedLiveScenePlan,
    preview: _GeneratedLivePreview | None = None,
) -> LiveSceneMetrics:
    stage_names = ["fast", *(("motion",) if include_motion else ())]
    stages = [bundle.manifest["stages"][stage_name] for stage_name in stage_names]
    if preview is not None:
        stages.insert(0, preview.bundle.manifest["stages"]["preview"])
    provider_seconds = 0.0
    inference_seconds = 0.0
    overhead_seconds = 0.0
    packaging_seconds = 0.0
    warm_states: list[LiveSceneWarmState] = []
    for stage in stages:
        fallback_inference = float(stage.get("image_seconds", 0)) + float(
            stage.get("depth_seconds", 0)
        )
        inference = float(stage.get("inference_seconds", fallback_inference))
        remote = float(stage.get("remote_seconds", inference))
        overhead = float(stage.get("provider_overhead_seconds", max(0.0, remote - inference)))
        provider_seconds += remote
        inference_seconds += inference
        overhead_seconds += overhead
        packaging_seconds += float(stage.get("packaging_seconds", 0))
        raw_warm_state = str(stage.get("warm_state", "unknown")).lower()
        if raw_warm_state in {"warm", "prewarmed"}:
            warm_states.append(LiveSceneWarmState.WARM)
        elif raw_warm_state == "cold":
            warm_states.append(LiveSceneWarmState.COLD)
        else:
            warm_states.append(LiveSceneWarmState.UNKNOWN)
    if warm_states and all(state is LiveSceneWarmState.WARM for state in warm_states):
        warm_state = LiveSceneWarmState.WARM
    elif any(state is LiveSceneWarmState.COLD for state in warm_states):
        warm_state = LiveSceneWarmState.COLD
    else:
        warm_state = LiveSceneWarmState.UNKNOWN
    fast_stage = bundle.manifest["stages"]["fast"]
    master_model, master_revision = _artifact_model_provenance(
        fast_stage,
        role="master",
        default_model=FAST_MODEL,
        default_revision=FAST_MODEL_REVISION,
    )
    depth_model, depth_revision = _artifact_model_provenance(
        fast_stage,
        role="depth",
        default_model=DEPTH_MODEL,
        default_revision=DEPTH_MODEL_REVISION,
    )
    models = [
        planning.provenance,
        *(
            [
                LiveSceneModelProvenance(
                    role="preview",
                    model=FAST_MODEL,
                    revision=FAST_MODEL_REVISION,
                )
            ]
            if preview is not None
            else []
        ),
        LiveSceneModelProvenance(
            role="master",
            model=master_model,
            revision=master_revision,
        ),
        LiveSceneModelProvenance(
            role="depth",
            model=depth_model,
            revision=depth_revision,
        ),
    ]
    if any(stage.get("quality_label") for stage in stages):
        models.append(
            LiveSceneModelProvenance(
                role="visual_fidelity",
                model=FIDELITY_MODEL,
                revision=FIDELITY_MODEL_REVISION,
            )
        )
    if include_motion:
        models.append(
            LiveSceneModelProvenance(
                role="motion",
                model=MOTION_MODEL,
                revision=MOTION_MODEL_REVISION,
            )
        )
    estimated_gpu_usd = sum(float(stage["estimated_gpu_usd"]) for stage in stages)
    cache_ms = max(0.0, cache_ms)
    provider_ms = max(0.0, provider_seconds * 1000)
    planning_ms = max(0.0, planning.planning_ms)
    preparation_ms = max(0.0, planning.preparation_ms)
    return LiveSceneMetrics(
        elapsed_ms=max(planning_ms, preparation_ms) + provider_ms + cache_ms,
        provider_ms=provider_ms,
        inference_ms=max(0.0, inference_seconds * 1000),
        cache_ms=cache_ms,
        overhead_ms=max(0.0, overhead_seconds * 1000),
        packaging_ms=max(0.0, packaging_seconds * 1000),
        planning_ms=planning_ms,
        preparation_ms=preparation_ms,
        planning_status=planning.status,
        planning_cache_hit=planning.planning_cache_hit,
        warm_state=warm_state,
        gpu=str(stages[0].get("gpu", FAST_GPU)),
        estimated_gpu_usd=max(0.0, estimated_gpu_usd),
        cost_source=LiveSceneCostSource.PROVIDER_MANIFEST,
        models=models,
    )


def _artifact_model_provenance(
    stage: Mapping[str, Any],
    *,
    role: str,
    default_model: str,
    default_revision: str,
) -> tuple[str, str]:
    if role in {"master", "preview", "motion"}:
        model = stage.get("model")
        revision = stage.get("model_revision")
        if isinstance(model, str) and model and isinstance(revision, str) and revision:
            return model[:200], revision[:200]
        return default_model, default_revision
    additional = stage.get("additional_models")
    if isinstance(additional, list):
        for candidate in additional:
            if not isinstance(candidate, dict):
                continue
            candidate_role = candidate.get("role")
            if candidate_role not in {None, role}:
                continue
            model = candidate.get("model")
            revision = candidate.get("model_revision")
            if isinstance(model, str) and model and isinstance(revision, str) and revision:
                return model[:200], revision[:200]
    return default_model, default_revision


def _preview_scene_metrics(preview: _GeneratedLivePreview) -> LiveSceneMetrics:
    stage = preview.bundle.manifest["stages"]["preview"]
    inference_seconds = float(stage.get("inference_seconds", stage.get("image_seconds", 0)))
    remote_seconds = float(stage.get("remote_seconds", inference_seconds))
    overhead_seconds = float(
        stage.get("provider_overhead_seconds", max(0.0, remote_seconds - inference_seconds))
    )
    raw_warm_state = str(stage.get("warm_state", "unknown")).lower()
    warm_state = (
        LiveSceneWarmState.WARM
        if raw_warm_state in {"warm", "prewarmed"}
        else LiveSceneWarmState.COLD
        if raw_warm_state == "cold"
        else LiveSceneWarmState.UNKNOWN
    )
    return LiveSceneMetrics(
        elapsed_ms=remote_seconds * 1_000 + preview.cache_ms,
        provider_ms=remote_seconds * 1_000,
        inference_ms=inference_seconds * 1_000,
        cache_ms=preview.cache_ms,
        overhead_ms=overhead_seconds * 1_000,
        packaging_ms=float(stage.get("packaging_seconds", 0)) * 1_000,
        warm_state=warm_state,
        gpu=str(stage.get("gpu", FAST_GPU)),
        estimated_gpu_usd=float(stage.get("estimated_gpu_usd", 0)),
        cost_source=LiveSceneCostSource.PROVIDER_MANIFEST,
        models=[
            LiveSceneModelProvenance(
                role="preview",
                model=FAST_MODEL,
                revision=FAST_MODEL_REVISION,
            )
        ],
    )


def _safe_preview_subject(text: str) -> str:
    """Map private text to one fixed, non-verbatim visual category."""

    lowered = text.casefold()
    categories = (
        (("fox",), "one elegant storybook fox"),
        (("whale",), "one graceful storybook whale"),
        (("turtle",), "one gentle storybook turtle"),
        (("moth", "butterfly"), "one luminous winged creature"),
        (("rabbit", "bunny"), "one curious storybook rabbit"),
        (("bear",), "one gentle storybook bear"),
        (("deer",), "one elegant storybook deer"),
        (("dragon",), "one friendly storybook dragon"),
        (("robot",), "one friendly storybook robot"),
        (("astronaut",), "one complete storybook astronaut"),
        (("bird",), "one graceful storybook bird"),
        (("fish",), "one luminous storybook fish"),
        (("child", "girl", "boy", "reader"), "one complete child silhouette"),
    )
    for tokens, description in categories:
        if any(re.search(rf"\b{re.escape(token)}\b", lowered) for token in tokens):
            return description
    return "one clear central storybook subject"


def _fidelity_focus_prompt(layers: list[Any]) -> str:
    return next(
        (str(layer.prompt) for layer in layers if layer.layer_id == "scene-focus"),
        "",
    )


def _fidelity_action_object(layers: list[Any]) -> str:
    prompt = _fidelity_focus_prompt(layers)
    match = re.search(r"\bshown\s+[a-z'-]+ing\s+(.+)$", prompt, flags=re.IGNORECASE)
    if match is None:
        return ""
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", match.group(1))
    while words and words[0].casefold() in {"a", "an", "one", "the"}:
        words.pop(0)
    if not words:
        return ""
    if len(words) >= 2 and words[-1].casefold() == "boat":
        material = words[-2].casefold()
        if material in {"acorn", "coconut", "walnut"}:
            return f"{material} shell boat"
    return " ".join(words[-2:])


def _fidelity_requires_overlap(layers: list[Any]) -> bool:
    prompt = _fidelity_focus_prompt(layers).casefold()
    return any(
        marker in prompt
        for marker in ("shown paddling ", "shown riding ", "shown rowing ", "shown steering ")
    )


def _background_layer_id(pack: StoryPack) -> str:
    backgrounds = [
        layer.layer_id for page in pack.pages for layer in page.layers if layer.kind == "background"
    ]
    if len(backgrounds) != 1:
        raise FiniteModalProviderError("live-scene plan must contain exactly one background")
    return backgrounds[0]


def _with_live_scene_assets(pack: StoryPack, assets: list[AssetRecord]) -> StoryPack:
    return StoryPack.model_validate(pack.model_copy(update={"assets": assets}).model_dump())


def _evaluate_motion_technical(path: Path) -> MotionTechnicalEvidence:
    evaluation: MediaEvaluation = evaluate_media(path)
    if evaluation.motion is None:
        raise FiniteModalProviderError("motion technical evaluator returned no motion metrics")
    return MotionTechnicalEvidence(
        sha256=evaluation.sha256,
        duration_seconds=evaluation.motion.duration_seconds,
        fps=evaluation.motion.fps,
        endpoint_ssim=evaluation.motion.endpoint_ssim,
        motion_stability=evaluation.motion.motion_stability,
        projection_legibility=evaluation.projection.projection_legibility,
    )


def _write_motion_gate_report(
    path: Path,
    evidence: MotionTechnicalEvidence,
    rejection_reasons: list[str],
) -> None:
    _atomic_write_json(
        path,
        {
            "schema_version": "1.0",
            "accepted": not rejection_reasons,
            "technical": asdict(evidence),
            "rejection_reasons": rejection_reasons,
            "semantic_identity_review": "not_performed",
            "semantic_identity_review_required_for_showcase": True,
            "promotion_scope": "experimental-technical-only",
        },
    )


def _fast_experiment_id(request: FastSceneRequest) -> str:
    identity = hashlib.sha256(
        json.dumps(
            {
                "scene_id": request.scene_id,
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "seed": request.seed,
                "width": request.width,
                "height": request.height,
                "steps": request.steps,
                "guidance_scale": request.guidance_scale,
                "fidelity_label": request.fidelity_label,
                "fidelity_object_label": request.fidelity_object_label,
                "require_subject_object_overlap": request.require_subject_object_overlap,
                "expected_subject_count": request.expected_subject_count,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    return f"fast-scene:{request.scene_id}:{identity}"


def _motion_experiment_id(
    bundle: FiniteSceneBundle,
    request: MotionUpgradeRequest,
) -> str:
    identity = hashlib.sha256(
        json.dumps(
            {
                "scene_id": bundle.scene_id,
                "master_sha256": bundle.master.sha256,
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "seed": request.seed,
                "width": request.width,
                "height": request.height,
                "generated_frames": request.generated_frames,
                "fps": request.fps,
                "steps": request.steps,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    return f"live-motion:{bundle.scene_id}:{identity}"


def _bounded_prompt(subject: str, direction: str) -> str:
    separator = ". "
    available = 4_000 - len(separator) - len(direction)
    if available < 1:
        raise ValueError("generation direction is too long")
    bounded_subject = subject[:available].rstrip(" \t\r\n.,;:!?")
    return f"{bounded_subject}{separator}{direction}"


def _deployed_warm_state(
    result: Mapping[str, Any],
    *,
    remote_seconds: float,
    uses_session: bool,
) -> str:
    if uses_session:
        return "prewarmed"
    try:
        container_age_seconds = float(result["container_age_seconds"])
    except (KeyError, TypeError, ValueError):
        return "unknown"
    # loaded_at is recorded after the model is resident. A container older than
    # this RPC was necessarily reused; a cold RPC includes startup outside age.
    return "warm" if container_age_seconds >= remote_seconds + 0.25 else "cold"


def _write_warm_preview_bundle(
    *,
    request: FastPreviewRequest,
    result: Mapping[str, Any],
    destination: Path,
    remote_seconds: float,
    estimated_gpu_usd: float,
    warm_state: str,
    reservation_id: str,
) -> FiniteSceneBundle:
    preview_path = destination / "preview.jpg"
    manifest_path = destination / "preview.manifest.json"
    if preview_path.exists() or manifest_path.exists():
        raise FiniteModalProviderError(f"preview output already exists: {destination}")
    try:
        preview = bytes(result["master"])
        image_seconds = float(result["image_seconds"])
        packaging_seconds = float(result["packaging_seconds"])
        master_jpeg_quality = int(result["master_jpeg_quality"])
    except (KeyError, TypeError, ValueError) as error:
        raise FiniteModalProviderError("warm preview class returned invalid output") from error
    if result.get("master_media_type") != "image/jpeg":
        raise FiniteModalProviderError("warm preview class returned an unsupported format")
    if result.get("negative_prompt_supported") is not False:
        raise FiniteModalProviderError("warm preview class returned ambiguous prompt provenance")
    if result.get("gpu") != WARM_FAST_GPU:
        raise FiniteModalProviderError("warm preview class ran on an unexpected GPU")
    if master_jpeg_quality != 95:
        raise FiniteModalProviderError("warm preview class returned unexpected JPEG quality")
    if _jpeg_dimensions(preview) != (request.width, request.height):
        raise FiniteModalProviderError("warm preview dimensions do not match the request")
    _atomic_write(preview_path, preview)
    now = datetime.now(UTC).isoformat()
    payload = {
        "schema_version": "1.0",
        "provider": PROVIDER_NAME,
        "scene_id": request.scene_id,
        "created_at": now,
        "updated_at": now,
        "request": {
            "prompt": request.prompt,
            "seed": request.seed,
            "width": request.width,
            "height": request.height,
        },
        "policy": {
            "finite_calls_only": True,
            "persistent_endpoint": False,
            "provider_mode": "authenticated-deployed-class",
            "provisional_preview_only": True,
            "source_text_allowed": False,
            "billing_reservation_id": reservation_id,
        },
        "stages": {
            "preview": {
                "model": FAST_MODEL,
                "model_revision": FAST_MODEL_REVISION,
                "gpu": WARM_FAST_GPU,
                "finite_call": True,
                "hard_timeout_seconds": FAST_STAGE_POLICY.remote_timeout_seconds,
                "remote_seconds": remote_seconds,
                "inference_seconds": image_seconds,
                "provider_overhead_seconds": max(0.0, remote_seconds - image_seconds),
                "image_seconds": image_seconds,
                "packaging_seconds": packaging_seconds,
                "master_jpeg_quality": master_jpeg_quality,
                "negative_prompt_supported": False,
                "model_load_seconds": float(result.get("model_load_seconds", 0)),
                "container_age_seconds": float(result.get("container_age_seconds", 0)),
                "warm_state": warm_state,
                "estimated_gpu_usd": estimated_gpu_usd,
                "steps": 1,
                "guidance_scale": request.guidance_scale,
            }
        },
        "artifacts": {
            "preview": _local_artifact_payload(
                preview_path,
                root=destination,
                mime_type="image/jpeg",
                width=request.width,
                height=request.height,
            )
        },
    }
    _atomic_write_json(manifest_path, payload)
    artifact_payload = payload["artifacts"]["preview"]
    return FiniteSceneBundle(
        manifest_path=manifest_path,
        scene_id=request.scene_id,
        artifacts={
            "preview": SceneArtifact(
                role="preview",
                path=preview_path,
                sha256=str(artifact_payload["sha256"]),
                mime_type="image/jpeg",
                width=request.width,
                height=request.height,
            )
        },
        manifest=payload,
    )


def _write_warm_fast_bundle(
    *,
    request: FastSceneRequest,
    result: Mapping[str, Any],
    destination: Path,
    remote_seconds: float,
    estimated_gpu_usd: float,
    warm_state: str,
    reservation_id: str,
) -> FiniteSceneBundle:
    master_path = destination / "master.jpg"
    depth_path = destination / "depth.jpg"
    manifest_path = destination / "scene.manifest.json"
    if any(path.exists() for path in (master_path, depth_path, manifest_path)):
        raise FiniteModalProviderError(f"scene output already exists: {destination}")
    try:
        master = bytes(result["master"])
        depth = bytes(result["depth"])
        image_seconds = float(result["image_seconds"])
        depth_seconds = float(result["depth_seconds"])
        quality_seconds = float(result.get("quality_seconds", 0))
        quality_attempts = int(result.get("quality_attempts", 0))
        selected_seed = int(result.get("selected_seed", request.seed))
        packaging_seconds = float(result["packaging_seconds"])
        master_jpeg_quality = int(result["master_jpeg_quality"])
        depth_jpeg_quality = int(result["depth_jpeg_quality"])
    except (KeyError, TypeError, ValueError) as error:
        raise FiniteModalProviderError("warm fast class returned invalid output") from error
    if result.get("master_media_type") != "image/jpeg":
        raise FiniteModalProviderError("warm fast class returned an unsupported master format")
    if result.get("depth_media_type") != "image/jpeg":
        raise FiniteModalProviderError("warm fast class returned an unsupported depth format")
    if result.get("negative_prompt_supported") is not False:
        raise FiniteModalProviderError("warm fast class returned ambiguous prompt provenance")
    if result.get("depth_dtype") != DEPTH_DTYPE:
        raise FiniteModalProviderError("warm fast class returned ambiguous depth precision")
    if result.get("gpu") != WARM_FAST_GPU:
        raise FiniteModalProviderError("warm fast class ran on an unexpected GPU")
    if request.fidelity_label:
        if result.get("quality_passed") is not True:
            raise FiniteModalProviderError("warm fast class did not pass visual fidelity")
        if result.get("quality_label") != request.fidelity_label:
            raise FiniteModalProviderError("warm fast class changed the fidelity label")
        if int(result.get("quality_subject_count", 0)) != request.expected_subject_count:
            raise FiniteModalProviderError("warm fast class returned the wrong subject count")
        if result.get("quality_object_label", "") != request.fidelity_object_label:
            raise FiniteModalProviderError("warm fast class changed the fidelity object label")
        if request.fidelity_object_label and int(result.get("quality_object_count", 0)) != 1:
            raise FiniteModalProviderError("warm fast class returned the wrong object count")
        if (
            request.require_subject_object_overlap
            and result.get("quality_subject_object_overlap") is not True
        ):
            raise FiniteModalProviderError("warm fast class failed subject/object placement")
        if not 1 <= quality_attempts <= 2:
            raise FiniteModalProviderError("warm fast class returned invalid fidelity attempts")
    elif quality_attempts != 0:
        raise FiniteModalProviderError("warm fast class ran an unrequested fidelity gate")
    _validate_seed(selected_seed)
    if master_jpeg_quality != 95 or depth_jpeg_quality != 85:
        raise FiniteModalProviderError("warm fast class returned unexpected JPEG quality")
    if _jpeg_dimensions(master) != (request.width, request.height):
        raise FiniteModalProviderError("warm master dimensions do not match the request")
    if _jpeg_dimensions(depth) != (request.width, request.height):
        raise FiniteModalProviderError("warm depth dimensions do not match the request")
    _atomic_write(master_path, master)
    _atomic_write(depth_path, depth)
    now = datetime.now(UTC).isoformat()
    inference_seconds = image_seconds + depth_seconds + quality_seconds
    payload = {
        "schema_version": "1.0",
        "provider": PROVIDER_NAME,
        "scene_id": request.scene_id,
        "created_at": now,
        "updated_at": now,
        "request": {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "seed": request.seed,
            "selected_seed": selected_seed,
            "width": request.width,
            "height": request.height,
        },
        "policy": {
            "finite_calls_only": True,
            "persistent_endpoint": False,
            "provider_mode": "authenticated-deployed-class",
            "billing_reservation_id": reservation_id,
        },
        "stages": {
            "fast": {
                "model": FAST_MODEL,
                "model_revision": FAST_MODEL_REVISION,
                "additional_models": [
                    {"model": DEPTH_MODEL, "model_revision": DEPTH_MODEL_REVISION},
                    *(
                        [
                            {
                                "model": FIDELITY_MODEL,
                                "model_revision": FIDELITY_MODEL_REVISION,
                            }
                        ]
                        if request.fidelity_label
                        else []
                    ),
                ],
                "gpu": WARM_FAST_GPU,
                "finite_call": True,
                "hard_timeout_seconds": FAST_STAGE_POLICY.remote_timeout_seconds,
                "remote_seconds": remote_seconds,
                "inference_seconds": inference_seconds,
                "provider_overhead_seconds": max(0.0, remote_seconds - inference_seconds),
                "image_seconds": image_seconds,
                "depth_seconds": depth_seconds,
                "quality_seconds": quality_seconds,
                "quality_attempts": quality_attempts,
                "quality_label": result.get("quality_label", ""),
                "quality_object_label": result.get("quality_object_label", ""),
                "quality_expected_count": result.get("quality_expected_count"),
                "quality_subject_count": result.get("quality_subject_count"),
                "quality_object_count": result.get("quality_object_count"),
                "quality_subject_object_overlap": result.get(
                    "quality_subject_object_overlap"
                ),
                "quality_scores": result.get("quality_scores", []),
                "quality_passed": result.get("quality_passed"),
                "selected_seed": selected_seed,
                "packaging_seconds": packaging_seconds,
                "master_jpeg_quality": master_jpeg_quality,
                "depth_jpeg_quality": depth_jpeg_quality,
                "depth_dtype": DEPTH_DTYPE,
                "negative_prompt_supported": False,
                "model_load_seconds": float(result.get("model_load_seconds", 0)),
                "container_age_seconds": float(result.get("container_age_seconds", 0)),
                "warm_state": warm_state,
                "estimated_gpu_usd": estimated_gpu_usd,
                "steps": request.steps,
                "guidance_scale": request.guidance_scale,
            }
        },
        "artifacts": {
            "master": _local_artifact_payload(
                master_path,
                root=destination,
                mime_type="image/jpeg",
                width=request.width,
                height=request.height,
            ),
            "depth": _local_artifact_payload(
                depth_path,
                root=destination,
                mime_type="image/jpeg",
                width=request.width,
                height=request.height,
            ),
        },
    }
    _atomic_write_json(manifest_path, payload)
    return load_finite_scene_bundle(manifest_path)


def _write_warm_motion_bundle(
    *,
    source: FiniteSceneBundle,
    request: MotionUpgradeRequest,
    result: Mapping[str, Any],
    remote_seconds: float,
    estimated_gpu_usd: float,
    warm_state: str,
    reservation_id: str,
) -> FiniteSceneBundle:
    payload = json.loads(source.manifest_path.read_text())
    if payload["artifacts"]["master"]["sha256"] != source.master.sha256:
        raise FiniteModalProviderError("warm motion source master changed")
    motion_path = source.manifest_path.parent / "motion.mp4"
    if motion_path.exists():
        raise FiniteModalProviderError(f"motion output already exists: {motion_path}")
    try:
        content = bytes(result["content"])
        generation_seconds = float(result["generation_seconds"])
        frames = int(result["frames"])
        fps = int(result["fps"])
        duration_ms = int(result["duration_ms"])
    except (KeyError, TypeError, ValueError) as error:
        raise FiniteModalProviderError("warm motion class returned invalid output") from error
    if result.get("gpu") != MOTION_GPU:
        raise FiniteModalProviderError("warm motion class ran on an unexpected GPU")
    expected_frames = request.generated_frames * 2 - 1
    if frames != expected_frames or fps != request.fps or duration_ms <= 0:
        raise FiniteModalProviderError("warm motion metadata does not match the request")
    _atomic_write(motion_path, content)
    payload["updated_at"] = datetime.now(UTC).isoformat()
    payload["policy"]["billing_reservation_id"] = reservation_id
    payload["stages"]["motion"] = {
        "model": MOTION_MODEL,
        "model_revision": MOTION_MODEL_REVISION,
        "source_master_sha256": source.master.sha256,
        "gpu": MOTION_GPU,
        "finite_call": True,
        "hard_timeout_seconds": MOTION_STAGE_POLICY.remote_timeout_seconds,
        "remote_seconds": remote_seconds,
        "inference_seconds": generation_seconds,
        "provider_overhead_seconds": max(0.0, remote_seconds - generation_seconds),
        "model_load_seconds": float(result.get("model_load_seconds", 0)),
        "container_age_seconds": float(result.get("container_age_seconds", 0)),
        "warm_state": warm_state,
        "estimated_gpu_usd": estimated_gpu_usd,
        "steps": request.steps,
        "generated_frames": request.generated_frames,
        "loop_strategy": "exact-ping-pong",
    }
    payload["artifacts"]["motion"] = _local_artifact_payload(
        motion_path,
        root=source.manifest_path.parent,
        mime_type="video/mp4",
        width=request.width,
        height=request.height,
        duration_ms=duration_ms,
        frames=frames,
        fps=fps,
    )
    _atomic_write_json(source.manifest_path, payload)
    return load_finite_scene_bundle(source.manifest_path)


def _local_artifact_payload(
    path: Path,
    *,
    root: Path,
    mime_type: str,
    width: int,
    height: int,
    duration_ms: int = 0,
    frames: int = 1,
    fps: int = 0,
) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "mime_type": mime_type,
        "width": width,
        "height": height,
        "duration_ms": duration_ms,
        "frames": frames,
        "fps": fps,
    }


def _prewarm_receipt_artifact(path: Path) -> SceneArtifact:
    content = path.read_bytes()
    return SceneArtifact(
        role="prewarm",
        path=path,
        sha256=hashlib.sha256(content).hexdigest(),
        mime_type="application/json",
        width=1,
        height=1,
    )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _atomic_write(path, serialized.encode())


def _png_dimensions(content: bytes) -> tuple[int, int]:
    if len(content) < 24 or content[:8] != b"\x89PNG\r\n\x1a\n":
        raise FiniteModalProviderError("deployed Modal class did not return a valid PNG")
    return int.from_bytes(content[16:20], "big"), int.from_bytes(content[20:24], "big")


def _jpeg_dimensions(content: bytes) -> tuple[int, int]:
    if len(content) < 4 or content[:2] != b"\xff\xd8":
        raise FiniteModalProviderError("deployed Modal class did not return a valid JPEG")
    offset = 2
    start_of_frame = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while offset + 3 < len(content):
        if content[offset] != 0xFF:
            raise FiniteModalProviderError("deployed Modal class returned malformed JPEG data")
        while offset < len(content) and content[offset] == 0xFF:
            offset += 1
        if offset >= len(content):
            break
        marker = content[offset]
        offset += 1
        if marker == 0x01 or 0xD0 <= marker <= 0xD9:
            continue
        if offset + 2 > len(content):
            break
        segment_length = int.from_bytes(content[offset : offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(content):
            raise FiniteModalProviderError("deployed Modal class returned malformed JPEG data")
        if marker in start_of_frame:
            if segment_length < 7:
                break
            height = int.from_bytes(content[offset + 3 : offset + 5], "big")
            width = int.from_bytes(content[offset + 5 : offset + 7], "big")
            if width and height:
                return width, height
            break
        if marker == 0xDA:
            break
        offset += segment_length
    raise FiniteModalProviderError("deployed Modal class JPEG omitted dimensions")


def load_finite_scene_bundle(manifest_path: Path) -> FiniteSceneBundle:
    path = manifest_path.resolve()
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise FiniteModalProviderError(f"invalid scene manifest: {path}") from error
    if payload.get("schema_version") != "1.0":
        raise FiniteModalProviderError("unsupported finite scene manifest schema")
    if payload.get("provider") != PROVIDER_NAME:
        raise FiniteModalProviderError("scene manifest provider is not modal-finite")
    scene_id = payload.get("scene_id")
    if not isinstance(scene_id, str):
        raise FiniteModalProviderError("scene manifest has no scene_id")
    _validate_identifier(scene_id)
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, dict):
        raise FiniteModalProviderError("scene manifest has no artifacts")
    if not {"master", "depth"}.issubset(raw_artifacts):
        raise FiniteModalProviderError("scene manifest requires master and depth artifacts")
    artifacts: dict[str, SceneArtifact] = {}
    for role, raw in raw_artifacts.items():
        if role not in {"master", "depth", "motion"} or not isinstance(raw, dict):
            raise FiniteModalProviderError(f"unsupported scene artifact: {role}")
        artifact_path = _resolve_manifest_artifact(path, raw.get("path"))
        content = artifact_path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != raw.get("sha256"):
            raise FiniteModalProviderError(f"{role} checksum mismatch")
        artifacts[role] = SceneArtifact(
            role=role,
            path=artifact_path,
            sha256=digest,
            mime_type=str(raw["mime_type"]),
            width=int(raw["width"]),
            height=int(raw["height"]),
            duration_ms=int(raw.get("duration_ms", 0)),
            frames=int(raw.get("frames", 1)),
            fps=int(raw.get("fps", 0)),
        )
    stages = payload.get("stages")
    if not isinstance(stages, dict) or "fast" not in stages:
        raise FiniteModalProviderError("scene manifest requires the fast stage")
    _verify_stage(
        stages["fast"],
        model=FAST_MODEL,
        revision=FAST_MODEL_REVISION,
        gpu=FAST_GPU,
        legacy_gpus={WARM_FAST_GPU},
    )
    if "motion" in stages:
        _verify_stage(
            stages["motion"],
            model=MOTION_MODEL,
            revision=MOTION_MODEL_REVISION,
            gpu=MOTION_GPU,
        )
        if "motion" not in artifacts:
            raise FiniteModalProviderError("motion stage has no motion artifact")
    elif "motion" in artifacts:
        raise FiniteModalProviderError("motion artifact has no provenance stage")
    return FiniteSceneBundle(
        manifest_path=path,
        scene_id=scene_id,
        artifacts=artifacts,
        manifest=payload,
    )


def _verify_stage(
    stage: object,
    *,
    model: str,
    revision: str,
    gpu: str,
    legacy_gpus: set[str] | None = None,
) -> None:
    if not isinstance(stage, dict):
        raise FiniteModalProviderError("invalid scene provenance stage")
    if stage.get("model") != model or stage.get("model_revision") != revision:
        raise FiniteModalProviderError("scene provenance model revision mismatch")
    accepted_gpus = {gpu, *(legacy_gpus or set())}
    if stage.get("gpu") not in accepted_gpus or stage.get("finite_call") is not True:
        raise FiniteModalProviderError("scene provenance does not describe a supported finite call")
    amount = stage.get("estimated_gpu_usd")
    if not isinstance(amount, (float, int)) or not math.isfinite(amount) or amount < 0:
        raise FiniteModalProviderError("scene provenance has invalid estimated cost")


def _resolve_manifest_artifact(manifest_path: Path, relative_value: object) -> Path:
    if not isinstance(relative_value, str) or not relative_value:
        raise FiniteModalProviderError("artifact path is missing")
    relative = Path(relative_value)
    if relative.is_absolute():
        raise FiniteModalProviderError("artifact path must be relative to its manifest")
    root = manifest_path.parent.resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise FiniteModalProviderError("artifact path escapes its scene directory")
    if not resolved.is_file():
        raise FiniteModalProviderError(f"scene artifact does not exist: {relative_value}")
    return resolved


def _validate_identifier(value: str) -> None:
    if not value or len(value) > 96 or any(not (char.isalnum() or char in "-_") for char in value):
        raise ValueError("scene_id requires 1-96 letters, numbers, hyphens, or underscores")


def _validate_prompt(value: str, *, name: str) -> None:
    if not value.strip() or len(value) > 4_000:
        raise ValueError(f"{name} requires 1-4000 characters")


def _validate_seed(seed: int) -> None:
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer")


def _validate_dimensions(width: int, height: int, *, minimum: int) -> None:
    if not minimum <= width <= 1536 or not minimum <= height <= 1536:
        raise ValueError(f"dimensions must be between {minimum} and 1536")
    if width % 32 or height % 32:
        raise ValueError("dimensions must be divisible by 32")


async def _run_command(command: list[str], timeout_seconds: float) -> None:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        output, _ = await asyncio.wait_for(process.communicate(), timeout_seconds)
    except (TimeoutError, asyncio.CancelledError):
        _terminate_process_group(process)
        await process.wait()
        raise
    if process.returncode != 0:
        detail = output.decode("utf-8", errors="replace")[-4_000:]
        raise FiniteModalProviderError(
            f"finite Modal command failed ({process.returncode}): {detail}"
        )


async def _read_authoritative_modal_total(
    modal_executable: str,
    timeout_seconds: float,
) -> float:
    process = await asyncio.create_subprocess_exec(
        modal_executable,
        "billing",
        "report",
        "--for",
        "this month",
        "--json",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        output, _ = await asyncio.wait_for(process.communicate(), timeout_seconds)
    except (TimeoutError, asyncio.CancelledError):
        _terminate_process_group(process)
        await process.wait()
        raise
    if process.returncode != 0:
        detail = output.decode("utf-8", errors="replace")[-2_000:]
        raise FiniteModalBudgetError(
            f"Modal billing report failed ({process.returncode}): {detail}"
        )
    try:
        total = _parse_modal_billing_total(output)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise FiniteModalBudgetError("Modal billing report returned invalid JSON") from error
    return total


def _parse_modal_billing_total(payload: bytes | str) -> float:
    """Parse current and legacy Modal CLI billing report JSON strictly."""
    entries = json.loads(payload)
    if not isinstance(entries, list):
        raise ValueError("billing report is not a list")

    costs: list[float] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("billing report entry is not an object")
        current = entry.get("cost")
        legacy = entry.get("Cost")
        if current is None and legacy is None:
            raise ValueError("billing report entry has no cost")
        if current is not None and legacy is not None and str(current) != str(legacy):
            raise ValueError("billing report entry has conflicting costs")
        cost = float(current if current is not None else legacy)
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("billing report entry has an invalid cost")
        costs.append(cost)
    return math.fsum(costs)


def _terminate_process_group(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
