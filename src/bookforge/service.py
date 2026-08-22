from bookforge.config import Settings
from bookforge.domain import (
    CompileResponse,
    GeneratedStoryPlan,
    InterventionDecision,
    InterventionRequest,
    InterventionResponse,
    StoryCompileRequest,
    StoryPack,
    SupportAction,
)
from bookforge.model_client import StructuredModelClient
from bookforge.prompts import SYSTEM_PROMPT, intervention_prompt, story_compile_prompt


class BookforgeService:
    def __init__(self, settings: Settings, model_client: StructuredModelClient) -> None:
        self.settings = settings
        self.model_client = model_client

    async def select_intervention(self, request: InterventionRequest) -> InterventionResponse:
        if request.pause_ms < 900 and request.attempt_count == 0:
            return InterventionResponse(
                source="fast_path",
                decision=InterventionDecision(
                    action=SupportAction.WAIT,
                    rationale_code="not_needed",
                    confidence=1,
                ),
            )

        if request.attempt_count == 0 and request.expected_grapheme:
            return InterventionResponse(
                source="fast_path",
                decision=InterventionDecision(
                    action=SupportAction.HIGHLIGHT_GRAPHEME,
                    target=request.expected_grapheme,
                    display_text=f"Try the {request.expected_grapheme} sound.",
                    rationale_code="initial_pause",
                    confidence=1,
                ),
            )

        decision, metrics = await self.model_client.generate(
            system=SYSTEM_PROMPT,
            prompt=intervention_prompt(request),
            output_type=InterventionDecision,
        )
        if decision.action not in request.allowed_actions:
            decision = InterventionDecision(
                action=SupportAction.WAIT,
                rationale_code="not_needed",
                confidence=0,
                display_text="",
                target="",
            )
        elif decision.action is SupportAction.HIGHLIGHT_GRAPHEME and not decision.target:
            decision = decision.model_copy(
                update={
                    "target": request.expected_grapheme,
                    "display_text": decision.display_text
                    or f"Try the {request.expected_grapheme} sound.",
                }
            )
        return InterventionResponse(decision=decision, source="model", metrics=metrics)

    async def compile_story(self, request: StoryCompileRequest) -> CompileResponse:
        generated, metrics = await self.model_client.generate(
            system=SYSTEM_PROMPT,
            prompt=story_compile_prompt(request),
            output_type=GeneratedStoryPlan,
        )
        self._validate_story_plan(request, generated)
        source_text_by_page = {page.page_id: page.text for page in request.pages}
        compiled_pages = [
            page.model_copy(update={"source_text": source_text_by_page[page.page_id]})
            for page in generated.pages
        ]
        pack = StoryPack(
            story_id=request.story_id,
            title=request.title,
            reading_level=request.reading_level,
            visual_style=request.visual_style,
            compiler_model=metrics.model,
            pages=compiled_pages,
        )
        return CompileResponse(story_pack=pack, metrics=metrics)

    @staticmethod
    def _validate_story_plan(request: StoryCompileRequest, plan: GeneratedStoryPlan) -> None:
        expected_pages = [page.page_id for page in request.pages]
        actual_pages = [page.page_id for page in plan.pages]
        if actual_pages != expected_pages:
            message = (
                f"Model changed page identity or order: expected {expected_pages}, "
                f"got {actual_pages}"
            )
            raise ValueError(message)

        source_pages = {page.page_id: page.text.lower().split() for page in request.pages}
        for page in plan.pages:
            layer_ids = {layer.layer_id for layer in page.layers}
            source_words = {
                word.strip(".,!?;:\"'()[]{}").lower() for word in source_pages[page.page_id]
            }
            for trigger in page.triggers:
                if trigger.target_layer_id not in layer_ids:
                    message = (
                        f"Trigger {trigger.trigger_id} references missing layer "
                        f"{trigger.target_layer_id}"
                    )
                    raise ValueError(message)
                if trigger.word.strip(".,!?;:\"'()[]{}").lower() not in source_words:
                    raise ValueError(
                        f"Trigger word {trigger.word!r} does not occur on page {page.page_id}"
                    )
