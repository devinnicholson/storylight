from storylight.config import Settings
from storylight.domain import (
    CompileResponse,
    GeneratedStoryPlan,
    InterventionDecision,
    InterventionRequest,
    InterventionResponse,
    StoryCompileRequest,
    StoryPack,
    SupportAction,
)
from storylight.model_client import StructuredModelClient
from storylight.prompts import SYSTEM_PROMPT, intervention_prompt, story_compile_prompt
from storylight.reader import tokenize


class StorylightService:
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
        generated = self._validate_story_plan(request, generated)
        source_text_by_page = {page.page_id: page.text for page in request.pages}
        compiled_pages = [
            page.model_copy(update={"source_text": source_text_by_page[page.page_id]})
            for page in generated.pages
        ]
        pack = StoryPack(
            schema_version="2.0",
            story_id=request.story_id,
            title=request.title,
            reading_level=request.reading_level,
            visual_style=request.visual_style,
            compiler_model=metrics.model,
            pages=compiled_pages,
        )
        return CompileResponse(story_pack=pack, metrics=metrics)

    @staticmethod
    def _validate_story_plan(
        request: StoryCompileRequest, plan: GeneratedStoryPlan
    ) -> GeneratedStoryPlan:
        expected_pages = [page.page_id for page in request.pages]
        actual_pages = [page.page_id for page in plan.pages]
        if actual_pages != expected_pages:
            message = (
                f"Model changed page identity or order: expected {expected_pages}, "
                f"got {actual_pages}"
            )
            raise ValueError(message)

        source_pages = {page.page_id: tokenize(page.text) for page in request.pages}
        normalized_pages = []
        for page in plan.pages:
            if page.scene_spec is None:
                raise ValueError(f"Model omitted SceneSpec v2 for page {page.page_id}")
            layer_ids = {layer.layer_id for layer in page.layers}
            source_words = source_pages[page.page_id]
            normalized_triggers = []
            for trigger in page.triggers:
                if trigger.target_layer_id not in layer_ids:
                    message = (
                        f"Trigger {trigger.trigger_id} references missing layer "
                        f"{trigger.target_layer_id}"
                    )
                    raise ValueError(message)
                trigger_words = tokenize(trigger.word)
                if not trigger_words:
                    raise ValueError(
                        f"Trigger word {trigger.word!r} does not occur on page {page.page_id}"
                    )
                phrase_starts = [
                    index
                    for index in range(len(source_words) - len(trigger_words) + 1)
                    if source_words[index : index + len(trigger_words)] == trigger_words
                ]
                if trigger.occurrence > len(phrase_starts):
                    raise ValueError(
                        f"Trigger word {trigger.word!r} occurrence {trigger.occurrence} "
                        f"does not occur on page {page.page_id}"
                    )
                phrase_start = phrase_starts[trigger.occurrence - 1]
                anchor_index = phrase_start + len(trigger_words) - 1
                anchor_word = source_words[anchor_index]
                anchor_occurrence = sum(
                    word == anchor_word for word in source_words[: anchor_index + 1]
                )
                normalized_triggers.append(
                    trigger.model_copy(
                        update={"word": anchor_word, "occurrence": anchor_occurrence}
                    )
                )
            normalized_pages.append(page.model_copy(update={"triggers": normalized_triggers}))
        return plan.model_copy(update={"pages": normalized_pages})
