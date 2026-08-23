import json

from bookforge.domain import InterventionRequest, StoryCompileRequest

SYSTEM_PROMPT = """You are the structured reasoning engine for The Book That Listens Back.
Protect reader agency. Prefer waiting over helping, and the smallest useful clue over giving an
answer. Never diagnose a reader. Never add frightening, sexual, discriminatory, commercial, or
personally identifying content. Return only data that conforms to the supplied JSON schema."""


def intervention_prompt(request: InterventionRequest) -> str:
    return """Select exactly one allowed reading-support action.

Rules:
- Use wait for a short first pause.
- Use highlight_grapheme as the first scaffold for a decoding hesitation.
- Escalate to show_mouth or speak_sound only after another attempt.
- Never return an action outside allowed_actions.
- Keep display_text under 60 characters.

Reading event:
""" + json.dumps(request.model_dump(mode="json"), indent=2)


def story_compile_prompt(request: StoryCompileRequest) -> str:
    return """Compile this book into projection-ready page plans.

Requirements:
- Preserve the supplied story. Do not rewrite or extend its plot.
- For every page, return scene_spec version 2.0. It is the authoritative 1920x1080 composition.
- scene_spec.master_prompt must describe one cohesive, finished, cinematic 16:9 illustration with
  no text, labels, interface, frames, split panels, or placeholder geometry. Include the supplied
  visual style, clear spatial composition, lighting, materials, palette, and all story subjects.
- scene_spec.composition must contain every visual layer exactly once. Use normalized coordinates
  from 0 to 1 for center_x, center_y, width, and height; keep each region inside the canvas.
  Use non-negative depth values up to 20 and assign smaller values to distant layers. Regions must
  tightly cover the intended subject so local effects land correctly.
- Give the camera a restrained, loopable ambient move. Give each layer deterministic loopable
  ambient_motion; backgrounds should use subtle parallax and characters may float or breathe.
- Add zero to three restrained ambient effects that support the page rather than distracting from
  reading.
- Make every visual layer independently renderable and stylistically consistent.
- Every trigger word must be exactly one token that literally appears on that page; never return
  a phrase such as "red gate". Anchor a phrase-level event to its final spoken word instead.
- Use concise image-generation prompts with subject, composition, palette, and transparent-layer
  intent where relevant.
- Motion descriptions must be deterministic and achievable with 2D transforms, opacity, masks,
  particles, or short cached video loops.
- Provide literacy support only for genuinely decodable or vocabulary-rich words.
- Hint ladders must go from least to most explicit without revealing the whole word first.
- Include at most two comprehension prompts per page.
- Every trigger target_layer_id must reference a layer on the same page.
- Return one page plan for every input page, preserving page_id exactly.

Book input:
""" + json.dumps(request.model_dump(mode="json"), indent=2)
