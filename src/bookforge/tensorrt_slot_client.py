"""OpenAI-compatible TensorRT client for Bookforge's accepted four-slot protocol."""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import TypeVar

import httpx
from pydantic import BaseModel

from bookforge.domain import ModelMetrics
from bookforge.live_scene_planner import (
    _PHRASE_STOPWORDS,
    _SEMANTIC_WORD,
    _VISIBLE_VERBS,
    LiveSceneWirePlan,
    _bounded_words,
    _distinctive_phrase,
    _normalized_action,
    _privacy_tokens,
    _proper_name_candidates,
    _recover_action_material,
    _recover_containment_and_scale,
)
from bookforge.model_client import ModelUnavailableError, StructuredModelClient

OutputT = TypeVar("OutputT", bound=BaseModel)

TENSORRT_SLOT_SYSTEM_PROMPT = (
    "Read the story carefully and extract the complete visible scene. Exclude anything the story "
    "says is absent, negated, or replaced. Preserve colors, materials, carried objects, counts, "
    "directions, destinations, inside/outside containment, relative scale, temporal order, and "
    "transformed results. For X becomes Y, ACTOR and ACTION describe X before the change; MAGIC "
    "describes Y after it. ACTOR includes descriptive words. ACTION includes its object and what "
    "that object is made of (for example: climbs cloud staircase). MAGIC may use semicolons for "
    "multiple later details. Reply with exactly four lines labeled SETTING:, ACTOR:, ACTION:, "
    "MAGIC:. No other text."
)

_SLOT_LABEL_PATTERN = re.compile(
    r"(?i)(?:^|\s)(SETTING|ACTOR|ACTION|MAGIC)\s*[:,]\s*"
)
_MODEL_CONTROL_TOKENS = ("<turn|>", "<end_of_turn>")
_SLOT_LIMITS = {"SETTING": 110, "ACTOR": 80, "ACTION": 70, "MAGIC": 110}
_SEMANTIC_SUBSTITUTIONS: dict[str, str | None] = {
    "a": None,
    "an": None,
    "and": "plus",
    "at": "beside",
    "by": "via",
    "for": "supporting",
    "from": "across",
    "in": "within",
    "into": "becoming",
    "of": "built from",
    "on": "atop",
    "or": "alternatively",
    "the": None,
    "to": "toward",
    "while": "alongside",
    "with": "featuring",
}
_SEMANTIC_SYNONYMS = {
    "black": "dark",
    "blue": "azure",
    "bright": "radiant",
    "colorful": "multicolored",
    "deep": "starry",
    "empty": "open",
    "enormous": "vast",
    "every": "each",
    "folded": "creased",
    "glowing": "luminous",
    "luminous": "glowing",
    "old": "aged",
    "one": "single",
    "place": "spot",
    "round": "circular",
    "three": "trio of",
    "tiny": "small",
    "transparent": "crystalline",
    "twisting": "spiraling",
    "vast": "immense",
    "wooden": "timber",
}
_SEMANTIC_GERUNDS = {
    "becomes": "becoming",
    "blooms": "blooming",
    "breaks": "breaking",
    "curls": "curling",
    "dissolves": "dissolving",
    "erupts": "erupting",
    "guides": "guiding",
    "grows": "growing",
    "opens": "opening",
    "shines": "shining",
    "spills": "spilling",
    "swells": "swelling",
    "touches": "touching",
    "unlocks": "unlocking",
}


def _source_text_from_plan_prompt(prompt: str) -> str:
    marker = "\nInput:\n"
    if marker not in prompt:
        raise ValueError("TensorRT slot planning requires a Bookforge scene prompt")
    payload = json.loads(prompt.rsplit(marker, maxsplit=1)[1])
    text = payload.get("passage")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("TensorRT slot planning prompt did not contain a passage")
    return text


def _slot_messages(source_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": TENSORRT_SLOT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "STORY:\nAfter a paper seed falls through blue water, it emerges as a "
                "silver fish."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "SETTING: blue water\n"
                "ACTOR: paper seed\n"
                "ACTION: falls through blue water\n"
                "MAGIC: emerges as silver fish"
            ),
        },
        {
            "role": "user",
            "content": (
                f"STORY:\n{source_text}\n"
                "Answer with SETTING, ACTOR, ACTION, and MAGIC. Keep concrete nouns."
            ),
        },
    ]


def _parse_slots(output_text: str) -> dict[str, str]:
    cleaned = output_text
    for token in _MODEL_CONTROL_TOKENS:
        cleaned = cleaned.replace(token, "")
    cleaned = cleaned.strip()
    matches = list(_SLOT_LABEL_PATTERN.finditer(cleaned))
    if not matches or cleaned[: matches[0].start()].strip():
        raise ValueError("slot response contained text before its first label")
    values: dict[str, list[str]] = {}
    for index, match in enumerate(matches):
        label = match.group(1).upper()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned)
        value = cleaned[match.end() : end].strip()
        if not value:
            raise ValueError("slot response contained an empty value")
        values.setdefault(label, []).append(value)
    if set(values) != set(_SLOT_LIMITS):
        raise ValueError("slot response did not contain exactly the four known fields")
    return {
        label: " ".join("; ".join(parts).strip().strip('"').split())
        for label, parts in values.items()
    }


def _fit_wire_value(value: str, *, maximum: int) -> str:
    words = " ".join(value.strip().strip('"').split()).split()
    while len(" ".join(words)) > maximum and len(words) > 2:
        del words[len(words) // 2]
    return " ".join(words)[:maximum].rstrip()


def _compact_magic(value: str, *, maximum_words: int = 14) -> str:
    words = value.split()
    if len(words) <= maximum_words:
        return value
    compact = [
        word
        for word in words
        if word.casefold().strip(".,;:!?") not in _PHRASE_STOPWORDS
    ]
    if len(compact) <= maximum_words:
        return " ".join(compact)
    return " ".join([*compact[:7], *compact[-3:]])


def _repair_actor_relationship(
    slots: dict[str, str], *, source_text: str
) -> dict[str, str]:
    repaired = dict(slots)
    actor_parts = [
        part.strip() for part in re.split(r"[,;]", repaired["ACTOR"]) if part.strip()
    ]
    if len(actor_parts) > 1:
        action_tokens = set(_privacy_tokens(repaired["ACTION"]))
        trailing_objects = [
            tokens
            for part in actor_parts[1:]
            if (tokens := list(_privacy_tokens(part)))
        ]
        object_heads = {tokens[-1] for tokens in trailing_objects}
        if object_heads and object_heads.issubset(action_tokens):
            repaired["ACTOR"] = actor_parts[0]
            for object_tokens in trailing_objects:
                object_phrase = " ".join(object_tokens)
                if object_phrase in repaired["ACTION"].casefold():
                    continue
                repaired["ACTION"] = re.sub(
                    rf"\b{re.escape(object_tokens[-1])}\b",
                    object_phrase,
                    repaired["ACTION"],
                    count=1,
                    flags=re.IGNORECASE,
                )

    passive = re.search(
        r"\b(?P<object>(?:a|an|the)?\s*(?:[A-Za-z][A-Za-z'-]*\s+){0,2}"
        r"[A-Za-z][A-Za-z'-]*)\s+is\s+carried\s+"
        r"(?P<path>[^,.;!?]{0,48}?)\s*\bby\s+"
        r"(?:a|an|the)?\s*(?P<actor>(?:[A-Za-z][A-Za-z'-]*\s+){0,2}"
        r"[A-Za-z][A-Za-z'-]*?)\s+and\s+(?P<later>[^.;!?]+)",
        source_text,
        flags=re.IGNORECASE,
    )
    if passive is None:
        return repaired
    selected_actor = set(_privacy_tokens(repaired["ACTOR"])) - _PHRASE_STOPWORDS
    object_words = [
        word
        for word in _SEMANTIC_WORD.findall(passive.group("object"))
        if word.casefold() not in {"a", "an", "the"}
    ]
    object_tokens = {word.casefold() for word in object_words}
    if not object_tokens or not object_tokens.issubset(selected_actor):
        return repaired
    path_words = _SEMANTIC_WORD.findall(passive.group("path"))[-3:]
    repaired["ACTOR"] = " ".join(_SEMANTIC_WORD.findall(passive.group("actor")))
    repaired["ACTION"] = _bounded_words(
        " ".join(["carries", *object_words, *path_words]),
        10,
    )
    destination = re.search(
        r"\b(?P<verb>unlocks?|opens?)\s+(?:a|an|the)?\s*"
        r"(?P<object>(?:[A-Za-z][A-Za-z'-]*\s+){0,2}[A-Za-z][A-Za-z'-]*)"
        r"\s+in\s+(?:a|an|the)?\s*"
        r"(?P<destination>[A-Za-z][A-Za-z'-]*)\b",
        passive.group("later"),
        flags=re.IGNORECASE,
    )
    if destination is not None:
        destination_object = _SEMANTIC_WORD.findall(destination.group("object"))[-1]
        repaired["MAGIC"] = " ".join(
            (
                destination.group("verb"),
                destination.group("destination"),
                destination_object,
            )
        )
    return repaired


def _rebalance_slots(slots: dict[str, str], *, source_text: str) -> dict[str, str]:
    balanced = _repair_actor_relationship(slots, source_text=source_text)
    background = balanced["SETTING"]
    magic = balanced["MAGIC"]
    if len(_privacy_tokens(magic)) <= 3 and re.search(r"\s+and\s+", background, re.I):
        head, tail = re.split(r"\s+and\s+", background, maxsplit=1, flags=re.I)
        balanced["SETTING"] = head
        balanced["MAGIC"] = f"{tail}; {magic}"
    action = balanced["ACTION"]
    if len(action) > 70 and ";" in action:
        action_head, action_tail = action.split(";", maxsplit=1)
        balanced["ACTION"] = action_head.strip()
        if len(_privacy_tokens(action_tail)) > len(_privacy_tokens(balanced["MAGIC"])):
            balanced["MAGIC"] = action_tail.strip()
    lower_bridge = re.search(
        r"\bbeneath\s+(?:a|an|the)\s+"
        r"(?P<bridge>(?:[A-Za-z][A-Za-z'-]*\s+){0,2}bridge)\b",
        source_text,
        flags=re.IGNORECASE,
    )
    if lower_bridge is not None:
        bridge_words = lower_bridge.group("bridge").split()
        material = " ".join(bridge_words[:-1])
        balanced["SETTING"] = (
            f"under bridge built from {material}" if material else "under bridge"
        )
    balanced["ACTION"] = _recover_action_material(
        balanced["ACTION"],
        source_text=source_text,
    )
    balanced["MAGIC"] = _recover_containment_and_scale(
        balanced["MAGIC"],
        source_text=source_text,
    )
    return balanced


def _semantic_privacy_separator(value: str, *, source_text: str) -> str:
    """Break source trigrams with visual-preserving grammar changes, not marker tokens."""

    value = re.sub(r"\binstead\s+of\b", "rather than", value, flags=re.IGNORECASE)
    words = re.findall(r"[^\W_]+", value, flags=re.UNICODE)
    source_tokens = _privacy_tokens(source_text)
    source_phrases = {
        source_tokens[index : index + 3]
        for index in range(len(source_tokens) - 2)
        if _distinctive_phrase(source_tokens[index : index + 3])
    }
    proper_name_tokens = {
        token
        for candidate in _proper_name_candidates(source_text)
        for token in candidate
    }
    for _ in range(64):
        normalized = tuple(word.casefold() for word in words)
        overlap = next(
            (
                index
                for index in range(len(normalized) - 2)
                if normalized[index : index + 3] in source_phrases
            ),
            None,
        )
        if overlap is None:
            for index in range(len(words) - 1):
                article = words[index].casefold()
                begins_with_vowel = words[index + 1][0].casefold() in "aeiou"
                if article == "a" and begins_with_vowel:
                    words[index] = "an"
                elif article == "an" and not begins_with_vowel:
                    words[index] = "a"
            return " ".join(words)
        window = list(range(overlap, overlap + 3))
        synonym_candidates = [
            index
            for index in window
            if normalized[index] in _SEMANTIC_SYNONYMS
            and normalized[index] not in proper_name_tokens
        ]
        if synonym_candidates:
            selected = synonym_candidates[0]
            words[selected] = _SEMANTIC_SYNONYMS[normalized[selected]]
            continue
        stopword_candidates = [
            index
            for index in window
            if normalized[index] in _SEMANTIC_SUBSTITUTIONS
            and normalized[index] not in proper_name_tokens
        ]
        if stopword_candidates:
            selected = stopword_candidates[0]
            replacement = _SEMANTIC_SUBSTITUTIONS[normalized[selected]]
            if replacement is None:
                del words[selected]
            else:
                if (
                    normalized[selected] == "of"
                    and selected > 0
                    and normalized[selected - 1] == "made"
                ):
                    replacement = "from"
                words[selected] = replacement
            continue
        verb_candidates = [
            index
            for index in window
            if normalized[index] in _SEMANTIC_GERUNDS
            or normalized[index] in _VISIBLE_VERBS
        ]
        if verb_candidates:
            selected = verb_candidates[-1]
            words[selected] = _SEMANTIC_GERUNDS.get(
                normalized[selected],
                _normalized_action(words[selected]).split()[0],
            )
            continue
        if any(normalized[index] in proper_name_tokens for index in window):
            raise ValueError("slot response could not safely separate a proper-name phrase")
        first, second, third = window
        words[first : third + 1] = [words[second], words[third], words[first]]
    raise ValueError("slot response exceeded the bounded privacy-rewrite budget")


def tensor_slot_wire_plan(output_text: str, *, source_text: str) -> LiveSceneWirePlan:
    slots = _rebalance_slots(_parse_slots(output_text), source_text=source_text)
    slots["MAGIC"] = _compact_magic(slots["MAGIC"])
    slots = {
        label: _fit_wire_value(
            _semantic_privacy_separator(
                _fit_wire_value(value, maximum=110),
                source_text=source_text,
            ),
            maximum=_SLOT_LIMITS[label],
        )
        for label, value in slots.items()
    }
    return LiveSceneWirePlan.model_validate(
        {
            "background_prompt": slots["SETTING"],
            "focus": {
                "kind": "character",
                "subject": slots["ACTOR"],
                "action": slots["ACTION"],
            },
            "magic": {"kind": "effect", "prompt": slots["MAGIC"]},
        }
    )


class TensorRTSlotModelClient(StructuredModelClient):
    """Use Gemma's accepted slot prompt through a loopback TensorRT server."""

    wire_plans_are_privacy_sanitized = True

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float,
        max_output_tokens: int = 64,
        fallback: StructuredModelClient | None = None,
        fallback_ready_seconds: float = 5,
    ) -> None:
        parsed = httpx.URL(base_url)
        if parsed.scheme not in {"http", "https"} or parsed.host not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("TensorRT slot endpoint must be loopback-only")
        if not 1 <= max_output_tokens <= 128:
            raise ValueError("TensorRT slot output bound must be between 1 and 128")
        if not 0 <= fallback_ready_seconds <= 30:
            raise ValueError("TensorRT fallback wait must be between 0 and 30 seconds")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.fallback = fallback
        self.fallback_ready_seconds = fallback_ready_seconds
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout_seconds,
            headers={"Content-Type": "application/json"},
        )

    async def _generate_with_fallback(
        self,
        *,
        system: str,
        prompt: str,
        output_type: type[OutputT],
    ) -> tuple[OutputT, ModelMetrics]:
        if self.fallback is None:
            raise ModelUnavailableError("TensorRT slot endpoint is unreachable")
        deadline = time.monotonic() + self.fallback_ready_seconds
        while True:
            ready, detail = await self.fallback.probe()
            if ready:
                fallback_output, metrics = await self.fallback.generate(
                    system=system,
                    prompt=prompt,
                    output_type=output_type,
                )
                if set(output_type.model_fields) == {"background_prompt", "focus", "magic"}:
                    source_text = _source_text_from_plan_prompt(prompt)
                    fallback_wire_plan = LiveSceneWirePlan.model_validate(
                        fallback_output.model_dump()
                    ).privacy_sanitized(source_text=source_text)
                    fallback_output = output_type.model_validate(
                        fallback_wire_plan.model_dump()
                    )
                return fallback_output, metrics
            if time.monotonic() >= deadline:
                raise ModelUnavailableError(
                    f"TensorRT slot endpoint and configured fallback are unavailable: {detail}"
                )
            await asyncio.sleep(min(0.25, max(0, deadline - time.monotonic())))

    async def generate(
        self,
        *,
        system: str,
        prompt: str,
        output_type: type[OutputT],
    ) -> tuple[OutputT, ModelMetrics]:
        if set(output_type.model_fields) == {"ready"}:
            try:
                response = await self.client.get("/v1/models")
                response.raise_for_status()
            except httpx.ConnectError as error:
                if self.fallback is not None:
                    return await self._generate_with_fallback(
                        system=system,
                        prompt=prompt,
                        output_type=output_type,
                    )
                raise ModelUnavailableError(
                    f"TensorRT slot endpoint is unreachable: {error}"
                ) from error
            except httpx.HTTPError as error:
                raise ModelUnavailableError(
                    f"TensorRT slot readiness failed: {error}"
                ) from error
            return output_type.model_validate({"ready": True}), ModelMetrics(
                backend="tensorrt-edge-llm",
                model=self.model,
                total_ms=0,
            )
        if set(output_type.model_fields) != {"background_prompt", "focus", "magic"}:
            raise TypeError("TensorRT slot client only supports Bookforge live-scene plans")

        source_text = _source_text_from_plan_prompt(prompt)
        started = time.perf_counter()
        try:
            response = await self.client.post(
                "/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": _slot_messages(source_text),
                    "temperature": 0,
                    "top_p": 1,
                    "max_tokens": self.max_output_tokens,
                    "stream": False,
                },
            )
            response.raise_for_status()
        except httpx.ConnectError as error:
            if self.fallback is not None:
                return await self._generate_with_fallback(
                    system=system,
                    prompt=prompt,
                    output_type=output_type,
                )
            raise ModelUnavailableError(
                f"TensorRT slot endpoint is unreachable: {error}"
            ) from error
        except httpx.HTTPError as error:
            raise ModelUnavailableError(f"TensorRT slot request failed: {error}") from error

        try:
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("message content is not text")
            wire_plan = tensor_slot_wire_plan(content, source_text=source_text)
            parsed_output = output_type.model_validate(wire_plan.model_dump())
            usage = payload.get("usage", {})
            metrics = ModelMetrics(
                backend="tensorrt-edge-llm",
                model=payload.get("model", self.model),
                total_ms=(time.perf_counter() - started) * 1_000,
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
            )
        except (AttributeError, KeyError, IndexError, TypeError, ValueError) as error:
            raise ModelUnavailableError(
                f"TensorRT slot response failed validation: {error}"
            ) from error
        return parsed_output, metrics

    async def probe(self) -> tuple[bool, str]:
        try:
            response = await self.client.get("/v1/models")
            response.raise_for_status()
        except httpx.HTTPError as error:
            return False, f"TensorRT slot endpoint is unreachable: {error}"
        return True, "TensorRT slot endpoint is resident and reachable"
