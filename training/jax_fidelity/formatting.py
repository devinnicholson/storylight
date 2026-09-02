"""Exact production message formatting and completion-only mask checks."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from bookforge.tensorrt_slot_client import _slot_messages

SLOT_LABELS = ("SETTING", "ACTOR", "ACTION", "MAGIC")
_TARGET_LINE = re.compile(r"^(SETTING|ACTOR|ACTION|MAGIC): ([^\r\n]+)$")
_PROMPT_CONTRACT_SENTINEL = "__BOOKFORGE_STORY_TEXT__"


def prompt_contract_sha256() -> str:
    """Hash the exact deployed message structure without binding a real passage."""

    payload = json.dumps(
        _slot_messages(_PROMPT_CONTRACT_SENTINEL),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_slot_target(target: str) -> str:
    """Require the canonical four lines; training never learns parser repairs."""

    if not isinstance(target, str):
        raise TypeError("slot target must be text")
    if target != target.strip():
        raise ValueError("slot target may not have surrounding whitespace")
    lines = target.splitlines()
    if len(lines) != len(SLOT_LABELS):
        raise ValueError("slot target must contain exactly four lines")
    observed: list[str] = []
    for line in lines:
        match = _TARGET_LINE.fullmatch(line)
        if match is None:
            raise ValueError("each slot must be a non-empty 'LABEL: value' line")
        observed.append(match.group(1))
    if tuple(observed) != SLOT_LABELS:
        raise ValueError(f"slot labels must appear once in order: {', '.join(SLOT_LABELS)}")
    return target


def target_from_slots(slots: Mapping[str, Any]) -> str:
    """Build the strict target from a record's slot mapping."""

    if set(slots) != set(SLOT_LABELS):
        raise ValueError("slot mapping must contain exactly the four production labels")
    lines: list[str] = []
    for label in SLOT_LABELS:
        value = slots[label]
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError(f"{label} must be non-empty normalized text")
        if "\n" in value or "\r" in value:
            raise ValueError(f"{label} may not contain line breaks")
        lines.append(f"{label}: {value}")
    return validate_slot_target("\n".join(lines))


def production_messages(story: str, *, target: str | None = None) -> list[dict[str, str]]:
    """Use the deployed formatter directly so training cannot silently drift."""

    if not isinstance(story, str) or not story.strip():
        raise ValueError("story must be non-empty text")
    if story != story.strip():
        raise ValueError("story must not have surrounding whitespace")
    messages = [dict(message) for message in _slot_messages(story)]
    if target is not None:
        messages.append({"role": "assistant", "content": validate_slot_target(target)})
    return messages


def format_training_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one fidelity record to MaxText's conversational SFT format."""

    story = record.get("passage", record.get("story"))
    if not isinstance(story, str):
        raise ValueError("record must contain a string passage or story")
    raw_target = record.get("target", record.get("target_slots", record.get("slots")))
    if isinstance(raw_target, Mapping):
        target = target_from_slots(raw_target)
    elif isinstance(raw_target, str):
        target = validate_slot_target(raw_target)
    else:
        raise ValueError("record must contain target text or a target slot mapping")
    return {
        "record_id": record.get("record_id", record.get("id")),
        "messages": production_messages(story, target=target),
    }


def _token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("tokenizer did not return an input ID sequence")
    result = list(value)
    if not result or any(type(item) is not int for item in result):
        raise ValueError("tokenizer returned empty or non-integer input IDs")
    return result


def _maxtext_completion_text(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> str:
    full_ids = _token_ids(
        tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True)
    )
    prompt_ids = _token_ids(
        tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=True)
    )
    common_length = 0
    for full_id, prompt_id in zip(full_ids, prompt_ids, strict=False):
        if full_id != prompt_id:
            break
        common_length += 1
    if common_length == 0 or common_length == len(full_ids):
        raise ValueError("chat template has no stable prompt/completion boundary")
    return tokenizer.decode(full_ids[common_length:], skip_special_tokens=False)


def maxtext_sft_segments(
    tokenizer: Any, messages: Sequence[Mapping[str, str]]
) -> list[tuple[str, bool]]:
    """Mirror pinned MaxText v0.2.4 conversational SFT segmentation."""

    segments: list[tuple[str, bool]] = []
    round_messages: list[Mapping[str, str]] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "system":
            if index != 0:
                raise ValueError("system message must be first")
            round_messages.append(message)
        elif role == "user":
            round_messages.append(message)
            rendered = tokenizer.apply_chat_template(
                round_messages, add_generation_prompt=True, tokenize=False
            )
            if not isinstance(rendered, str) or not rendered:
                raise ValueError("chat template returned an empty prompt segment")
            segments.append((rendered, True))
        elif role == "assistant":
            round_messages.append(message)
            segments.append((_maxtext_completion_text(tokenizer, round_messages), False))
            round_messages = []
        else:
            raise ValueError(f"unsupported chat role: {role!r}")
    if round_messages:
        raise ValueError("conversation does not end with an assistant completion")
    return segments


def completion_only_example(
    tokenizer: Any,
    *,
    story: str,
    target: str,
    input_budget_tokens: int = 512,
    completion_budget_tokens: int = 64,
    masked_label_id: int = 0,
) -> dict[str, list[int]]:
    """Build pinned MaxText's completion-only mask for the sole assistant turn."""

    segments = maxtext_sft_segments(tokenizer, production_messages(story, target=target))
    tokenized = [
        _token_ids(tokenizer(text, truncation=False, max_length=input_budget_tokens))
        for text, _ in segments
    ]
    prefix_count = sum(len(ids) for ids in tokenized[:-1])
    final_completion_count = len(tokenized[-1])
    if prefix_count > input_budget_tokens:
        raise ValueError(f"prompt uses {prefix_count} tokens; budget is {input_budget_tokens}")
    if final_completion_count > completion_budget_tokens:
        raise ValueError(
            f"completion uses {final_completion_count} tokens; budget is {completion_budget_tokens}"
        )
    input_ids = [token for ids in tokenized for token in ids]
    labels = [
        masked_label_id if is_prompt else token
        for ids, (_, is_prompt) in zip(tokenized, segments, strict=True)
        for token in ids
    ]
    supervised_count = sum(
        len(ids) for ids, (_, is_prompt) in zip(tokenized, segments, strict=True) if not is_prompt
    )
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "prompt_token_count": [prefix_count],
        "masked_prompt_token_count": [
            sum(len(ids) for ids, (_, flag) in zip(tokenized, segments, strict=True) if flag)
        ],
        "completion_token_count": [final_completion_count],
        "supervised_token_count": [supervised_count],
        "segment_is_prompt": [int(is_prompt) for _, is_prompt in segments],
        "segment_token_counts": [len(ids) for ids in tokenized],
    }
