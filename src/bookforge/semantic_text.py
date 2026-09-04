"""Shared lexical normalization for graph construction, grounding, and evaluation."""

from __future__ import annotations

import re
import unicodedata

_IRREGULAR_LEMMAS = {
    "are": "be",
    "became": "become",
    "becomes": "become",
    "carried": "carry",
    "carries": "carry",
    "children": "child",
    "feet": "foot",
    "held": "hold",
    "is": "be",
    "mice": "mouse",
    "ran": "run",
    "swam": "swim",
    "was": "be",
    "were": "be",
    "wore": "wear",
}


def semantic_lemma(token: str) -> str:
    irregular = _IRREGULAR_LEMMAS.get(token)
    if irregular is not None:
        return irregular
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        root = token[:-3]
        return root[:-1] if len(root) > 2 and root[-1] == root[-2] else root
    if len(token) > 4 and token.endswith("ed"):
        root = token[:-2]
        return root[:-1] if len(root) > 2 and root[-1] == root[-2] else root
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def normalize_semantic_phrase(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"([^\W_]+)[’']s\b", r"\1", normalized)
    normalized = normalized.replace("-", " ").replace("–", " ").replace("—", " ")
    return tuple(semantic_lemma(token) for token in re.findall(r"[^\W_]+", normalized))
