"""Shared local privacy policy for text that may leave the edge device."""

from __future__ import annotations

import re
import unicodedata

EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d\s()./-]{6,}\d)(?!\w)")
URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)

PHRASE_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "by",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)
COUNT_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
COLOR_WORDS = frozenset(
    {
        "amber",
        "black",
        "blue",
        "bronze",
        "brown",
        "copper",
        "crimson",
        "gold",
        "golden",
        "green",
        "grey",
        "indigo",
        "orange",
        "pink",
        "purple",
        "red",
        "silver",
        "teal",
        "violet",
        "white",
        "yellow",
    }
)
VISIBLE_VERBS = frozenset(
    {
        "arc",
        "arcs",
        "carry",
        "carries",
        "circle",
        "circles",
        "climb",
        "climbs",
        "drift",
        "drifts",
        "enter",
        "enters",
        "float",
        "floats",
        "fold",
        "folds",
        "hold",
        "holds",
        "lift",
        "lifts",
        "open",
        "opens",
        "point",
        "points",
        "push",
        "pushes",
        "rise",
        "rises",
        "run",
        "runs",
        "sail",
        "sails",
        "spiral",
        "spirals",
        "swim",
        "swims",
        "tumble",
        "tumbles",
        "unfold",
        "unfolds",
        "wait",
        "waits",
    }
)

_NAME_AFTER_MARKER = re.compile(
    r"\b(?:named|called|mr|mrs|ms|miss|dr|professor)\.?\s+"
    r"([^\W\d_][\w'’\-]*(?:\s+[^\W\d_][\w'’\-]*){0,2})",
    re.IGNORECASE,
)
_CAPITALIZED_WORD = re.compile(r"(?<!\w)[^\W\d_][\w'’\-]*(?!\w)", re.UNICODE)
_PRINTED_SOURCE_MARKER = re.compile(
    r"\b(?P<subject>[^\W\d_][\w'’\-]*)\s+"
    r"(?P<verb>(?:reads?|says?|shows?|displays?|contains?|includes?|features?|has|had|"
    r"carries?|depicts?|bears?)|"
    r"(?:spells?\s+out)|"
    r"(?:(?:is|was)\s+)?"
    r"(?:printed|written|labeled|inscribed|engraved|captioned|titled|marked|painted|"
    r"emblazoned)(?:\s+with)?)"
    r"\b\s*(?:[:：=\-]\s*)?",
    re.IGNORECASE,
)
_PRINTED_NOUN_PAYLOAD = re.compile(
    r"\b(?:the\s+)?(?:words?|text|message|caption|inscription)\s+"
    r"(?P<payload>\S(?:.*\S)?)\s*$",
    re.IGNORECASE,
)
_TRAILING_PRINTED_PREDICATE = re.compile(
    r"\s+\b(?:appear(?:s|ed)?|glow(?:s|ed)?|hang(?:s|ing)?|hung|is|are|was|were)\b"
    r"(?=\s+(?:above|at|beside|in|inside|near|on|over|under)\b|\s*$)",
    re.IGNORECASE,
)
_UNMARKED_CLAUSE_HEAD = re.compile(
    r"(?:\A|[.!?]\s+|\b(?:meanwhile|then|whereas|while)\s+)"
    r"([^\W\d_][\w'’\-]*)",
    re.IGNORECASE | re.UNICODE,
)
_NON_NAME_CAPITALIZED = frozenset(
    {
        "a",
        "after",
        "an",
        "and",
        "as",
        "at",
        "before",
        "beneath",
        "beside",
        "but",
        "each",
        "every",
        "from",
        "he",
        "her",
        "his",
        "i",
        "if",
        "in",
        "inside",
        "it",
        "its",
        "later",
        "meanwhile",
        "no",
        "not",
        "on",
        "once",
        "or",
        "she",
        "suddenly",
        "that",
        "the",
        "their",
        "they",
        "then",
        "this",
        "through",
        "to",
        "toward",
        "towards",
        "under",
        "we",
        "when",
        "while",
        "with",
        "without",
        "you",
    }
)
_UNMARKED_ENTITY_VERBS = VISIBLE_VERBS | {
    "asked",
    "asks",
    "looked",
    "looks",
    "read",
    "reads",
    "said",
    "saw",
    "says",
    "sees",
    "sat",
    "sits",
    "walked",
    "walks",
}
_SAFE_UNMARKED_ENTITY_HEADS = frozenset(
    {
        "adult",
        "alpaca",
        "baby",
        "baker",
        "bird",
        "book",
        "boy",
        "cat",
        "child",
        "dog",
        "dormouse",
        "father",
        "fireflies",
        "firefly",
        "fish",
        "fox",
        "girl",
        "heron",
        "ibis",
        "jellyfish",
        "keeper",
        "king",
        "knight",
        "lynx",
        "monster",
        "moth",
        "mother",
        "otter",
        "owl",
        "pangolin",
        "parent",
        "pilot",
        "prince",
        "queen",
        "rabbit",
        "reader",
        "robot",
        "seal",
        "student",
        "teacher",
        "turtle",
        "unicorn",
        "whale",
        "wizard",
    }
)
_SAFE_UNMARKED_MODIFIERS = COLOR_WORDS | {
    "clockwork",
    "enormous",
    "giant",
    "glowing",
    "large",
    "linen",
    "little",
    "luminous",
    "paper",
    "porcelain",
    "small",
    "striped",
    "tiny",
    "velvet",
    "wooden",
}
_READING_OBJECTS = frozenset(
    {
        "book",
        "card",
        "document",
        "label",
        "letter",
        "magazine",
        "menu",
        "newspaper",
        "note",
        "page",
        "paper",
        "poem",
        "sign",
        "story",
    }
)
_TEXT_BEARING_MEDIA = _READING_OBJECTS | {
    "badge",
    "banner",
    "billboard",
    "blackboard",
    "chalkboard",
    "display",
    "door",
    "envelope",
    "mug",
    "notice",
    "placard",
    "poster",
    "receipt",
    "screen",
    "shirt",
    "tablet",
    "tag",
    "tattoo",
    "wall",
    "whiteboard",
}
_INVERTED_PRINTED_PAYLOAD = re.compile(
    r"(?P<payload>[^.!?;,\r\n]{1,160}?)\s+"
    r"(?:is|are|was|were)\s+"
    r"(?:printed|written|labeled|inscribed|engraved|captioned|titled|marked|painted|"
    r"emblazoned)\s+(?:on|in|across)\s+(?:a|an|the)?\s*"
    r"(?P<medium>[^\W\d_][\w'’\-]*)\b",
    re.IGNORECASE,
)


def privacy_tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"([^\W_]+)[’']s\b", r"\1", normalized)
    normalized = normalized.replace("-", " ").replace("–", " ").replace("—", " ")
    return tuple(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def proper_name_candidates(source_text: str) -> frozenset[tuple[str, ...]]:
    normalized_source = unicodedata.normalize("NFKC", source_text)
    candidates: set[tuple[str, ...]] = set()
    for match in _NAME_AFTER_MARKER.finditer(normalized_source):
        tokens = privacy_tokens(match.group(1))
        verb_index = next(
            (
                index
                for index, token in enumerate(tokens)
                if token in VISIBLE_VERBS or token in _NON_NAME_CAPITALIZED
            ),
            len(tokens),
        )
        name_tokens = tokens[:verb_index]
        if name_tokens:
            candidates.add(name_tokens)
            candidates.update((token,) for token in name_tokens)
    for match in _UNMARKED_CLAUSE_HEAD.finditer(normalized_source):
        word = match.group(1)
        normalized_word = word.casefold()
        if normalized_word in _NON_NAME_CAPITALIZED or normalized_word in COUNT_WORDS:
            continue
        if normalized_word == "nothing":
            continue
        if normalized_word in _SAFE_UNMARKED_ENTITY_HEADS or (
            normalized_word.endswith("s")
            and normalized_word[:-1] in _SAFE_UNMARKED_ENTITY_HEADS
        ):
            continue
        following = privacy_tokens(normalized_source[match.end() :])
        verb_index = next(
            (
                index
                for index, token in enumerate(following[:3])
                if token in _UNMARKED_ENTITY_VERBS
            ),
            None,
        )
        if verb_index is None:
            continue
        candidate = (*privacy_tokens(word), *following[:verb_index])
        head = candidate[-1]
        if (
            head in _SAFE_UNMARKED_ENTITY_HEADS
            or (head.endswith("s") and head[:-1] in _SAFE_UNMARKED_ENTITY_HEADS)
            or candidate[0] in _SAFE_UNMARKED_MODIFIERS
        ):
            continue
        candidates.add(candidate)
        candidates.update((token,) for token in candidate)
    for match in _CAPITALIZED_WORD.finditer(normalized_source):
        word = match.group(0)
        letters = tuple(character for character in word if character.isalpha())
        uncased_script = bool(letters) and not any(
            character.islower() or character.isupper() for character in letters
        )
        if not word[0].isupper() and not uncased_script:
            continue
        if word.casefold() in _NON_NAME_CAPITALIZED or word.casefold() in COUNT_WORDS:
            continue
        if word.casefold() == "nothing" and re.match(
            r"\s+(?:glows|floats|moves|happens|appears)\b",
            normalized_source[match.end() :],
            re.I,
        ):
            continue
        if word.casefold() == "exactly":
            following = privacy_tokens(normalized_source[match.end() :])
            if following and (following[0] in COUNT_WORDS or following[0].isdigit()):
                continue
        candidates.add(privacy_tokens(word))
    return frozenset(candidate for candidate in candidates if candidate)


def printed_source_payload_candidates(source_text: str) -> frozenset[tuple[str, ...]]:
    normalized_source = unicodedata.normalize("NFKC", source_text)
    candidates: set[tuple[str, ...]] = set()
    quote_pairs = {'"': '"', "'": "'", "“": "”", "‘": "’"}
    for marker in _PRINTED_SOURCE_MARKER.finditer(normalized_source):
        remainder = normalized_source[marker.end() :].lstrip()
        if not remainder:
            continue
        quoted_payload = remainder[0] in {'"', "'", "“", "‘"}
        closing_quote = quote_pairs.get(remainder[0])
        if closing_quote is not None:
            remainder = remainder[1:]
            payload = remainder.split(closing_quote, 1)[0]
        else:
            payload = re.split(r"[.!?;\r\n]", remainder, maxsplit=1)[0]
        tokens = privacy_tokens(payload)
        if not tokens:
            continue
        verb_tokens = privacy_tokens(marker.group("verb"))
        subject = marker.group("subject").casefold()
        known_animate = subject in _SAFE_UNMARKED_ENTITY_HEADS or (
            subject.endswith("s") and subject[:-1] in _SAFE_UNMARKED_ENTITY_HEADS
        )
        if verb_tokens and verb_tokens[-1] in {"read", "reads"}:
            if known_animate and subject not in _TEXT_BEARING_MEDIA:
                continue
            if tokens[0] == "aloud":
                continue
            if tokens[0] in {"a", "an", "the"} and _READING_OBJECTS.intersection(tokens[1:5]):
                continue
        if (
            verb_tokens
            and verb_tokens[-1]
            in {
                "carry",
                "carries",
                "bear",
                "bears",
                "contain",
                "contains",
                "depict",
                "depicts",
                "display",
                "displays",
                "feature",
                "features",
                "has",
                "had",
                "include",
                "includes",
                "say",
                "says",
                "show",
                "shows",
            }
            and subject not in _TEXT_BEARING_MEDIA
            and not quoted_payload
        ):
            continue
        candidates.add(tokens)
        candidates.update((token,) for token in tokens if token not in PHRASE_STOPWORDS)

    # Printed payloads are also commonly introduced by a noun phrase rather than a
    # reporting verb: "with the words ...", "on the poster are the words ...", or
    # "the words ... appear on a poster". Treat these as printed text whenever the
    # same sentence names a text-bearing medium.
    for sentence in re.split(r"[.!?;\r\n]+", normalized_source):
        sentence_tokens = privacy_tokens(sentence)
        if not _TEXT_BEARING_MEDIA.intersection(sentence_tokens):
            continue
        for marker in _PRINTED_NOUN_PAYLOAD.finditer(sentence):
            payload = marker.group("payload").strip().strip('"\'“”‘’')
            payload = _TRAILING_PRINTED_PREDICATE.split(payload, maxsplit=1)[0]
            tokens = privacy_tokens(payload)
            if not tokens:
                continue
            candidates.add(tokens)
            candidates.update((token,) for token in tokens if token not in PHRASE_STOPWORDS)

    for marker in _INVERTED_PRINTED_PAYLOAD.finditer(normalized_source):
        medium = marker.group("medium").casefold()
        if medium not in _TEXT_BEARING_MEDIA:
            continue
        payload = marker.group("payload").strip().strip('"\'“”‘’')
        tokens = privacy_tokens(payload)
        if not tokens:
            continue
        candidates.add(tokens)
        candidates.update((token,) for token in tokens if token not in PHRASE_STOPWORDS)
    return frozenset(candidates)


def contains_token_sequence(tokens: tuple[str, ...], candidate: tuple[str, ...]) -> bool:
    width = len(candidate)
    return width > 0 and any(
        tokens[index : index + width] == candidate for index in range(len(tokens) - width + 1)
    )


def distinctive_phrase(tokens: tuple[str, ...]) -> bool:
    content = [token for token in tokens if token not in PHRASE_STOPWORDS]
    return len(content) >= 2 and any(len(token) >= 4 for token in content)


def contains_distinctive_source_phrase(
    output_tokens: tuple[str, ...],
    source_tokens: tuple[str, ...],
) -> bool:
    if len(source_tokens) < 3 or len(output_tokens) < 3:
        return False
    source_phrases = {
        source_tokens[index : index + 3]
        for index in range(len(source_tokens) - 2)
        if distinctive_phrase(source_tokens[index : index + 3])
    }
    return any(
        output_tokens[index : index + 3] in source_phrases
        for index in range(len(output_tokens) - 2)
    )
