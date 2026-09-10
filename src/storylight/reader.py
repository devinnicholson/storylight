"""Deterministic alignment of live speech transcripts to trusted page text."""

from __future__ import annotations

import re
import time
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def tokenize(text: str) -> tuple[str, ...]:
    """Return case-folded word tokens with surrounding punctuation removed.

    Apostrophes inside a word are retained and normalized to ASCII. Hyphens and
    other punctuation delimit words. NFKC normalization makes visually equivalent
    Unicode forms compare consistently.
    """

    normalized = unicodedata.normalize("NFKC", text).replace("’", "'").casefold()
    return tuple(match.group(0) for match in _WORD_PATTERN.finditer(normalized))


class TranscriptMode(StrEnum):
    """How a transcript update relates to earlier updates."""

    CUMULATIVE = "cumulative"
    PARTIAL = "partial"


class ReaderState(StrEnum):
    READY = "ready"
    READING = "reading"
    PAGE_COMPLETE = "page_complete"


@dataclass(frozen=True, slots=True)
class WordReachedEvent:
    """Immutable notification that speech reached a trusted page word."""

    session_id: str
    page_id: str
    index: int
    word: str
    transcript_started_at_ms: int
    transcript_ended_at_ms: int
    reached_at_ms: int
    type: Literal["word.reached"] = field(default="word.reached", init=False)


class ReaderAligner:
    """Monotonically align ASR text to the words on a page.

    Cumulative mode aligns the entire current ASR hypothesis against the entire
    page. Partial mode treats each call as a new speech chunk and aligns it only
    after the last reached word. Partial mode is therefore the correct choice for
    distinct chunks and for consecutive identical words; that distinction cannot
    be inferred from text alone.
    """

    def __init__(
        self,
        *,
        session_id: str,
        page_id: str,
        page_text: str,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._clock_ms = clock_ms or _monotonic_ms
        self.session_id = _nonempty("session_id", session_id)
        self.page_id = _nonempty("page_id", page_id)
        self.page_text = page_text
        self.words = _page_tokens(page_text)
        self._last_reached_index = -1
        self._state = ReaderState.READY

    @property
    def state(self) -> ReaderState:
        return self._state

    @property
    def page_complete(self) -> bool:
        return self._state is ReaderState.PAGE_COMPLETE

    @property
    def last_reached_index(self) -> int | None:
        if self._last_reached_index < 0:
            return None
        return self._last_reached_index

    @property
    def next_word(self) -> str | None:
        next_index = self._last_reached_index + 1
        if next_index >= len(self.words):
            return None
        return self.words[next_index]

    def ingest(
        self,
        transcript: str,
        *,
        mode: TranscriptMode | str = TranscriptMode.CUMULATIVE,
        started_at_ms: int | None = None,
        ended_at_ms: int | None = None,
    ) -> tuple[WordReachedEvent, ...]:
        """Align one transcript update and return newly reached page words."""

        if self.page_complete:
            return ()

        transcript_words = tokenize(transcript)
        if not transcript_words:
            return ()

        try:
            transcript_mode = TranscriptMode(mode)
        except ValueError as error:
            allowed = ", ".join(item.value for item in TranscriptMode)
            raise ValueError(f"mode must be one of: {allowed}") from error

        started, ended = self._timestamps(started_at_ms, ended_at_ms)
        if transcript_mode is TranscriptMode.CUMULATIVE:
            matches = _lcs_matches(self.words, transcript_words)
        else:
            offset = self._last_reached_index + 1
            relative_matches = _lcs_matches(self.words[offset:], transcript_words)
            matches = tuple((match[0] + offset, match[1]) for match in relative_matches)

        new_matches = tuple(
            (page_index, spoken_index)
            for page_index, spoken_index in matches
            if page_index > self._last_reached_index
        )
        if not new_matches:
            return ()

        events = tuple(
            WordReachedEvent(
                session_id=self.session_id,
                page_id=self.page_id,
                index=page_index,
                word=self.words[page_index],
                transcript_started_at_ms=started,
                transcript_ended_at_ms=ended,
                reached_at_ms=_interpolate_timestamp(
                    started,
                    ended,
                    spoken_index,
                    len(transcript_words),
                ),
            )
            for page_index, spoken_index in new_matches
        )
        self._last_reached_index = new_matches[-1][0]
        self._state = (
            ReaderState.PAGE_COMPLETE
            if self._last_reached_index == len(self.words) - 1
            else ReaderState.READING
        )
        return events

    def reset(
        self,
        *,
        session_id: str | None = None,
        page_id: str | None = None,
        page_text: str | None = None,
    ) -> None:
        """Clear progress, optionally replacing the session or trusted page."""

        if session_id is not None:
            self.session_id = _nonempty("session_id", session_id)
        if page_id is not None:
            self.page_id = _nonempty("page_id", page_id)
        if page_text is not None:
            words = _page_tokens(page_text)
            self.page_text = page_text
            self.words = words
        self._last_reached_index = -1
        self._state = ReaderState.READY

    def _timestamps(
        self,
        started_at_ms: int | None,
        ended_at_ms: int | None,
    ) -> tuple[int, int]:
        if started_at_ms is None and ended_at_ms is None:
            observed_at = self._clock_ms()
            return observed_at, observed_at
        if started_at_ms is None:
            started_at_ms = ended_at_ms
        if ended_at_ms is None:
            ended_at_ms = started_at_ms
        assert started_at_ms is not None
        assert ended_at_ms is not None
        if started_at_ms < 0 or ended_at_ms < 0:
            raise ValueError("timestamps must be non-negative")
        if ended_at_ms < started_at_ms:
            raise ValueError("ended_at_ms must be greater than or equal to started_at_ms")
        return started_at_ms, ended_at_ms


def _page_tokens(page_text: str) -> tuple[str, ...]:
    words = tokenize(page_text)
    if not words:
        raise ValueError("page_text must contain at least one word")
    return words


def _nonempty(field_name: str, value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be empty")
    return stripped


def _monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def _interpolate_timestamp(
    started_at_ms: int,
    ended_at_ms: int,
    spoken_index: int,
    spoken_count: int,
) -> int:
    duration = ended_at_ms - started_at_ms
    return started_at_ms + round(duration * (spoken_index + 1) / spoken_count)


def _lcs_matches(
    trusted_words: Sequence[str],
    spoken_words: Sequence[str],
) -> tuple[tuple[int, int], ...]:
    """Return a deterministic earliest-page longest common subsequence."""

    trusted_count = len(trusted_words)
    spoken_count = len(spoken_words)
    lengths = [[0] * (spoken_count + 1) for _ in range(trusted_count + 1)]

    for trusted_index in range(trusted_count - 1, -1, -1):
        for spoken_index in range(spoken_count - 1, -1, -1):
            if trusted_words[trusted_index] == spoken_words[spoken_index]:
                lengths[trusted_index][spoken_index] = (
                    lengths[trusted_index + 1][spoken_index + 1] + 1
                )
            else:
                lengths[trusted_index][spoken_index] = max(
                    lengths[trusted_index + 1][spoken_index],
                    lengths[trusted_index][spoken_index + 1],
                )

    matches: list[tuple[int, int]] = []
    trusted_index = 0
    spoken_index = 0
    while trusted_index < trusted_count and spoken_index < spoken_count:
        if trusted_words[trusted_index] == spoken_words[spoken_index]:
            matches.append((trusted_index, spoken_index))
            trusted_index += 1
            spoken_index += 1
            continue

        skip_trusted = lengths[trusted_index + 1][spoken_index]
        skip_spoken = lengths[trusted_index][spoken_index + 1]
        if skip_trusted > skip_spoken:
            trusted_index += 1
        else:
            # Preserve the earliest possible trusted-page position on ties.
            spoken_index += 1

    return tuple(matches)
