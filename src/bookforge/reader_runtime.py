"""Configured live-reader sessions shared by typed and ASR transcript sources."""

from __future__ import annotations

import asyncio
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from bookforge.event_hub import PageId, SessionId
from bookforge.reader import ReaderAligner, ReaderState, TranscriptMode, WordReachedEvent, tokenize

PageText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=8_000),
]
TranscriptText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=8_000),
]


class StrictReaderModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReaderSessionConfigureRequest(StrictReaderModel):
    page_id: PageId
    page_text: PageText

    @field_validator("page_text")
    @classmethod
    def require_words(cls, value: str) -> str:
        if not tokenize(value):
            raise ValueError("page_text must contain at least one word")
        return value


class ReaderSessionStatus(StrictReaderModel):
    session_id: SessionId
    page_id: PageId
    page_text: PageText
    generation: Annotated[int, Field(ge=1)]
    state: ReaderState
    word_count: Annotated[int, Field(ge=1)]
    last_reached_index: int | None
    next_word: str | None


class TranscriptUpdateRequest(StrictReaderModel):
    source: Literal["typed", "asr"]
    text: TranscriptText
    page_id: PageId | None = None
    language: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=2, max_length=35),
    ] = "en"
    is_final: bool = False
    generation: Annotated[int, Field(ge=1)]
    mode: TranscriptMode = TranscriptMode.CUMULATIVE
    started_at_ms: Annotated[int, Field(ge=0)] | None = None
    ended_at_ms: Annotated[int, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def validate_timestamps(self) -> TranscriptUpdateRequest:
        if (
            self.started_at_ms is not None
            and self.ended_at_ms is not None
            and self.ended_at_ms < self.started_at_ms
        ):
            raise ValueError("ended_at_ms must be greater than or equal to started_at_ms")
        return self


class TranscriptUpdateResult(StrictReaderModel):
    status: ReaderSessionStatus
    source: Literal["typed", "asr"]
    language: str
    is_final: bool
    word_events: list[WordReachedEvent]


class ReaderSessionNotConfiguredError(LookupError):
    pass


class ReaderSessionPageMismatchError(ValueError):
    pass


class ReaderSessionGenerationMismatchError(ValueError):
    pass


class ReaderSessionRegistry:
    """Process-local aligner registry with serialized updates per application."""

    def __init__(self) -> None:
        self._aligners: dict[str, ReaderAligner] = {}
        self._generations: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def configure(
        self,
        session_id: str,
        request: ReaderSessionConfigureRequest,
    ) -> ReaderSessionStatus:
        async with self._lock:
            aligner = self._aligners.get(session_id)
            if (
                aligner is None
                or aligner.page_id != request.page_id
                or aligner.page_text != request.page_text
            ):
                aligner = ReaderAligner(
                    session_id=session_id,
                    page_id=request.page_id,
                    page_text=request.page_text,
                )
                self._aligners[session_id] = aligner
                self._generations[session_id] = self._generations.get(session_id, 0) + 1
            return self._status(aligner, self._generations[session_id])

    async def ingest(
        self,
        session_id: str,
        request: TranscriptUpdateRequest,
    ) -> TranscriptUpdateResult:
        async with self._lock:
            aligner = self._aligners.get(session_id)
            if aligner is None:
                raise ReaderSessionNotConfiguredError(
                    f"Reader session {session_id!r} must be configured with trusted page text"
                )
            if request.page_id is not None and request.page_id != aligner.page_id:
                raise ReaderSessionPageMismatchError(
                    f"Reader session is configured for page {aligner.page_id!r}, "
                    f"not {request.page_id!r}"
                )
            generation = self._generations[session_id]
            if request.generation != generation:
                raise ReaderSessionGenerationMismatchError(
                    f"Reader session generation is {generation}, not {request.generation}"
                )
            events = list(
                aligner.ingest(
                    request.text,
                    mode=request.mode,
                    started_at_ms=request.started_at_ms,
                    ended_at_ms=request.ended_at_ms,
                )
            )
            return TranscriptUpdateResult(
                status=self._status(aligner, generation),
                source=request.source,
                language=request.language,
                is_final=request.is_final,
                word_events=events,
            )

    async def reset(self, session_id: str) -> ReaderSessionStatus:
        async with self._lock:
            aligner = self._aligners.get(session_id)
            if aligner is None:
                raise ReaderSessionNotConfiguredError(
                    f"Reader session {session_id!r} is not configured"
                )
            aligner.reset()
            self._generations[session_id] += 1
            return self._status(aligner, self._generations[session_id])

    async def status(self, session_id: str) -> ReaderSessionStatus:
        async with self._lock:
            aligner = self._aligners.get(session_id)
            if aligner is None:
                raise ReaderSessionNotConfiguredError(
                    f"Reader session {session_id!r} is not configured"
                )
            return self._status(aligner, self._generations[session_id])

    @staticmethod
    def _status(aligner: ReaderAligner, generation: int) -> ReaderSessionStatus:
        return ReaderSessionStatus(
            session_id=aligner.session_id,
            page_id=aligner.page_id,
            page_text=aligner.page_text,
            generation=generation,
            state=aligner.state,
            word_count=len(aligner.words),
            last_reached_index=aligner.last_reached_index,
            next_word=aligner.next_word,
        )
