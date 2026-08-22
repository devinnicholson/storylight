from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

SessionId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
]
PageId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
Word = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
Transcript = Annotated[str, StringConstraints(max_length=8_000)]
ReaderEventType = Literal["transcript.partial", "word.reached", "session.reset"]


class StrictEventModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReaderEventPublishRequest(StrictEventModel):
    """A deliberately small event surface for the local reader simulator."""

    type: Literal["transcript.partial", "word.reached"]
    page_id: PageId | None = None
    index: Annotated[int, Field(ge=0, le=100_000)] | None = None
    word: Word | None = None
    transcript: Transcript | None = None
    generation: Annotated[int, Field(ge=1)] | None = None

    @model_validator(mode="after")
    def validate_payload_for_type(self) -> ReaderEventPublishRequest:
        if self.type == "transcript.partial":
            if self.transcript is None or self.page_id is None or self.generation is None:
                raise ValueError(
                    "transcript.partial requires transcript, page_id, and generation"
                )
            if self.index is not None or self.word is not None:
                raise ValueError("transcript.partial does not accept index or word")
        elif (
            self.page_id is None
            or self.index is None
            or self.word is None
            or self.generation is None
        ):
            raise ValueError("word.reached requires page_id, index, word, and generation")
        elif self.transcript is not None:
            raise ValueError("word.reached does not accept transcript")
        return self

    def event_payload(self) -> dict[str, str | int]:
        if self.type == "transcript.partial":
            return {
                "transcript": self.transcript or "",
                "page_id": self.page_id or "",
                "generation": self.generation or 0,
            }
        return {
            "page_id": self.page_id or "",
            "index": self.index if self.index is not None else 0,
            "word": self.word or "",
            "generation": self.generation or 0,
        }


class ReaderEvent(StrictEventModel):
    session_id: SessionId
    sequence: Annotated[int, Field(ge=1)]
    published_at: datetime
    type: ReaderEventType
    payload: dict[str, Any]
    dropped_before_sequence: Annotated[int, Field(ge=1)] | None = None


class PublishResult(StrictEventModel):
    event: ReaderEvent
    subscriber_count: Annotated[int, Field(ge=0)]


class EventHubClosedError(RuntimeError):
    pass


_CLOSED = object()


@dataclass(eq=False, slots=True)
class ReaderEventSubscription:
    _hub: ReaderEventHub
    session_id: str
    _queue: asyncio.Queue[ReaderEvent | object]
    _closed: bool = False

    async def receive(self) -> ReaderEvent:
        item = await self._queue.get()
        if item is _CLOSED:
            raise EventHubClosedError("Reader event hub is closed")
        assert isinstance(item, ReaderEvent)
        return item

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._hub.unsubscribe(self)

    async def __aenter__(self) -> ReaderEventSubscription:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


class ReaderEventHub:
    """Process-local, bounded fan-out for live reader session events."""

    def __init__(self, queue_size: int = 32) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be at least 1")
        self._queue_size = queue_size
        self._subscribers: dict[str, set[ReaderEventSubscription]] = {}
        self._sequences: dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    async def subscribe(self, session_id: SessionId) -> ReaderEventSubscription:
        queue: asyncio.Queue[ReaderEvent | object] = asyncio.Queue(self._queue_size)
        subscription = ReaderEventSubscription(self, session_id, queue)
        async with self._lock:
            if self._closed:
                raise EventHubClosedError("Reader event hub is closed")
            self._subscribers.setdefault(session_id, set()).add(subscription)
        return subscription

    async def unsubscribe(self, subscription: ReaderEventSubscription) -> None:
        async with self._lock:
            subscribers = self._subscribers.get(subscription.session_id)
            if subscribers is None:
                return
            subscribers.discard(subscription)
            if not subscribers:
                self._subscribers.pop(subscription.session_id, None)
                self._sequences.pop(subscription.session_id, None)

    async def publish(
        self,
        session_id: SessionId,
        event_type: ReaderEventType,
        payload: Mapping[str, Any],
    ) -> PublishResult:
        async with self._lock:
            if self._closed:
                raise EventHubClosedError("Reader event hub is closed")

            subscribers = tuple(self._subscribers.get(session_id, ()))
            sequence = self._sequences.get(session_id, 0) + 1
            event = ReaderEvent(
                session_id=session_id,
                sequence=sequence,
                published_at=datetime.now(UTC),
                type=event_type,
                payload=dict(payload),
            )

            if subscribers:
                self._sequences[session_id] = sequence
            for subscription in subscribers:
                queue = subscription._queue
                queued_event = event
                if queue.full():
                    dropped = queue.get_nowait()
                    assert isinstance(dropped, ReaderEvent)
                    queued_event = event.model_copy(
                        update={
                            "dropped_before_sequence": (
                                dropped.dropped_before_sequence or dropped.sequence
                            )
                        }
                    )
                queue.put_nowait(queued_event)

        return PublishResult(event=event, subscriber_count=len(subscribers))

    async def publish_request(
        self,
        session_id: SessionId,
        request: ReaderEventPublishRequest,
    ) -> PublishResult:
        return await self.publish(session_id, request.type, request.event_payload())

    async def publish_word_reached(
        self,
        session_id: SessionId,
        *,
        page_id: PageId,
        index: int,
        word: Word,
        generation: int,
    ) -> PublishResult:
        request = ReaderEventPublishRequest(
            type="word.reached",
            page_id=page_id,
            index=index,
            word=word,
            generation=generation,
        )
        return await self.publish_request(session_id, request)

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            subscriptions = [
                subscription
                for subscribers in self._subscribers.values()
                for subscription in subscribers
            ]
            self._subscribers.clear()
            self._sequences.clear()

            for subscription in subscriptions:
                subscription._closed = True
                queue = subscription._queue
                if queue.full():
                    queue.get_nowait()
                queue.put_nowait(_CLOSED)
