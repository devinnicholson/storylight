import asyncio

import pytest

from bookforge.reader import ReaderState
from bookforge.reader_runtime import (
    ReaderSessionConfigureRequest,
    ReaderSessionNotConfiguredError,
    ReaderSessionPageMismatchError,
    ReaderSessionRegistry,
    TranscriptUpdateRequest,
)


def test_configured_session_ingests_cumulative_transcripts() -> None:
    async def exercise() -> None:
        registry = ReaderSessionRegistry()
        configured = await registry.configure(
            "moon-gate",
            ReaderSessionConfigureRequest(
                page_id="page-01",
                page_text="The small moth went through the red gate.",
            ),
        )
        first = await registry.ingest(
            "moon-gate",
            TranscriptUpdateRequest(
                source="typed", text="The small moth", generation=configured.generation
            ),
        )
        final = await registry.ingest(
            "moon-gate",
            TranscriptUpdateRequest(
                source="asr",
                text="The small moth went through the red gate",
                is_final=True,
                generation=configured.generation,
            ),
        )

        assert configured.state is ReaderState.READY
        assert [event.index for event in first.word_events] == [0, 1, 2]
        assert [event.index for event in final.word_events] == [3, 4, 5, 6, 7]
        assert final.status.state is ReaderState.PAGE_COMPLETE

    asyncio.run(exercise())


def test_session_requires_configuration_and_matching_page() -> None:
    async def exercise() -> None:
        registry = ReaderSessionRegistry()
        with pytest.raises(ReaderSessionNotConfiguredError):
            await registry.ingest(
                "missing",
                TranscriptUpdateRequest(source="typed", text="hello", generation=1),
            )

        configured = await registry.configure(
            "configured",
            ReaderSessionConfigureRequest(page_id="page-01", page_text="Moon gate"),
        )
        with pytest.raises(ReaderSessionPageMismatchError, match="page-01"):
            await registry.ingest(
                "configured",
                TranscriptUpdateRequest(
                    source="typed",
                    text="Moon",
                    page_id="page-02",
                    generation=configured.generation,
                ),
            )

    asyncio.run(exercise())


def test_reconfiguring_identical_page_preserves_progress_but_new_text_resets() -> None:
    async def exercise() -> None:
        registry = ReaderSessionRegistry()
        request = ReaderSessionConfigureRequest(page_id="page-01", page_text="Moon gate")
        configured = await registry.configure("session", request)
        await registry.ingest(
            "session",
            TranscriptUpdateRequest(source="typed", text="Moon", generation=configured.generation),
        )

        preserved = await registry.configure("session", request)
        replaced = await registry.configure(
            "session",
            ReaderSessionConfigureRequest(page_id="page-02", page_text="Red moth"),
        )

        assert preserved.last_reached_index == 0
        assert replaced.last_reached_index is None
        assert replaced.next_word == "red"

    asyncio.run(exercise())
