

from storylight.reader import ReaderAligner, ReaderState, TranscriptMode, tokenize

MOON_GATE_TEXT = "The small moth went through the red gate."


def event_indexes(events: tuple) -> list[int]:
    return [event.index for event in events]


def test_tokenize_normalizes_case_unicode_apostrophes_and_punctuation() -> None:
    assert tokenize("“HELLO,” said Moon-Gate. DON’T stop_2!") == (
        "hello",
        "said",
        "moon",
        "gate",
        "don't",
        "stop",
        "2",
    )


def test_moon_gate_cumulative_transcripts_emit_only_new_words() -> None:
    reader = ReaderAligner(
        session_id="moon-session",
        page_id="page-01",
        page_text=MOON_GATE_TEXT,
    )

    first = reader.ingest("THE small,", started_at_ms=0, ended_at_ms=200)
    second = reader.ingest("The small moth went", started_at_ms=0, ended_at_ms=400)
    duplicate = reader.ingest("The small moth went", started_at_ms=0, ended_at_ms=400)
    final = reader.ingest(
        "The small moth went through the red gate!",
        started_at_ms=0,
        ended_at_ms=800,
    )

    assert event_indexes(first) == [0, 1]
    assert [event.word for event in first] == ["the", "small"]
    assert [event.reached_at_ms for event in first] == [100, 200]
    assert event_indexes(second) == [2, 3]
    assert duplicate == ()
    assert event_indexes(final) == [4, 5, 6, 7]
    assert reader.state is ReaderState.PAGE_COMPLETE
    assert reader.page_complete is True
    assert reader.next_word is None


def test_partial_chunks_progress_across_consecutive_repeated_words() -> None:
    reader = ReaderAligner(session_id="s1", page_id="p1", page_text="Go, go, go!")

    events = [reader.ingest("go", mode=TranscriptMode.PARTIAL)[0] for _ in range(3)]

    assert [event.index for event in events] == [0, 1, 2]
    assert reader.page_complete


def test_cumulative_alignment_uses_correct_occurrences_of_repeated_words() -> None:
    reader = ReaderAligner(
        session_id="s1",
        page_id="p1",
        page_text="The cat and the cat sat.",
    )

    first = reader.ingest("the cat")
    second = reader.ingest("the cat and the cat")
    final = reader.ingest("the cat and the cat sat")

    assert event_indexes(first) == [0, 1]
    assert event_indexes(second) == [2, 3, 4]
    assert event_indexes(final) == [5]
    assert reader.page_complete


def test_skipped_words_can_advance_but_late_corrections_never_move_backward() -> None:
    reader = ReaderAligner(
        session_id="s1",
        page_id="p1",
        page_text="The very small moth flew.",
    )

    skipped = reader.ingest("the moth", mode="partial")
    late_correction = reader.ingest("very small", mode="partial")
    final = reader.ingest("flew", mode="partial")

    assert event_indexes(skipped) == [0, 3]
    assert late_correction == ()
    assert event_indexes(final) == [4]
    assert reader.page_complete


def test_cumulative_hypothesis_correction_emits_corrected_word_once() -> None:
    reader = ReaderAligner(session_id="s1", page_id="p1", page_text="The small moth went")

    mistaken = reader.ingest("the small mut")
    corrected = reader.ingest("the small moth")
    retracted = reader.ingest("the small")
    extended = reader.ingest("the small moth went")

    assert event_indexes(mistaken) == [0, 1]
    assert event_indexes(corrected) == [2]
    assert retracted == ()
    assert event_indexes(extended) == [3]


def test_complete_page_ignores_later_transcripts_until_reset() -> None:
    reader = ReaderAligner(session_id="s1", page_id="p1", page_text="Moon gate")
    reader.ingest("moon gate")

    assert reader.ingest("moon gate moon gate", mode="partial") == ()
    assert reader.last_reached_index == 1


def test_reset_can_reread_or_replace_page_and_session() -> None:
    reader = ReaderAligner(session_id="s1", page_id="p1", page_text="Moon gate")
    reader.ingest("moon gate")

    reader.reset()
    assert reader.state is ReaderState.READY
    assert reader.last_reached_index is None
    assert event_indexes(reader.ingest("moon")) == [0]

    reader.reset(session_id="s2", page_id="p2", page_text="Red moth")
    event = reader.ingest("red", started_at_ms=50, ended_at_ms=50)[0]
    assert reader.words == ("red", "moth")
    assert event.session_id == "s2"
    assert event.page_id == "p2"
    assert event.index == 0
