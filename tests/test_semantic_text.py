from storylight.semantic_text import normalize_semantic_phrase, semantic_lemma


def test_shared_semantic_normalization_preserves_words_ending_in_double_s() -> None:
    assert semantic_lemma("glass") == "glass"
    assert normalize_semantic_phrase("glass branches") == ("glass", "branch")


def test_shared_semantic_normalization_handles_graph_inflections() -> None:
    assert normalize_semantic_phrase("The fox's carried boxes") == (
        "the",
        "fox",
        "carry",
        "box",
    )
