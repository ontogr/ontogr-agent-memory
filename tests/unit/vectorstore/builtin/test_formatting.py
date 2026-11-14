"""Exact input formatting for the reviewed FastEmbed model profiles."""

import pytest

from ontogr_mem.vectorstore.builtin.formatting import (
    ENCODING_POLICIES,
    SUPPORTED_MODELS,
    EmbeddingPurpose,
    format_text,
)


class TestFormatText:
    """Formatting rules for supported native FastEmbed models."""

    @pytest.mark.parametrize(
        ("model_name", "purpose", "text", "expected"),
        [
            (
                "ibm-granite/granite-embedding-small-english-r2",
                "query",
                "  query: user text  ",
                "  query: user text  ",
            ),
            (
                "ibm-granite/granite-embedding-small-english-r2",
                "document",
                "  query: user text  ",
                "  query: user text  ",
            ),
            (
                "sentence-transformers/all-MiniLM-L6-v2",
                "query",
                "  query: user text  ",
                "  query: user text  ",
            ),
            (
                "sentence-transformers/all-MiniLM-L6-v2",
                "document",
                "  query: user text  ",
                "  query: user text  ",
            ),
            (
                "jinaai/jina-embeddings-v3",
                "query",
                "  query: user text  ",
                "  query: user text  ",
            ),
            (
                "jinaai/jina-embeddings-v3",
                "document",
                "  query: user text  ",
                "  query: user text  ",
            ),
            (
                "nomic-ai/nomic-embed-text-v1.5",
                "query",
                "  query: user text  ",
                "search_query:   query: user text  ",
            ),
            (
                "nomic-ai/nomic-embed-text-v1.5",
                "document",
                "  query: user text  ",
                "search_document:   query: user text  ",
            ),
            (
                "nomic-ai/nomic-embed-text-v1.5-Q",
                "query",
                "  query: user text  ",
                "search_query:   query: user text  ",
            ),
            (
                "nomic-ai/nomic-embed-text-v1.5-Q",
                "document",
                "  query: user text  ",
                "search_document:   query: user text  ",
            ),
            (
                "intfloat/multilingual-e5-large",
                "query",
                "  query: user text  ",
                "query:   query: user text  ",
            ),
            (
                "intfloat/multilingual-e5-large",
                "document",
                "  query: user text  ",
                "passage:   query: user text  ",
            ),
            (
                "Qwen/Qwen3-Embedding-0.6B",
                "query",
                "  query: user text  ",
                "Instruct: Given a web search query, retrieve relevant passages that "
                "answer the query\nQuery:  query: user text  ",
            ),
            (
                "Qwen/Qwen3-Embedding-0.6B",
                "document",
                "  query: user text  ",
                "  query: user text  ",
            ),
            (
                "Qwen/Qwen3-Embedding-0.6B-Q",
                "query",
                "  query: user text  ",
                "Instruct: Given a web search query, retrieve relevant passages that "
                "answer the query\nQuery:  query: user text  ",
            ),
            (
                "Qwen/Qwen3-Embedding-0.6B-Q",
                "document",
                "  query: user text  ",
                "  query: user text  ",
            ),
            (
                "google/embeddinggemma-300m",
                "query",
                "  query: user text  ",
                "task: search result | query:   query: user text  ",
            ),
            (
                "google/embeddinggemma-300m",
                "document",
                "  query: user text  ",
                "title: none | text:   query: user text  ",
            ),
            (
                "minishlab/potion-base-8M",
                "query",
                "  query: user text  ",
                "  query: user text  ",
            ),
            (
                "minishlab/potion-base-8M",
                "document",
                "  query: user text  ",
                "  query: user text  ",
            ),
            (
                "minishlab/potion-retrieval-32M",
                "query",
                "  query: user text  ",
                "  query: user text  ",
            ),
            (
                "minishlab/potion-retrieval-32M",
                "document",
                "  query: user text  ",
                "  query: user text  ",
            ),
        ],
    )
    def test_formats_each_supported_model_for_both_purposes(
        self, model_name: str, purpose: EmbeddingPurpose, text: str, expected: str
    ) -> None:
        """Apply exactly the reviewed prefix while preserving the raw input."""
        assert format_text(model_name, text, purpose=purpose) == expected

    @pytest.mark.parametrize(
        ("model_name", "purpose", "text", "expected"),
        [
            (
                "GOOGLE/EMBEDDINGGEMMA-300M",
                "query",
                "question",
                "task: search result | query: question",
            ),
            (
                "NOMIC-AI/NOMIC-EMBED-TEXT-V1.5-q",
                "document",
                "passage",
                "search_document: passage",
            ),
        ],
    )
    def test_resolves_canonical_names_without_case_sensitivity(
        self, model_name: str, purpose: EmbeddingPurpose, text: str, expected: str
    ) -> None:
        """Name casing does not change a supported model's formatting policy."""
        assert format_text(model_name, text, purpose=purpose) == expected

    def test_lists_canonical_names_and_versioned_native_task_policy(self) -> None:
        """Profiles expose canonical names and Jina's purpose-specific task IDs."""
        assert "Qwen/Qwen3-Embedding-0.6B-Q" in SUPPORTED_MODELS
        jina = ENCODING_POLICIES["jinaai/jina-embeddings-v3"]
        assert jina.version == 1
        assert jina.query_task == "retrieval.query"
        assert jina.document_task == "retrieval.passage"

    @pytest.mark.parametrize(
        "model_name",
        ["Qwen/Qwen3-Embedding-8B", "sentence-transformers/unknown-model"],
    )
    def test_refuses_models_without_a_reviewed_profile(self, model_name: str) -> None:
        """Unknown native names fail instead of inheriting a family policy."""
        with pytest.raises(ValueError, match=model_name):
            format_text(model_name, "text", purpose="query")
