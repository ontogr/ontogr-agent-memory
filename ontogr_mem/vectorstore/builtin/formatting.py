"""Reviewed FastEmbed input formats and purpose-specific encoding metadata."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ontogr_mem.vectorstore import EmbeddingPurpose


@dataclass(frozen=True, slots=True)
class EncodingPolicy:
    """Versioned text prefixes and native task labels for one model."""

    version: int
    query_prefix: str = ""
    document_prefix: str = ""
    query_task: str | None = None
    document_task: str | None = None


ENCODING_POLICIES: Mapping[str, EncodingPolicy] = MappingProxyType(
    {
        "ibm-granite/granite-embedding-small-english-r2": EncodingPolicy(1),
        "sentence-transformers/all-MiniLM-L6-v2": EncodingPolicy(1),
        "jinaai/jina-embeddings-v3": EncodingPolicy(
            1,
            query_task="retrieval.query",
            document_task="retrieval.passage",
        ),
        "nomic-ai/nomic-embed-text-v1.5": EncodingPolicy(
            1,
            query_prefix="search_query: ",
            document_prefix="search_document: ",
        ),
        "nomic-ai/nomic-embed-text-v1.5-Q": EncodingPolicy(
            1,
            query_prefix="search_query: ",
            document_prefix="search_document: ",
        ),
        "intfloat/multilingual-e5-large": EncodingPolicy(
            1,
            query_prefix="query: ",
            document_prefix="passage: ",
        ),
        "Qwen/Qwen3-Embedding-0.6B": EncodingPolicy(
            1,
            query_prefix=(
                "Instruct: Given a web search query, retrieve relevant passages that "
                "answer the query\nQuery:"
            ),
        ),
        "Qwen/Qwen3-Embedding-0.6B-Q": EncodingPolicy(
            1,
            query_prefix=(
                "Instruct: Given a web search query, retrieve relevant passages that "
                "answer the query\nQuery:"
            ),
        ),
        "google/embeddinggemma-300m": EncodingPolicy(
            1,
            query_prefix="task: search result | query: ",
            document_prefix="title: none | text: ",
        ),
        "minishlab/potion-base-8M": EncodingPolicy(1),
        "minishlab/potion-retrieval-32M": EncodingPolicy(1),
    }
)
SUPPORTED_MODELS: tuple[str, ...] = tuple(ENCODING_POLICIES)
_POLICIES_BY_CASEFOLD = MappingProxyType(
    {model_name.casefold(): policy for model_name, policy in ENCODING_POLICIES.items()}
)


def format_text(model_name: str, text: str, *, purpose: EmbeddingPurpose) -> str:
    """Format one raw input with the reviewed prefix for its model and purpose.

    Args:
        model_name: A canonical FastEmbed model name, matched without case sensitivity.
        text: Raw user text, which is preserved without trimming or prefix detection.
        purpose: Whether the text is a query or a stored document.

    Raises:
        ValueError: If the model has no reviewed profile or the purpose is invalid.
    """
    policy = _POLICIES_BY_CASEFOLD.get(model_name.casefold())
    if policy is None:
        raise ValueError(f"Unsupported FastEmbed formatting model: {model_name!r}")

    if purpose == "query":
        return policy.query_prefix + text
    if purpose == "document":
        return policy.document_prefix + text
    raise ValueError("purpose must be 'query' or 'document'")
