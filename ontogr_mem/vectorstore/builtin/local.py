"""Local FastEmbed embeddings with purpose-aware formatting."""

import asyncio
import hashlib
import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import TYPE_CHECKING

from ontogr_mem._extras import import_extra
from ontogr_mem.vectorstore.builtin.formatting import ENCODING_POLICIES, format_text
from ontogr_mem.vectorstore.builtin.settings import EmbeddingSettings
from ontogr_mem.vectorstore.embedder import EmbeddingPurpose
from ontogr_mem.vectorstore.errors import (
    EmbedderUnavailableError,
    EmbeddingError,
    PermanentEmbeddingError,
)
from ontogr_mem.vectorstore.vecmath import to_unit


if TYPE_CHECKING:
    from fastembed import TextEmbedding


_DEFAULT_MODEL_NAME = "ibm-granite/granite-embedding-small-english-r2"
_SUPPORTED_PROVIDERS = frozenset({"CPUExecutionProvider", "CUDAExecutionProvider"})
_NON_SEMANTIC_OPTIONS = frozenset(
    {
        "model_name",
        "cache_dir",
        "threads",
        "providers",
        "cuda",
        "device_id",
        "device_ids",
        "lazy_load",
        "local_files_only",
        "batch_size",
    }
)


def _normalize_provider_options(options: dict[str, object]) -> list[str]:
    """Convert JSON provider pairs and reject unsupported provider ordering."""
    provider_option = options.get("providers")
    if provider_option is None:
        return []
    if not isinstance(provider_option, list) or not provider_option:
        raise EmbedderUnavailableError(
            "FastEmbed providers must be a non-empty list",
            hint=(
                "Use CPUExecutionProvider or CUDAExecutionProvider, optionally "
                "paired with native provider options."
            ),
        )

    provider_names: list[str] = []
    native_providers: list[object] = []
    for provider in provider_option:
        if isinstance(provider, str):
            provider_name = provider
            native_provider = provider
        elif (
            isinstance(provider, list)
            and len(provider) == 2
            and isinstance(provider[0], str)
            and isinstance(provider[1], dict)
        ):
            provider_name = provider[0]
            native_provider = (provider_name, provider[1])
        else:
            raise EmbedderUnavailableError(
                "FastEmbed providers has an invalid entry",
                hint=(
                    "Use provider names or [name, options] pairs for "
                    "CPUExecutionProvider and CUDAExecutionProvider."
                ),
            )
        if provider_name not in _SUPPORTED_PROVIDERS:
            raise EmbedderUnavailableError(
                "FastEmbed execution provider is not supported",
                hint="Use CPUExecutionProvider or CUDAExecutionProvider.",
            )
        provider_names.append(provider_name)
        native_providers.append(native_provider)

    if (
        "CUDAExecutionProvider" in provider_names
        and provider_names[0] != "CUDAExecutionProvider"
    ):
        raise EmbedderUnavailableError(
            "CUDA must be first in the FastEmbed providers list",
            hint=(
                "Put CUDAExecutionProvider first in the providers list when "
                "selecting CUDA."
            ),
        )
    options["providers"] = native_providers
    return provider_names


def _prepare_execution_options(options: dict[str, object]) -> tuple[bool, str | None]:
    """Apply the CPU default and validate native execution selection."""
    if options.get("lazy_load", False) is not False:
        raise EmbedderUnavailableError(
            "FastEmbed lazy loading is not supported",
            hint="Set FastEmbed lazy_load to false or omit it.",
        )

    has_cuda_option = "cuda" in options
    has_provider_option = "providers" in options
    cuda = options.get("cuda")
    if cuda is not None and type(cuda) is not bool:
        raise EmbedderUnavailableError(
            "FastEmbed cuda must be true, false, or null",
            hint="Set FastEmbed cuda to a boolean or use providers.",
        )

    provider_names = _normalize_provider_options(options)
    if provider_names and cuda is not None:
        requests_cuda = "CUDAExecutionProvider" in provider_names
        if cuda is True:
            raise EmbedderUnavailableError(
                "FastEmbed cuda and providers are mutually exclusive",
                hint="Set only one of the FastEmbed cuda and providers options.",
            )
        if requests_cuda is not cuda:
            raise EmbedderUnavailableError(
                "FastEmbed cuda and providers settings conflict",
                hint=(
                    "Set cuda to false with CPU providers, or select CUDA only "
                    "through providers."
                ),
            )

    if not has_cuda_option and not has_provider_option:
        options["providers"] = ["CPUExecutionProvider"]
    options.setdefault("lazy_load", False)

    requests_cuda = cuda is True or "CUDAExecutionProvider" in provider_names
    if requests_cuda:
        return True, "CUDAExecutionProvider"
    if provider_names:
        return False, provider_names[0]
    if cuda is False or (not has_cuda_option and not has_provider_option):
        return False, "CPUExecutionProvider"
    return False, None


def _ensure_cuda_available() -> None:
    """Reject explicit CUDA when ONNX Runtime cannot provide it."""
    try:
        onnxruntime = import_extra("onnxruntime", "fastembed")
        available_providers = onnxruntime.get_available_providers()
    except Exception as exc:
        raise EmbedderUnavailableError(
            "CUDA provider availability could not be checked",
            hint=(
                "Install a CUDA-capable FastEmbed runtime and verify "
                "CUDAExecutionProvider availability."
            ),
        ) from exc
    if "CUDAExecutionProvider" not in available_providers:
        raise EmbedderUnavailableError(
            "CUDAExecutionProvider is not available",
            hint=(
                "Install a CUDA-capable FastEmbed runtime and make "
                "CUDAExecutionProvider available."
            ),
        )


def _inspect_execution_session(
    model: "TextEmbedding", model_name: str, expected_provider: str | None
) -> None:
    """Verify the loaded backend and disable ONNX fallback for CUDA sessions."""
    session = getattr(model.model, "model", None)
    if session is None:
        if model_name.startswith("minishlab/potion-"):
            return
        raise EmbedderUnavailableError(
            "FastEmbed model has no inspectable ONNX session",
            hint=(
                "Use an eager ONNX-backed model with public execution provider "
                "inspection."
            ),
        )

    get_providers = getattr(session, "get_providers", None)
    if not callable(get_providers):
        raise EmbedderUnavailableError(
            "FastEmbed ONNX session has no provider inspection",
            hint="Use an ONNX Runtime session with public get_providers support.",
        )
    try:
        actual_providers = get_providers()
    except Exception as exc:
        raise EmbedderUnavailableError(
            "FastEmbed ONNX providers could not be inspected",
            hint="Use an ONNX Runtime session with public get_providers support.",
        ) from exc
    if not actual_providers or any(
        provider not in _SUPPORTED_PROVIDERS for provider in actual_providers
    ):
        raise EmbedderUnavailableError(
            "FastEmbed selected an unsupported ONNX provider",
            hint=(
                "Use CPUExecutionProvider or CUDAExecutionProvider as active providers."
            ),
        )
    if expected_provider is not None and actual_providers[0] != expected_provider:
        raise EmbedderUnavailableError(
            "FastEmbed did not activate the requested execution provider",
            hint=(
                f"The ONNX session must place {expected_provider} first. "
                "Check the cuda and providers settings."
            ),
        )
    if actual_providers[0] != "CUDAExecutionProvider":
        return

    disable_fallback = getattr(session, "disable_fallback", None)
    if not callable(disable_fallback):
        raise EmbedderUnavailableError(
            "FastEmbed ONNX session cannot disable provider fallback",
            hint="Use an ONNX Runtime session with public disable_fallback support.",
        )
    try:
        disable_fallback()
    except Exception as exc:
        raise EmbedderUnavailableError(
            "FastEmbed ONNX provider fallback could not be disabled",
            hint="Use an ONNX Runtime session that supports disable_fallback.",
        ) from exc


class LocalEmbedder:
    """Embed text with a reviewed FastEmbed model.

    Attributes:
        model_id: Semantic model identity stored with each space.
        dim: Width reported by the initialized FastEmbed model.
        max_input_tokens: Effective initialized tokenizer truncation limit.
    """

    def __init__(
        self,
        model: "TextEmbedding",
        *,
        model_name: str,
        model_id: str,
        dim: int,
        max_input_tokens: int,
        batch_size: int,
    ) -> None:
        """Bind a loaded model with its initialized encoding limits.

        Args:
            model: Native model loaded by open, or a test substitute.
            model_name: Canonical native model name used for input formatting.
            model_id: Semantic model identity stored with each space.
            dim: Width reported by the initialized model.
            max_input_tokens: Effective tokenizer truncation limit.
            batch_size: Native inference batch size.
        """
        self._model: TextEmbedding | None = model
        self._model_name = model_name
        self.model_id = model_id
        self.dim = dim
        self.max_input_tokens = max_input_tokens
        self._batch_size = batch_size
        # A cancelled caller cannot stop a native call that is already running.
        # One worker thread runs batches in submission order, so a later batch
        # waits for an abandoned one instead of entering the model beside it.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="local-embedder"
        )

    @classmethod
    async def open(cls, settings: EmbeddingSettings | None = None) -> "LocalEmbedder":
        """Load a reviewed FastEmbed model and inspect its initialized limits.

        The first open may download model files. fastembed uses
        FASTEMBED_CACHE_PATH, or fastembed_cache in the system temp directory.

        Args:
            settings: Native FastEmbed options and inference batch size. If
                omitted, values load from the environment.

        Returns:
            A local embedder ready for a Tenant stack.

        Raises:
            EmbedderUnavailableError: If the extra, model, or reviewed format
                cannot load.
        """
        selected = settings if settings is not None else EmbeddingSettings()

        def load() -> tuple["TextEmbedding", str, str, int, int]:
            module = import_extra("fastembed", "fastembed")
            options: dict[str, object] = {}
            options.update(deepcopy(selected.fastembed_kwargs))
            requests_cuda, expected_provider = _prepare_execution_options(options)

            requested_name = options.get("model_name", _DEFAULT_MODEL_NAME)
            if not isinstance(requested_name, str):
                raise ValueError("invalid FastEmbed model name")
            native_description = next(
                (
                    description
                    for description in module.TextEmbedding.list_supported_models()
                    if description["model"].casefold() == requested_name.casefold()
                ),
                None,
            )
            if native_description is None:
                raise ValueError("unsupported FastEmbed model")
            native_name = native_description["model"]
            canonical_name = next(
                (
                    name
                    for name in ENCODING_POLICIES
                    if name.casefold() == native_name.casefold()
                ),
                None,
            )
            if canonical_name is None:
                raise EmbedderUnavailableError(
                    "FastEmbed model has no reviewed encoding policy",
                    hint="Choose a supported FastEmbed formatting model.",
                )
            if native_description["dim"] > 4000:
                raise ValueError("unsupported FastEmbed embedding dimensions")

            if requests_cuda and canonical_name.startswith("minishlab/potion-"):
                raise EmbedderUnavailableError(
                    "FastEmbed Potion models are CPU-only",
                    hint=(
                        "Potion is CPU-only. Choose an ONNX-backed model to use "
                        "CUDA execution."
                    ),
                )
            if requests_cuda:
                _ensure_cuda_available()

            options["model_name"] = canonical_name
            model = module.TextEmbedding(**options)
            dim = model.embedding_size
            if not 1 <= dim <= 4000:
                raise ValueError("unsupported FastEmbed embedding dimensions")

            tokenizer = model.model.tokenizer
            truncation = tokenizer.truncation
            if truncation is None:
                raise ValueError("FastEmbed tokenizer has no truncation limit")
            max_input_tokens = truncation["max_length"]
            if not isinstance(max_input_tokens, int) or max_input_tokens < 1:
                raise ValueError("invalid FastEmbed tokenizer truncation limit")

            _inspect_execution_session(model, canonical_name, expected_provider)

            encoding_policy = ENCODING_POLICIES[canonical_name]
            identity_payload = {
                "identity_version": 1,
                "model_name": canonical_name,
                "encoding_policy": {
                    "version": encoding_policy.version,
                    "query_prefix": encoding_policy.query_prefix,
                    "document_prefix": encoding_policy.document_prefix,
                    "query_task": encoding_policy.query_task,
                    "document_task": encoding_policy.document_task,
                },
                "normalization_policy": "l2-unit-v1",
                "max_input_tokens": max_input_tokens,
                "native_options": {
                    name: value
                    for name, value in options.items()
                    if name not in _NON_SEMANTIC_OPTIONS
                },
            }
            serialized_identity = json.dumps(
                identity_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            digest = hashlib.sha256(serialized_identity).hexdigest()
            model_id = f"fastembed:{canonical_name}:{digest}"
            return model, canonical_name, model_id, dim, max_input_tokens

        try:
            (
                model,
                model_name,
                model_id,
                dim,
                max_input_tokens,
            ) = await asyncio.to_thread(load)
        except Exception as exc:
            if isinstance(exc, EmbedderUnavailableError):
                raise
            raise EmbedderUnavailableError(
                "local embedding model could not load",
                hint=(
                    'Install "ontogr-mem[fastembed]" on a supported platform, '
                    "or set MEMORY_EMBEDDING_MODEL and MEMORY_EMBEDDING_BASE_URL. "
                    "Check model cache access and download connectivity."
                ),
            ) from exc
        return cls(
            model,
            model_name=model_name,
            model_id=model_id,
            dim=dim,
            max_input_tokens=max_input_tokens,
            batch_size=selected.batch_size,
        )

    async def embed_batch(
        self, texts: Sequence[str], *, purpose: EmbeddingPurpose
    ) -> list[list[float]]:
        """Return aligned unit vectors without blocking the event loop.

        Batches run one at a time on a single worker thread. If the awaiting
        caller is cancelled, its batch keeps running on that thread, and later
        batches wait for it to finish.

        Args:
            texts: One gateway batch. The tokenizer applies its own truncation.
            purpose: Whether the text will be searched or stored.

        Returns:
            One unit vector per text, or an empty list for empty input.

        Raises:
            PermanentEmbeddingError: If the model fails or returns bad vectors.
            EmbedderUnavailableError: If this embedder has been closed.
        """
        # Read the model and submit the batch without yielding to the loop. This
        # keeps aclose from shutting the executor down between the check and the
        # submission.
        model = self._model
        if model is None:
            raise EmbedderUnavailableError(
                "local embedder is closed", hint="Open a new LocalEmbedder."
            )
        if not texts:
            return []
        documents = list(texts)
        formatted = [
            format_text(self._model_name, text, purpose=purpose) for text in documents
        ]
        encode = model.query_embed if purpose == "query" else model.passage_embed

        def embed() -> list[list[float]]:
            vectors = [
                list(to_unit(vector.tolist(), self.dim))
                for vector in encode(
                    formatted, batch_size=self._batch_size, parallel=None
                )
            ]
            if len(vectors) != len(formatted):
                raise PermanentEmbeddingError("local embedding count mismatch")
            return vectors

        try:
            return await asyncio.get_running_loop().run_in_executor(
                self._executor, embed
            )
        except EmbeddingError:
            raise
        except Exception as exc:
            raise PermanentEmbeddingError("local embedding failed") from exc

    async def aclose(self) -> None:
        """Close the embedder and wait for batches already submitted to finish.

        Batches started after this call raise EmbedderUnavailableError. Batches
        submitted before it, queued or running, still finish and return. The
        wait runs in a worker thread, so the event loop stays free. Calling this
        method again is safe.
        """
        self._model = None
        await asyncio.to_thread(self._executor.shutdown, wait=True)
