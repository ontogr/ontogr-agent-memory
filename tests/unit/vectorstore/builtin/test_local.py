"""Local embedder tests with the fastembed model replaced by an in-process fake."""

import asyncio
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from ontogr_mem.vectorstore.builtin import local as local_module
from ontogr_mem.vectorstore.builtin.formatting import ENCODING_POLICIES
from ontogr_mem.vectorstore.builtin.local import LocalEmbedder
from ontogr_mem.vectorstore.builtin.settings import EmbeddingSettings
from ontogr_mem.vectorstore.errors import (
    EmbedderUnavailableError,
    PermanentEmbeddingError,
)


SETUP_HINTS = (
    "ontogr-mem[fastembed]",
    "MEMORY_EMBEDDING_MODEL",
    "MEMORY_EMBEDDING_BASE_URL",
)


def vector_with(*leading: float, dimension: int = 384) -> list[float]:
    """Return a vector of `dimension` values starting with `leading`."""
    return [*leading, *([0.0] * (dimension - len(leading)))]


def default_vector_for(_text: str, purpose: str, dimension: int) -> list[float]:
    """Return normalized-distinct vectors for queries and documents."""
    leading = (4.0, 3.0) if purpose == "query" else (3.0, 4.0)
    return vector_with(*leading, dimension=dimension)


class FakeVector:
    """Stands in for the numpy array that fastembed yields for each document."""

    def __init__(self, values: list[float]) -> None:
        """Keep the values that `tolist` returns."""
        self.values = values

    def tolist(self) -> list[float]:
        """Return the values as a plain list."""
        return list(self.values)


class FakeOnnxSession:
    """Expose the public provider and fallback controls used by ONNX Runtime."""

    def __init__(self, providers: Sequence[str]) -> None:
        """Record the providers and whether runtime fallback was disabled."""
        self.providers = list(providers)
        self.fallback_disabled = False

    def get_providers(self) -> list[str]:
        """Return the active execution provider order."""
        return list(self.providers)

    def disable_fallback(self) -> None:
        """Prevent ONNX Runtime from retrying on a later provider."""
        self.fallback_disabled = True


class FakeOnnxSessionWithoutFallback:
    """Expose provider inspection without fallback control."""

    def __init__(self, providers: Sequence[str]) -> None:
        """Keep the active provider order."""
        self.providers = list(providers)

    def get_providers(self) -> list[str]:
        """Return the active execution provider order."""
        return list(self.providers)


class FailingFakeOnnxSession(FakeOnnxSession):
    """Raise when asked to disable ONNX Runtime provider fallback."""

    def disable_fallback(self) -> None:
        """Raise a private native error to check safe translation."""
        raise RuntimeError("/private/model/cache")


class FakeTextEmbedding:
    """Stands in for `fastembed.TextEmbedding` so that no model is loaded.

    Each instance records its constructor arguments, the thread that built it,
    and the threads that iterated its embeddings. Subclasses change the failure
    mode. `install_fake` derives the class that a test installs, so `instances`
    and `vector_for` are never shared between tests.
    """

    instances: list["FakeTextEmbedding"] = []
    vector_for = staticmethod(default_vector_for)
    supported_models = [
        {
            "model": "ibm-granite/granite-embedding-small-english-r2",
            "dim": 384,
        },
        {"model": "nomic-ai/nomic-embed-text-v1.5", "dim": 768},
        {"model": "google/embeddinggemma-300m", "dim": 768},
        {"model": "jinaai/jina-embeddings-v3", "dim": 1024},
    ]
    embedding_size_override: int | None = None
    truncation_max_length = 512
    session_providers: list[str] | None = None
    session_factory: Callable[[Sequence[str]], object] = FakeOnnxSession

    @classmethod
    def list_supported_models(cls) -> list[dict[str, object]]:
        """Return native-style model metadata without loading model files."""
        return [dict(description) for description in cls.supported_models]

    def __init__(self, **options: object) -> None:
        """Record native options and expose initialized model metadata."""
        self.constructor_options = options.copy()
        self.model_name = str(options["model_name"])
        description = next(
            item
            for item in type(self).supported_models
            if item["model"] == self.model_name
        )
        self.embedding_size = type(self).embedding_size_override or int(
            description["dim"]
        )
        native_model = SimpleNamespace(
            tokenizer=SimpleNamespace(
                truncation={"max_length": type(self).truncation_max_length}
            ),
        )
        if not self.model_name.startswith("minishlab/potion-"):
            providers = type(self).session_providers
            if providers is None:
                configured_providers = options.get("providers")
                provider_names = (
                    [
                        provider[0] if isinstance(provider, (list, tuple)) else provider
                        for provider in configured_providers
                    ]
                    if isinstance(configured_providers, list)
                    else []
                )
                uses_cuda = (
                    options.get("cuda") is True
                    or "CUDAExecutionProvider" in provider_names
                )
                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if uses_cuda
                    else ["CPUExecutionProvider"]
                )
            native_model.model = type(self).session_factory(providers)
        self.model = native_model
        self.init_thread = threading.get_ident()
        self.embed_threads: list[int] = []
        self.calls: list[tuple[str, list[str], int, int | None]] = []
        self.in_progress = False
        type(self).instances.append(self)

    def query_embed(
        self, texts: Sequence[str], *, batch_size: int, parallel: int | None
    ) -> Iterator[FakeVector]:
        """Yield query vectors through the native purpose-specific method."""
        yield from self._encode("query", texts, batch_size, parallel)

    def passage_embed(
        self, texts: Sequence[str], *, batch_size: int, parallel: int | None
    ) -> Iterator[FakeVector]:
        """Yield document vectors through the native passage method."""
        yield from self._encode("document", texts, batch_size, parallel)

    def _encode(
        self,
        purpose: str,
        texts: Sequence[str],
        batch_size: int,
        parallel: int | None,
    ) -> Iterator[FakeVector]:
        """Record native inference and yield vectors in input order."""
        self.calls.append((purpose, list(texts), batch_size, parallel))
        self.embed_threads.append(threading.get_ident())
        for text in texts:
            yield FakeVector(type(self).vector_for(text, purpose, self.embedding_size))


class ShortFakeTextEmbedding(FakeTextEmbedding):
    """Yields one vector fewer than the number of documents."""

    def _encode(
        self,
        purpose: str,
        texts: Sequence[str],
        batch_size: int,
        parallel: int | None,
    ) -> Iterator[FakeVector]:
        """Drop the vector for the last input."""
        self.calls.append((purpose, list(texts), batch_size, parallel))
        self.embed_threads.append(threading.get_ident())
        for text in texts[:-1]:
            yield FakeVector(type(self).vector_for(text, purpose, self.embedding_size))


class FailingFakeTextEmbedding(FakeTextEmbedding):
    """Raises a runtime error after the vectors are produced."""

    def _encode(
        self,
        purpose: str,
        texts: Sequence[str],
        batch_size: int,
        parallel: int | None,
    ) -> Iterator[FakeVector]:
        """Yield the vectors, then fail as a broken runtime does."""
        self.calls.append((purpose, list(texts), batch_size, parallel))
        self.embed_threads.append(threading.get_ident())
        for text in texts:
            yield FakeVector(type(self).vector_for(text, purpose, self.embedding_size))
        raise RuntimeError("onnx session failed")


class TypeErrorFakeTextEmbedding(FakeTextEmbedding):
    """Raises a type error before it yields, as a mismatched argument would."""

    def _encode(
        self,
        purpose: str,
        texts: Sequence[str],
        batch_size: int,
        parallel: int | None,
    ) -> Iterator[FakeVector]:
        """Fail on the first call, before any vector is produced."""
        raise TypeError("model received an unexpected argument")


class SlowFakeTextEmbedding(FakeTextEmbedding):
    """Holds each call open so that an overlapping call is detected."""

    def _encode(
        self,
        purpose: str,
        texts: Sequence[str],
        batch_size: int,
        parallel: int | None,
    ) -> Iterator[FakeVector]:
        """Fail if another call is iterating, otherwise yield slowly."""
        if self.in_progress:
            raise AssertionError("embed overlapped with another call")
        self.in_progress = True
        try:
            self.calls.append((purpose, list(texts), batch_size, parallel))
            self.embed_threads.append(threading.get_ident())
            # A short pause gives overlapping callers time to reach the model.
            time.sleep(0.05)
            for text in texts:
                yield FakeVector(
                    type(self).vector_for(text, purpose, self.embedding_size)
                )
        finally:
            self.in_progress = False


class UnloadableFakeTextEmbedding(FakeTextEmbedding):
    """Fails on construction, as a missing or unreadable model file does."""

    def __init__(self, **options: object) -> None:
        """Raise the OSError that a model loader raises for a bad file."""
        raise OSError("model file could not be read")


def blocking_fake_type(
    started: threading.Event, release: threading.Event
) -> type[FakeTextEmbedding]:
    """Return a fake whose first call holds the model until `release` is set.

    Args:
        started: Set by the fake as soon as a call enters the model.
        release: Awaited by the fake for at most five seconds, so a failing test
            cannot leave a worker thread blocked forever.
    """

    class BlockingFakeTextEmbedding(FakeTextEmbedding):
        """Blocks inside `embed` and fails if a second call enters the model."""

        def _encode(
            self,
            purpose: str,
            texts: Sequence[str],
            batch_size: int,
            parallel: int | None,
        ) -> Iterator[FakeVector]:
            """Hold the model until `release` is set, then yield the vectors."""
            if self.in_progress:
                raise AssertionError("embed overlapped with another call")
            self.in_progress = True
            try:
                self.calls.append((purpose, list(texts), batch_size, parallel))
                self.embed_threads.append(threading.get_ident())
                started.set()
                release.wait(timeout=5)
                for text in texts:
                    yield FakeVector(
                        type(self).vector_for(text, purpose, self.embedding_size)
                    )
            finally:
                self.in_progress = False

    return BlockingFakeTextEmbedding


def install_fake(
    monkeypatch: pytest.MonkeyPatch,
    fake_type: type[FakeTextEmbedding] = FakeTextEmbedding,
    vector_for: Callable[[str, str, int], list[float]] = default_vector_for,
    *,
    supported_models: list[dict[str, object]] | None = None,
    truncation_max_length: int = 512,
    embedding_size: int | None = None,
    session_providers: list[str] | None = None,
    session_factory: Callable[[Sequence[str]], object] | None = None,
) -> type[FakeTextEmbedding]:
    """Install a new subclass of `fake_type` as the `fastembed` module.

    The stub replaces the module in `sys.modules`, so the real package, and the
    network probes its imports perform, never load. Each test gets its own
    class, so the `instances` list and the vector function never carry over
    from one test to the next. Monkeypatch restores the real module when the
    test ends.

    Args:
        monkeypatch: Fixture that undoes the patch after the test.
        fake_type: Fake to subclass. Its overrides define the failure mode.
        vector_for: Maps one document to the vector it receives.
        supported_models: Native-style model metadata exposed by the fake.
        truncation_max_length: Effective truncation limit exposed by its tokenizer.
        embedding_size: Actual initialized width, when different from metadata.
        session_providers: Actual provider order exposed by the fake ONNX session.
        session_factory: ONNX session shape exposed by the initialized model.

    Returns:
        The installed class. Its `instances` list holds every built object.
    """
    installed = type(
        "FakeTextEmbedding",
        (fake_type,),
        {
            "instances": [],
            "vector_for": staticmethod(vector_for),
            "supported_models": (
                supported_models
                if supported_models is not None
                else fake_type.supported_models
            ),
            "truncation_max_length": truncation_max_length,
            "embedding_size_override": embedding_size,
            "session_providers": (
                session_providers if session_providers is not None else None
            ),
            "session_factory": session_factory or fake_type.session_factory,
        },
    )
    stub = ModuleType("fastembed")
    stub.TextEmbedding = installed
    monkeypatch.setitem(sys.modules, "fastembed", stub)
    return installed


def install_onnxruntime(
    monkeypatch: pytest.MonkeyPatch, providers: Sequence[str]
) -> None:
    """Replace the optional ONNX Runtime import with an available-provider stub."""
    stub = ModuleType("onnxruntime")
    stub.get_available_providers = lambda: list(providers)
    monkeypatch.setitem(sys.modules, "onnxruntime", stub)


def assert_setup_hint(error: EmbedderUnavailableError) -> None:
    """Assert the hint names the install extra and both embedding settings."""
    for expected in SETUP_HINTS:
        assert expected in error.hint


async def wait_for_event(event: threading.Event) -> None:
    """Poll `event` from the event loop until it is set, failing after five seconds."""
    for _ in range(500):
        if event.is_set():
            return
        await asyncio.sleep(0.01)
    pytest.fail("the worker thread never reached the model")


class TestLocalEmbedder:
    """Behavior of `LocalEmbedder` with the fastembed model replaced by a fake."""

    @pytest.mark.parametrize(
        ("first_options", "second_options", "first_batch", "second_batch"),
        [
            pytest.param(
                {},
                {"model_name": "ibm-granite/granite-embedding-small-english-r2"},
                32,
                32,
                id="omitted-or-explicit-default-model",
            ),
            pytest.param(
                {"cache_dir": "/cache/first"},
                {"cache_dir": "/cache/second"},
                32,
                32,
                id="cache-directory",
            ),
            pytest.param({"threads": 1}, {"threads": 8}, 32, 32, id="threads"),
            pytest.param(
                {"providers": ["CPUExecutionProvider"]},
                {"providers": ["CUDAExecutionProvider"]},
                32,
                32,
                id="providers",
            ),
            pytest.param({"cuda": False}, {"cuda": True}, 32, 32, id="cuda"),
            pytest.param({"device_id": 0}, {"device_id": 1}, 32, 32, id="device-id"),
            pytest.param(
                {"device_ids": [0]}, {"device_ids": [1]}, 32, 32, id="device-ids"
            ),
            pytest.param({}, {"lazy_load": False}, 32, 32, id="eager-loading"),
            pytest.param(
                {"local_files_only": False},
                {"local_files_only": True},
                32,
                32,
                id="local-files-only",
            ),
            pytest.param({}, {}, 4, 128, id="call-batch-size"),
        ],
    )
    async def test_model_id_ignores_model_execution_and_cache_options(
        self,
        monkeypatch: pytest.MonkeyPatch,
        first_options: dict[str, Any],
        second_options: dict[str, Any],
        first_batch: int,
        second_batch: int,
    ) -> None:
        """Execution, cache and call-batch choices do not change model identity."""
        install_fake(monkeypatch)
        install_onnxruntime(
            monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        first = await LocalEmbedder.open(
            EmbeddingSettings(
                _env_file=None,
                fastembed_kwargs=first_options,
                batch_size=first_batch,
            )
        )
        second = await LocalEmbedder.open(
            EmbeddingSettings(
                _env_file=None,
                fastembed_kwargs=second_options,
                batch_size=second_batch,
            )
        )

        try:
            assert first.model_id == second.model_id
        finally:
            await asyncio.gather(first.aclose(), second.aclose())

    async def test_model_id_is_stable_when_semantic_kwargs_are_reordered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """JSON object key order does not change the model identity digest."""
        install_fake(monkeypatch)
        first = await LocalEmbedder.open(
            EmbeddingSettings(
                _env_file=None,
                fastembed_kwargs={
                    "custom_option": {"second": 2, "first": 1},
                    "specific_model_path": "/models/granite",
                },
            )
        )
        second = await LocalEmbedder.open(
            EmbeddingSettings(
                _env_file=None,
                fastembed_kwargs={
                    "specific_model_path": "/models/granite",
                    "custom_option": {"first": 1, "second": 2},
                },
            )
        )

        try:
            assert first.model_id == second.model_id
        finally:
            await asyncio.gather(first.aclose(), second.aclose())

    @pytest.mark.parametrize(
        ("first_options", "second_options", "private_values"),
        [
            pytest.param(
                {"specific_model_path": "/models/one"},
                {"specific_model_path": "/models/two"},
                ("/models/one", "/models/two"),
                id="specific-model-path",
            ),
            pytest.param(
                {"unrecognized_native_option": "first"},
                {"unrecognized_native_option": "second"},
                ("first", "second"),
                id="unknown-native-option",
            ),
        ],
    )
    async def test_model_id_changes_for_semantic_kwargs_without_exposing_them(
        self,
        monkeypatch: pytest.MonkeyPatch,
        first_options: dict[str, Any],
        second_options: dict[str, Any],
        private_values: tuple[str, str],
    ) -> None:
        """Semantic native options affect the digest but are not written in it."""
        install_fake(monkeypatch)
        first = await LocalEmbedder.open(
            EmbeddingSettings(_env_file=None, fastembed_kwargs=first_options)
        )
        second = await LocalEmbedder.open(
            EmbeddingSettings(_env_file=None, fastembed_kwargs=second_options)
        )

        try:
            assert first.model_id != second.model_id
            assert all(value not in first.model_id for value in private_values)
            assert all(value not in second.model_id for value in private_values)
        finally:
            await asyncio.gather(first.aclose(), second.aclose())

    async def test_model_id_changes_with_the_initialized_tokenizer_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The effective initialized tokenizer limit participates in identity."""
        install_fake(monkeypatch, truncation_max_length=256)
        first = await LocalEmbedder.open(EmbeddingSettings(_env_file=None))
        install_fake(monkeypatch, truncation_max_length=512)
        second = await LocalEmbedder.open(EmbeddingSettings(_env_file=None))

        try:
            assert first.max_input_tokens == 256
            assert second.max_input_tokens == 512
            assert first.model_id != second.model_id
        finally:
            await asyncio.gather(first.aclose(), second.aclose())

    @pytest.mark.parametrize(
        "field",
        [
            "version",
            "query_prefix",
            "document_prefix",
            "query_task",
            "document_task",
        ],
    )
    async def test_model_id_changes_when_reviewed_encoding_policy_changes(
        self, monkeypatch: pytest.MonkeyPatch, field: str
    ) -> None:
        """Every reviewed query or document policy field is identity-bearing."""
        install_fake(monkeypatch)
        first = await LocalEmbedder.open(EmbeddingSettings(_env_file=None))
        model_name = "ibm-granite/granite-embedding-small-english-r2"
        policy = ENCODING_POLICIES[model_name]
        changes = {
            "version": policy.version + 1,
            "query_prefix": policy.query_prefix + "changed",
            "document_prefix": policy.document_prefix + "changed",
            "query_task": "changed.query",
            "document_task": "changed.document",
        }
        updated_policies = dict(ENCODING_POLICIES)
        updated_policies[model_name] = replace(policy, **{field: changes[field]})
        monkeypatch.setattr(local_module, "ENCODING_POLICIES", updated_policies)
        second = await LocalEmbedder.open(EmbeddingSettings(_env_file=None))

        try:
            assert first.model_id != second.model_id
        finally:
            await asyncio.gather(first.aclose(), second.aclose())

    async def test_model_id_contains_canonical_name_and_full_sha256(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Opened local identities use the versioned provider/name/digest shape."""
        install_fake(monkeypatch)
        model_name = "ibm-granite/granite-embedding-small-english-r2"
        embedder = await LocalEmbedder.open(EmbeddingSettings(_env_file=None))

        try:
            identity_parts = embedder.model_id.split(":", maxsplit=2)
            assert identity_parts[:2] == ["fastembed", model_name]
            digest = identity_parts[2]
            assert len(digest) == 64
            assert all(character in "0123456789abcdef" for character in digest)
        finally:
            await embedder.aclose()

    async def test_open_builds_the_model_on_cpu_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The model is built for the embedder's model id on the CPU provider."""
        fake = install_fake(monkeypatch)

        embedder = await LocalEmbedder.open(EmbeddingSettings(_env_file=None))

        built = fake.instances[0]
        assert embedder.model_id.startswith(
            "fastembed:ibm-granite/granite-embedding-small-english-r2:"
        )
        assert built.constructor_options == {
            "model_name": "ibm-granite/granite-embedding-small-english-r2",
            "providers": ["CPUExecutionProvider"],
            "lazy_load": False,
        }

    async def test_uses_initialized_model_dimensions_and_tokenizer_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The embedder reports the actual width and shorter tokenizer cap."""
        install_fake(monkeypatch, truncation_max_length=256)
        settings = EmbeddingSettings(
            _env_file=None,
            fastembed_kwargs={"model_name": "NOMIC-AI/NOMIC-EMBED-TEXT-V1.5"},
        )

        embedder = await LocalEmbedder.open(settings)

        assert embedder.model_id.startswith("fastembed:nomic-ai/nomic-embed-text-v1.5:")
        assert embedder.dim == 768
        assert embedder.max_input_tokens == 256

    @pytest.mark.parametrize(
        ("native_options", "expected_options"),
        [
            pytest.param(
                {"threads": 2, "cache_dir": "/models"},
                {
                    "threads": 2,
                    "cache_dir": "/models",
                    "model_name": "ibm-granite/granite-embedding-small-english-r2",
                    "providers": ["CPUExecutionProvider"],
                    "lazy_load": False,
                },
                id="default-cpu",
            ),
            pytest.param(
                {"cuda": False, "threads": 3},
                {
                    "cuda": False,
                    "threads": 3,
                    "model_name": "ibm-granite/granite-embedding-small-english-r2",
                    "lazy_load": False,
                },
                id="explicit-cuda-option",
            ),
            pytest.param(
                {"providers": ["CUDAExecutionProvider"], "threads": 4},
                {
                    "providers": ["CUDAExecutionProvider"],
                    "threads": 4,
                    "model_name": "ibm-granite/granite-embedding-small-english-r2",
                    "lazy_load": False,
                },
                id="explicit-provider",
            ),
        ],
    )
    async def test_forwards_native_options_without_mutating_settings(
        self,
        monkeypatch: pytest.MonkeyPatch,
        native_options: dict[str, Any],
        expected_options: dict[str, Any],
    ) -> None:
        """Native options are preserved, defaulted on a copy, and passed through."""
        fake = install_fake(monkeypatch)
        install_onnxruntime(
            monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        settings = EmbeddingSettings(_env_file=None, fastembed_kwargs=native_options)

        await LocalEmbedder.open(settings)

        assert fake.instances[0].constructor_options == expected_options
        assert settings.fastembed_kwargs == native_options

    async def test_cuda_provider_pairs_are_normalized_without_mutating_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """JSON provider pairs reach FastEmbed as tuples with options unchanged."""
        native_options = {
            "providers": [
                ["CUDAExecutionProvider", {"device_id": "2", "arena": "kSame"}]
            ],
            "device_id": 7,
        }
        fake = install_fake(
            monkeypatch,
            session_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        install_onnxruntime(
            monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        settings = EmbeddingSettings(_env_file=None, fastembed_kwargs=native_options)

        embedder = await LocalEmbedder.open(settings)
        result = await embedder.embed_batch(["query"], purpose="query")

        assert len(result) == 1
        assert fake.instances[0].constructor_options["providers"] == [
            ("CUDAExecutionProvider", {"device_id": "2", "arena": "kSame"})
        ]
        assert fake.instances[0].constructor_options["device_id"] == 7
        assert settings.fastembed_kwargs == native_options
        assert fake.instances[0].model.model.fallback_disabled is True

    async def test_rejects_cuda_when_the_onnx_provider_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unavailable CUDA is reported before FastEmbed starts loading a model."""
        fake = install_fake(monkeypatch)
        install_onnxruntime(monkeypatch, ["CPUExecutionProvider"])
        settings = EmbeddingSettings(
            _env_file=None, fastembed_kwargs={"cuda": True, "device_id": 3}
        )

        with pytest.raises(EmbedderUnavailableError) as excinfo:
            await LocalEmbedder.open(settings)

        assert "CUDAExecutionProvider" in excinfo.value.hint
        assert "device_id" not in excinfo.value.hint
        assert fake.instances == []

    async def test_rejects_unsupported_provider_names_without_echoing_them(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only CPU and CUDA provider names cross the FastEmbed boundary."""
        fake = install_fake(monkeypatch)
        settings = EmbeddingSettings(
            _env_file=None,
            fastembed_kwargs={"providers": ["OpenVINOExecutionProvider"]},
        )

        with pytest.raises(EmbedderUnavailableError) as excinfo:
            await LocalEmbedder.open(settings)

        assert "CPUExecutionProvider" in excinfo.value.hint
        assert "CUDAExecutionProvider" in excinfo.value.hint
        assert "OpenVINOExecutionProvider" not in str(excinfo.value)
        assert "OpenVINOExecutionProvider" not in excinfo.value.hint
        assert fake.instances == []

    async def test_rejects_cpu_first_provider_lists_that_request_cuda(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A CUDA provider list must put CUDA first instead of relying on precedence."""
        fake = install_fake(monkeypatch)
        install_onnxruntime(
            monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        settings = EmbeddingSettings(
            _env_file=None,
            fastembed_kwargs={
                "providers": ["CPUExecutionProvider", "CUDAExecutionProvider"]
            },
        )

        with pytest.raises(EmbedderUnavailableError) as excinfo:
            await LocalEmbedder.open(settings)

        assert "CUDAExecutionProvider" in excinfo.value.hint
        assert "first" in excinfo.value.hint
        assert fake.instances == []

    @pytest.mark.parametrize(
        "native_options",
        [
            {"cuda": True, "providers": ["CPUExecutionProvider"]},
            {"cuda": False, "providers": ["CUDAExecutionProvider"]},
            {"cuda": True, "providers": ["CUDAExecutionProvider"]},
        ],
        ids=["cuda-with-cpu-list", "cpu-with-cuda-list", "mutually-exclusive"],
    )
    async def test_rejects_conflicting_cuda_and_provider_options(
        self,
        monkeypatch: pytest.MonkeyPatch,
        native_options: dict[str, Any],
    ) -> None:
        """Conflicting native selectors fail before their precedence is applied."""
        fake = install_fake(monkeypatch)
        settings = EmbeddingSettings(_env_file=None, fastembed_kwargs=native_options)

        with pytest.raises(EmbedderUnavailableError) as excinfo:
            await LocalEmbedder.open(settings)

        assert "cuda" in excinfo.value.hint
        assert "providers" in excinfo.value.hint
        assert fake.instances == []

    @pytest.mark.parametrize(
        "session_providers",
        [
            ["CPUExecutionProvider"],
            ["CPUExecutionProvider", "CUDAExecutionProvider"],
        ],
        ids=["cuda-missing", "cpu-first"],
    )
    async def test_rejects_a_loaded_cuda_model_without_cuda_first(
        self,
        monkeypatch: pytest.MonkeyPatch,
        session_providers: list[str],
    ) -> None:
        """A loaded session must confirm the requested CUDA provider is first."""
        fake = install_fake(monkeypatch, session_providers=session_providers)
        install_onnxruntime(
            monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        settings = EmbeddingSettings(_env_file=None, fastembed_kwargs={"cuda": True})

        with pytest.raises(EmbedderUnavailableError) as excinfo:
            await LocalEmbedder.open(settings)

        assert "CUDAExecutionProvider" in excinfo.value.hint
        assert len(fake.instances) == 1

    async def test_static_potion_model_runs_on_cpu_without_an_onnx_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CPU-only Potion accepts its native static path without ONNX controls."""
        model_name = "minishlab/potion-base-8M"
        fake = install_fake(
            monkeypatch,
            supported_models=[{"model": model_name, "dim": 256}],
        )
        settings = EmbeddingSettings(
            _env_file=None,
            fastembed_kwargs={
                "model_name": model_name,
                "providers": ["CPUExecutionProvider"],
                "device_id": 4,
            },
        )

        embedder = await LocalEmbedder.open(settings)
        result = await embedder.embed_batch(["text"], purpose="document")

        assert embedder.dim == 256
        assert len(result[0]) == 256
        assert not hasattr(fake.instances[0].model, "model")

    async def test_rejects_cuda_for_static_potion_models(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit CUDA is refused for a static model before native loading."""
        model_name = "minishlab/potion-base-8M"
        fake = install_fake(
            monkeypatch,
            supported_models=[{"model": model_name, "dim": 256}],
        )
        settings = EmbeddingSettings(
            _env_file=None,
            fastembed_kwargs={"model_name": model_name, "cuda": True},
        )

        with pytest.raises(EmbedderUnavailableError) as excinfo:
            await LocalEmbedder.open(settings)

        assert "CPU-only" in str(excinfo.value)
        assert fake.instances == []

    @pytest.mark.parametrize(
        ("session_providers", "fallback_disabled"),
        [
            (["CPUExecutionProvider"], False),
            (["CUDAExecutionProvider", "CPUExecutionProvider"], True),
        ],
        ids=["automatic-cpu", "automatic-cuda"],
    )
    async def test_native_automatic_selection_is_inspected_not_overridden(
        self,
        monkeypatch: pytest.MonkeyPatch,
        session_providers: list[str],
        fallback_disabled: bool,
    ) -> None:
        """Explicit native automatic selection is passed through and inspected."""
        fake = install_fake(monkeypatch, session_providers=session_providers)
        settings = EmbeddingSettings(
            _env_file=None,
            fastembed_kwargs={"cuda": None, "providers": None},
        )

        embedder = await LocalEmbedder.open(settings)

        assert fake.instances[0].constructor_options["cuda"] is None
        assert fake.instances[0].constructor_options["providers"] is None
        assert fake.instances[0].model.model.fallback_disabled is fallback_disabled
        assert embedder.dim == 384

    async def test_rejects_unsupported_provider_selected_during_native_auto(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Automatic native selection cannot activate an unsupported provider."""
        install_fake(
            monkeypatch,
            session_providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"],
        )
        settings = EmbeddingSettings(
            _env_file=None,
            fastembed_kwargs={"cuda": None, "providers": None},
        )

        with pytest.raises(EmbedderUnavailableError) as excinfo:
            await LocalEmbedder.open(settings)

        assert "unsupported ONNX provider" in str(excinfo.value)
        assert "CPUExecutionProvider" in excinfo.value.hint
        assert "CUDAExecutionProvider" in excinfo.value.hint
        assert "OpenVINOExecutionProvider" not in str(excinfo.value)

    @pytest.mark.parametrize(
        "session_factory",
        [
            pytest.param(FakeOnnxSessionWithoutFallback, id="missing-control"),
            pytest.param(FailingFakeOnnxSession, id="control-raises"),
        ],
    )
    async def test_rejects_cuda_session_when_fallback_cannot_be_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        session_factory: Callable[[Sequence[str]], object],
    ) -> None:
        """CUDA startup fails closed without working public fallback control."""
        install_fake(
            monkeypatch,
            session_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            session_factory=session_factory,
        )
        settings = EmbeddingSettings(
            _env_file=None,
            fastembed_kwargs={"cuda": None, "providers": None},
        )

        with pytest.raises(EmbedderUnavailableError) as excinfo:
            await LocalEmbedder.open(settings)

        assert "disable_fallback" in excinfo.value.hint
        assert "/private/model/cache" not in str(excinfo.value)

    async def test_cuda_inference_failure_does_not_retry_on_cpu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CUDA failure remains one call after fallback is disabled."""
        fake = install_fake(
            monkeypatch,
            fake_type=FailingFakeTextEmbedding,
            session_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        install_onnxruntime(
            monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        settings = EmbeddingSettings(
            _env_file=None,
            fastembed_kwargs={"providers": ["CUDAExecutionProvider"]},
        )
        embedder = await LocalEmbedder.open(settings)

        with pytest.raises(PermanentEmbeddingError) as excinfo:
            await embedder.embed_batch(["text"], purpose="query")

        assert str(excinfo.value) == "local embedding failed"
        assert len(fake.instances) == 1
        assert len(fake.instances[0].calls) == 1
        assert fake.instances[0].model.model.fallback_disabled is True

    async def test_forwards_configured_batch_size_to_native_inference(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Batch size and disabled parallelism reach the native call, not open."""
        fake = install_fake(monkeypatch)
        settings = EmbeddingSettings(_env_file=None, batch_size=7)
        embedder = await LocalEmbedder.open(settings)

        await embedder.embed_batch(["alpha"], purpose="document")

        built = fake.instances[0]
        assert "batch_size" not in built.constructor_options
        assert built.calls == [("document", ["alpha"], 7, None)]

    async def test_uses_query_and_passage_formatting_and_native_methods(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Raw text gets one reviewed prefix and purpose-specific native output."""
        fake = install_fake(monkeypatch)
        settings = EmbeddingSettings(
            _env_file=None,
            fastembed_kwargs={"model_name": "nomic-ai/nomic-embed-text-v1.5"},
        )
        embedder = await LocalEmbedder.open(settings)

        query = await embedder.embed_batch(
            ["search_query: current prefix"], purpose="query"
        )
        document = await embedder.embed_batch(
            ["search_document: current prefix"], purpose="document"
        )

        assert fake.instances[0].calls == [
            ("query", ["search_query: search_query: current prefix"], 32, None),
            (
                "document",
                ["search_document: search_document: current prefix"],
                32,
                None,
            ),
        ]
        assert [len(vector) for vector in query + document] == [768, 768]
        assert query[0][:2] == pytest.approx([0.8, 0.6])
        assert document[0][:2] == pytest.approx([0.6, 0.8])

    async def test_uses_each_jina_native_task_for_its_purpose(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Jina query and passage calls retain their distinct native tasks."""
        fake = install_fake(monkeypatch)
        settings = EmbeddingSettings(
            _env_file=None,
            fastembed_kwargs={"model_name": "jinaai/jina-embeddings-v3"},
        )
        embedder = await LocalEmbedder.open(settings)

        query = await embedder.embed_batch(["question"], purpose="query")
        document = await embedder.embed_batch(["passage"], purpose="document")

        assert [call[0] for call in fake.instances[0].calls] == ["query", "document"]
        assert [len(vector) for vector in query + document] == [1024, 1024]
        assert query[0][:2] == pytest.approx([0.8, 0.6])
        assert document[0][:2] == pytest.approx([0.6, 0.8])

    async def test_rejects_models_without_a_reviewed_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A native model without a reviewed formatting policy is not loaded."""
        fake = install_fake(
            monkeypatch,
            supported_models=[
                {"model": "unreviewed/model", "dim": 768},
            ],
        )
        settings = EmbeddingSettings(
            _env_file=None,
            fastembed_kwargs={"model_name": "unreviewed/model"},
        )

        with pytest.raises(EmbedderUnavailableError) as excinfo:
            await LocalEmbedder.open(settings)

        assert "supported FastEmbed formatting model" in excinfo.value.hint
        assert fake.instances == []

    async def test_rejects_models_over_the_vector_dimension_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A supported formatting model over 4000 dimensions is not loaded."""
        fake = install_fake(
            monkeypatch,
            supported_models=[
                {
                    "model": "ibm-granite/granite-embedding-small-english-r2",
                    "dim": 4001,
                },
            ],
        )

        with pytest.raises(EmbedderUnavailableError):
            await LocalEmbedder.open(EmbeddingSettings(_env_file=None))

        assert fake.instances == []

    async def test_rejects_an_initialized_model_over_the_dimension_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The loaded model's actual embedding size must remain within 4000."""
        fake = install_fake(monkeypatch, embedding_size=4001)

        with pytest.raises(EmbedderUnavailableError):
            await LocalEmbedder.open(EmbeddingSettings(_env_file=None))

        assert len(fake.instances) == 1

    async def test_rejects_lazy_model_loading(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A local embedder cannot open before the tokenizer is initialized."""
        fake = install_fake(monkeypatch)
        settings = EmbeddingSettings(
            _env_file=None, fastembed_kwargs={"lazy_load": True}
        )

        with pytest.raises(EmbedderUnavailableError) as excinfo:
            await LocalEmbedder.open(settings)

        assert "lazy_load" in excinfo.value.hint
        assert "false" in excinfo.value.hint
        assert fake.instances == []

    async def test_model_is_built_and_iterated_off_the_event_loop_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Building and iterating the model both leave the event loop thread."""
        loop_thread = threading.get_ident()
        fake = install_fake(monkeypatch)

        embedder = await LocalEmbedder.open()
        await embedder.embed_batch(["alpha"], purpose="document")

        built = fake.instances[0]
        assert built.init_thread != loop_thread
        assert built.embed_threads
        assert loop_thread not in built.embed_threads

    async def test_returns_unit_vectors_in_input_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each text gets its own unit-length vector, in input order."""
        vectors = {
            "alpha": vector_with(3.0, 4.0),
            "beta": vector_with(0.0, 0.0, 3.0, 4.0),
        }
        install_fake(
            monkeypatch,
            vector_for=lambda text, purpose, dimension: vectors[text],
        )
        embedder = await LocalEmbedder.open()

        out = await embedder.embed_batch(["alpha", "beta"], purpose="document")

        assert [len(vector) for vector in out] == [384, 384]
        assert out[0][:4] == pytest.approx([0.6, 0.8, 0.0, 0.0])
        assert out[1][:4] == pytest.approx([0.0, 0.0, 0.6, 0.8])

    async def test_returns_empty_list_for_empty_input(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty batch returns an empty list."""
        install_fake(monkeypatch)
        embedder = await LocalEmbedder.open()

        assert await embedder.embed_batch([], purpose="document") == []

    async def test_raises_permanent_error_when_a_vector_is_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model that returns fewer vectors than documents fails permanently."""
        install_fake(monkeypatch, fake_type=ShortFakeTextEmbedding)
        embedder = await LocalEmbedder.open()

        with pytest.raises(PermanentEmbeddingError):
            await embedder.embed_batch(["alpha", "beta"], purpose="document")

    @pytest.mark.parametrize(
        "bad_vector",
        [[1.0] * 10, [0.0] * 384],
        ids=["ten_entries", "all_zero"],
    )
    async def test_raises_permanent_error_for_a_bad_vector(
        self, monkeypatch: pytest.MonkeyPatch, bad_vector: list[float]
    ) -> None:
        """A vector of the wrong length or with zero norm fails permanently."""
        install_fake(
            monkeypatch,
            vector_for=lambda text, purpose, dimension: bad_vector,
        )
        embedder = await LocalEmbedder.open()

        with pytest.raises(PermanentEmbeddingError):
            await embedder.embed_batch(["alpha"], purpose="document")

    async def test_raises_permanent_error_when_the_runtime_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An error raised while the model runs fails permanently."""
        install_fake(monkeypatch, fake_type=FailingFakeTextEmbedding)
        embedder = await LocalEmbedder.open()

        with pytest.raises(PermanentEmbeddingError):
            await embedder.embed_batch(["alpha"], purpose="document")

    async def test_raises_permanent_error_for_a_type_error_from_the_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model error that is not a runtime error still fails permanently."""
        install_fake(monkeypatch, fake_type=TypeErrorFakeTextEmbedding)
        embedder = await LocalEmbedder.open()

        with pytest.raises(PermanentEmbeddingError):
            await embedder.embed_batch(["alpha"], purpose="document")

    async def test_concurrent_calls_are_serialized_on_the_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Overlapping callers wait for each other and all of them succeed."""
        install_fake(monkeypatch, fake_type=SlowFakeTextEmbedding)
        embedder = await LocalEmbedder.open()

        results = await asyncio.gather(
            embedder.embed_batch(["alpha"], purpose="document"),
            embedder.embed_batch(["beta"], purpose="document"),
            embedder.embed_batch(["gamma"], purpose="document"),
        )

        assert [len(batch) for batch in results] == [1, 1, 1]
        for batch in results:
            assert batch[0][:2] == pytest.approx([0.6, 0.8])

    async def test_a_cancelled_batch_does_not_overlap_the_next_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A later batch waits while a cancelled batch is still inside the model."""
        started = threading.Event()
        release = threading.Event()
        install_fake(monkeypatch, fake_type=blocking_fake_type(started, release))
        embedder = await LocalEmbedder.open()
        first = asyncio.create_task(embedder.embed_batch(["alpha"], purpose="document"))

        try:
            await wait_for_event(started)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            second = asyncio.create_task(
                embedder.embed_batch(["beta"], purpose="document")
            )
            await asyncio.sleep(0.05)
            assert not second.done()
            release.set()
            results = await second
        finally:
            release.set()

        assert len(results) == 1
        assert results[0][:2] == pytest.approx([0.6, 0.8])

    async def test_open_raises_unavailable_with_setup_hint_without_fastembed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing fastembed package makes `open` raise the setup error."""
        # A None entry makes the import fail the same way a missing package does.
        monkeypatch.setitem(sys.modules, "fastembed", None)

        with pytest.raises(EmbedderUnavailableError) as excinfo:
            await LocalEmbedder.open()

        assert_setup_hint(excinfo.value)

    async def test_open_raises_unavailable_when_the_model_cannot_load(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model that fails to load makes `open` raise the setup error."""
        install_fake(monkeypatch, fake_type=UnloadableFakeTextEmbedding)

        with pytest.raises(EmbedderUnavailableError) as excinfo:
            await LocalEmbedder.open()

        assert_setup_hint(excinfo.value)

    @pytest.mark.parametrize(
        "texts",
        [["alpha"], []],
        ids=["text", "empty_input"],
    )
    async def test_embed_batch_raises_unavailable_after_close(
        self, monkeypatch: pytest.MonkeyPatch, texts: list[str]
    ) -> None:
        """A closed embedder refuses every batch, including an empty one."""
        install_fake(monkeypatch)
        embedder = await LocalEmbedder.open()
        await embedder.aclose()

        with pytest.raises(EmbedderUnavailableError):
            await embedder.embed_batch(texts, purpose="document")

    async def test_second_aclose_keeps_the_embedder_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Closing twice does not raise, and the embedder stays unusable."""
        install_fake(monkeypatch)
        embedder = await LocalEmbedder.open()
        await embedder.aclose()

        await embedder.aclose()

        with pytest.raises(EmbedderUnavailableError):
            await embedder.embed_batch(["alpha"], purpose="document")

    async def test_aclose_waits_for_the_running_batch_to_finish(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Closing waits for the running batch, which still returns its vectors."""
        started = threading.Event()
        release = threading.Event()
        install_fake(monkeypatch, fake_type=blocking_fake_type(started, release))
        embedder = await LocalEmbedder.open()
        batch = asyncio.create_task(embedder.embed_batch(["alpha"], purpose="document"))

        try:
            await wait_for_event(started)
            closing = asyncio.create_task(embedder.aclose())
            await asyncio.sleep(0.05)
            assert not closing.done()
            release.set()
            results = await batch
            await closing
        finally:
            release.set()

        assert len(results) == 1
        assert results[0][:2] == pytest.approx([0.6, 0.8])
