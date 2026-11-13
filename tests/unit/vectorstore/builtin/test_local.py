"""Local embedder tests with the fastembed model replaced by an in-process fake."""

import asyncio
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from types import ModuleType

import pytest

from ontogr_mem.vectorstore.builtin.local import LocalEmbedder
from ontogr_mem.vectorstore.errors import (
    EmbedderUnavailableError,
    PermanentEmbeddingError,
)


SETUP_HINTS = (
    "ontogr-mem[fastembed]",
    "MEMORY_EMBEDDING_MODEL",
    "MEMORY_EMBEDDING_BASE_URL",
)


def vector_with(*leading: float) -> list[float]:
    """Return a 384-long vector that starts with `leading` and is zero after it."""
    return [*leading, *([0.0] * (384 - len(leading)))]


def default_vector_for(document: str) -> list[float]:
    """Return the same vector for every document, starting with 3.0 and 4.0."""
    return vector_with(3.0, 4.0)


class FakeVector:
    """Stands in for the numpy array that fastembed yields for each document."""

    def __init__(self, values: list[float]) -> None:
        """Keep the values that `tolist` returns."""
        self.values = values

    def tolist(self) -> list[float]:
        """Return the values as a plain list."""
        return list(self.values)


class FakeTextEmbedding:
    """Stands in for `fastembed.TextEmbedding` so that no model is loaded.

    Each instance records its constructor arguments, the thread that built it,
    and the threads that iterated its embeddings. Subclasses change the failure
    mode. `install_fake` derives the class that a test installs, so `instances`
    and `vector_for` are never shared between tests.
    """

    instances: list["FakeTextEmbedding"] = []
    vector_for = staticmethod(default_vector_for)

    def __init__(
        self, *, model_name: str, providers: list[str], lazy_load: bool
    ) -> None:
        """Record the constructor arguments and the thread that builds the fake."""
        self.model_name = model_name
        self.providers = providers
        self.lazy_load = lazy_load
        self.init_thread = threading.get_ident()
        self.embed_threads: list[int] = []
        self.in_progress = False
        type(self).instances.append(self)

    def embed(
        self, documents: Sequence[str], batch_size: int, parallel: int | None
    ) -> Iterator[FakeVector]:
        """Yield one vector per document and record the iterating thread."""
        self.embed_threads.append(threading.get_ident())
        for document in documents:
            yield FakeVector(type(self).vector_for(document))


class ShortFakeTextEmbedding(FakeTextEmbedding):
    """Yields one vector fewer than the number of documents."""

    def embed(
        self, documents: Sequence[str], batch_size: int, parallel: int | None
    ) -> Iterator[FakeVector]:
        """Drop the vector of the last document."""
        self.embed_threads.append(threading.get_ident())
        for document in documents[:-1]:
            yield FakeVector(type(self).vector_for(document))


class FailingFakeTextEmbedding(FakeTextEmbedding):
    """Raises a runtime error after the vectors are produced."""

    def embed(
        self, documents: Sequence[str], batch_size: int, parallel: int | None
    ) -> Iterator[FakeVector]:
        """Yield the vectors, then fail as a broken runtime does."""
        for document in documents:
            yield FakeVector(type(self).vector_for(document))
        raise RuntimeError("onnx session failed")


class TypeErrorFakeTextEmbedding(FakeTextEmbedding):
    """Raises a type error before it yields, as a mismatched argument would."""

    def embed(
        self, documents: Sequence[str], batch_size: int, parallel: int | None
    ) -> Iterator[FakeVector]:
        """Fail on the first call, before any vector is produced."""
        raise TypeError("model received an unexpected argument")


class SlowFakeTextEmbedding(FakeTextEmbedding):
    """Holds each call open so that an overlapping call is detected."""

    def embed(
        self, documents: Sequence[str], batch_size: int, parallel: int | None
    ) -> Iterator[FakeVector]:
        """Fail if another call is iterating, otherwise yield slowly."""
        if self.in_progress:
            raise AssertionError("embed overlapped with another call")
        self.in_progress = True
        try:
            self.embed_threads.append(threading.get_ident())
            # A short pause gives overlapping callers time to reach the model.
            time.sleep(0.05)
            for document in documents:
                yield FakeVector(type(self).vector_for(document))
        finally:
            self.in_progress = False


class UnloadableFakeTextEmbedding(FakeTextEmbedding):
    """Fails on construction, as a missing or unreadable model file does."""

    def __init__(
        self, *, model_name: str, providers: list[str], lazy_load: bool
    ) -> None:
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

        def embed(
            self, documents: Sequence[str], batch_size: int, parallel: int | None
        ) -> Iterator[FakeVector]:
            """Hold the model until `release` is set, then yield the vectors."""
            if self.in_progress:
                raise AssertionError("embed overlapped with another call")
            self.in_progress = True
            try:
                started.set()
                release.wait(timeout=5)
                for document in documents:
                    yield FakeVector(type(self).vector_for(document))
            finally:
                self.in_progress = False

    return BlockingFakeTextEmbedding


def install_fake(
    monkeypatch: pytest.MonkeyPatch,
    fake_type: type[FakeTextEmbedding] = FakeTextEmbedding,
    vector_for: Callable[[str], list[float]] = default_vector_for,
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

    Returns:
        The installed class. Its `instances` list holds every built object.
    """
    installed = type(
        "FakeTextEmbedding",
        (fake_type,),
        {"instances": [], "vector_for": staticmethod(vector_for)},
    )
    stub = ModuleType("fastembed")
    stub.TextEmbedding = installed
    monkeypatch.setitem(sys.modules, "fastembed", stub)
    return installed


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

    async def test_open_builds_the_model_on_cpu_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The model is built for the embedder's model id on the CPU provider."""
        fake = install_fake(monkeypatch)

        await LocalEmbedder.open()

        built = fake.instances[0]
        assert built.model_name == LocalEmbedder.model_id
        assert built.providers == ["CPUExecutionProvider"]
        assert built.lazy_load is False

    def test_declares_the_dimension_and_input_limit(self) -> None:
        """The embedder reports 384 dimensions and an 8192 token input limit."""
        assert LocalEmbedder.dim == 384
        assert LocalEmbedder.max_input_tokens == 8192

    async def test_model_is_built_and_iterated_off_the_event_loop_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Building and iterating the model both leave the event loop thread."""
        loop_thread = threading.get_ident()
        fake = install_fake(monkeypatch)

        embedder = await LocalEmbedder.open()
        await embedder.embed_batch(["alpha"])

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
        install_fake(monkeypatch, vector_for=lambda document: vectors[document])
        embedder = await LocalEmbedder.open()

        out = await embedder.embed_batch(["alpha", "beta"])

        assert [len(vector) for vector in out] == [384, 384]
        assert out[0][:4] == pytest.approx([0.6, 0.8, 0.0, 0.0])
        assert out[1][:4] == pytest.approx([0.0, 0.0, 0.6, 0.8])

    async def test_returns_empty_list_for_empty_input(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty batch returns an empty list."""
        install_fake(monkeypatch)
        embedder = await LocalEmbedder.open()

        assert await embedder.embed_batch([]) == []

    async def test_raises_permanent_error_when_a_vector_is_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model that returns fewer vectors than documents fails permanently."""
        install_fake(monkeypatch, fake_type=ShortFakeTextEmbedding)
        embedder = await LocalEmbedder.open()

        with pytest.raises(PermanentEmbeddingError):
            await embedder.embed_batch(["alpha", "beta"])

    @pytest.mark.parametrize(
        "bad_vector",
        [[1.0] * 10, [0.0] * 384],
        ids=["ten_entries", "all_zero"],
    )
    async def test_raises_permanent_error_for_a_bad_vector(
        self, monkeypatch: pytest.MonkeyPatch, bad_vector: list[float]
    ) -> None:
        """A vector of the wrong length or with zero norm fails permanently."""
        install_fake(monkeypatch, vector_for=lambda document: bad_vector)
        embedder = await LocalEmbedder.open()

        with pytest.raises(PermanentEmbeddingError):
            await embedder.embed_batch(["alpha"])

    async def test_raises_permanent_error_when_the_runtime_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An error raised while the model runs fails permanently."""
        install_fake(monkeypatch, fake_type=FailingFakeTextEmbedding)
        embedder = await LocalEmbedder.open()

        with pytest.raises(PermanentEmbeddingError):
            await embedder.embed_batch(["alpha"])

    async def test_raises_permanent_error_for_a_type_error_from_the_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model error that is not a runtime error still fails permanently."""
        install_fake(monkeypatch, fake_type=TypeErrorFakeTextEmbedding)
        embedder = await LocalEmbedder.open()

        with pytest.raises(PermanentEmbeddingError):
            await embedder.embed_batch(["alpha"])

    async def test_concurrent_calls_are_serialized_on_the_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Overlapping callers wait for each other and all of them succeed."""
        install_fake(monkeypatch, fake_type=SlowFakeTextEmbedding)
        embedder = await LocalEmbedder.open()

        results = await asyncio.gather(
            embedder.embed_batch(["alpha"]),
            embedder.embed_batch(["beta"]),
            embedder.embed_batch(["gamma"]),
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
        first = asyncio.create_task(embedder.embed_batch(["alpha"]))

        try:
            await wait_for_event(started)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            second = asyncio.create_task(embedder.embed_batch(["beta"]))
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
            await embedder.embed_batch(texts)

    async def test_second_aclose_keeps_the_embedder_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Closing twice does not raise, and the embedder stays unusable."""
        install_fake(monkeypatch)
        embedder = await LocalEmbedder.open()
        await embedder.aclose()

        await embedder.aclose()

        with pytest.raises(EmbedderUnavailableError):
            await embedder.embed_batch(["alpha"])

    async def test_aclose_waits_for_the_running_batch_to_finish(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Closing waits for the running batch, which still returns its vectors."""
        started = threading.Event()
        release = threading.Event()
        install_fake(monkeypatch, fake_type=blocking_fake_type(started, release))
        embedder = await LocalEmbedder.open()
        batch = asyncio.create_task(embedder.embed_batch(["alpha"]))

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
