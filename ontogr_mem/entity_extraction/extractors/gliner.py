"""GLiNER: local zero-shot entity recognition over the catalog labels."""

import asyncio
import threading
from typing import Any, ClassVar

from ontogr_mem._extras import import_extra
from ontogr_mem.data_models import Method, TypeCatalog
from ontogr_mem.entity_extraction.extractors.base import RawSpan, SpanEntityExtractor


class GLiNERExtractor(SpanEntityExtractor):
    """Zero-shot entity recognition with a local GLiNER model.

    Needs the `gliner` extra. Entities scoring between `0.75 * threshold` and
    `threshold` are typed Unknown. Set the threshold from a labelled sample for
    the model in use.
    """

    method = Method.GLINER
    DEFAULT_MODEL = "urchade/gliner_medium-v2.1"
    DEFAULT_THRESHOLD = 0.75
    UNKNOWN_BAND = 0.75

    _MODEL_CACHE: ClassVar[dict[str, Any]] = {}
    _CACHE_LOCK: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        model_name: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        floor: float | None = None,
        languages: frozenset[str] | None = frozenset({"en"}),
        model: Any = None,
    ) -> None:
        """Configure the model.

        Args:
            model_name: Hugging Face model name.
            threshold: Score from which a span gets its catalog type.
            floor: Score below which a span is ignored.
            languages: Languages handled.
            model: A loaded model, used instead of loading one.
        """
        self._model_name = model_name or self.DEFAULT_MODEL
        self._model = model
        super().__init__(
            threshold=threshold,
            floor=floor if floor is not None else threshold * self.UNKNOWN_BAND,
            languages=languages,
        )

    @classmethod
    def _shared_model(cls, model_name: str) -> Any:
        with cls._CACHE_LOCK:
            if model_name not in cls._MODEL_CACHE:
                gliner = import_extra("gliner", "gliner")
                cls._MODEL_CACHE[model_name] = gliner.GLiNER.from_pretrained(model_name)
            return cls._MODEL_CACHE[model_name]

    def _run(self, text: str, labels: list[str]) -> list[RawSpan]:
        model = self._model or self._shared_model(self._model_name)
        hits = model.predict_entities(text, labels, threshold=self._floor)
        return [
            RawSpan(h["start"], h["end"], h["text"], h["label"], float(h["score"]))
            for h in hits
        ]

    async def _predict(self, text: str, types: TypeCatalog) -> list[RawSpan]:
        labels = [t.label for t in types.entity_types]
        return await asyncio.to_thread(self._run, text, labels)
