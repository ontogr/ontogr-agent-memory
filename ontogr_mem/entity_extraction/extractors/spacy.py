"""spaCy: classic entity recognition for well-known proper nouns."""

import asyncio
import threading
from collections.abc import Iterable
from typing import Any, ClassVar

from ontogr_mem._extras import import_extra
from ontogr_mem.data_models import Method, TypeCatalog
from ontogr_mem.entity_extraction.extractors.base import RawSpan, SpanEntityExtractor


class SpacyExtractor(SpanEntityExtractor):
    """Classic entity recognition with spaCy.

    The spaCy label set is fixed, so it cannot represent custom catalog types. It
    is most useful as the second member of a composite extractor. Only
    `DEFAULT_LABELS` are kept, because a wrong type is worse than a missing entity.

    Needs the `spacy` extra and a downloaded model, for example
    `python -m spacy download en_core_web_lg`.
    """

    method = Method.SPACY
    DEFAULT_LABELS: ClassVar[frozenset[str]] = frozenset(
        {"PERSON", "ORG", "GPE", "FAC"}
    )
    LABEL_MAP: ClassVar[dict[str, tuple[str, ...]]] = {
        "PERSON": ("Person",),
        "ORG": ("Organization", "Institution", "Company"),
        "GPE": ("Location", "Place", "GeographicLocation"),
        "LOC": ("Location", "Place", "GeographicLocation"),
        "FAC": ("Building", "Facility", "Location", "Place"),
        "NORP": ("Group", "Nationality", "Organization"),
        "EVENT": ("Event",),
        "PRODUCT": ("Product",),
        "WORK_OF_ART": ("Work", "Publication", "Product"),
        "LAW": ("Law",),
        "DATE": ("Date",),
        "LANGUAGE": ("Language",),
    }
    DEFAULT_MODEL = "en_core_web_lg"
    DISABLED_COMPONENTS: ClassVar[tuple[str, ...]] = (
        "tagger",
        "parser",
        "attribute_ruler",
        "lemmatizer",
    )
    _MODEL_CACHE: ClassVar[dict[str, Any]] = {}
    _CACHE_LOCK: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        model_name: str | None = None,
        labels: Iterable[str] | None = None,
        confidence: float = 0.75,
        languages: frozenset[str] | None = frozenset({"en"}),
        nlp: Any = None,
    ) -> None:
        """Configure the pipeline.

        Args:
            model_name: spaCy model name.
            labels: spaCy labels to keep.
            confidence: Fixed confidence for every entity, because spaCy gives no
                per-entity probability.
            languages: Languages handled.
            nlp: A loaded pipeline, used instead of loading one.
        """
        if isinstance(labels, str):
            msg = (
                f"labels must be an iterable of spaCy labels, not a string: {labels!r}"
            )
            raise TypeError(msg)
        self._model_name = model_name or self.DEFAULT_MODEL
        self._labels = frozenset(labels) if labels is not None else self.DEFAULT_LABELS
        self._confidence = confidence
        self._nlp = nlp
        super().__init__(threshold=0.0, floor=0.0, languages=languages)

    @classmethod
    def _shared_nlp(cls, model_name: str) -> Any:
        with cls._CACHE_LOCK:
            if model_name not in cls._MODEL_CACHE:
                spacy = import_extra("spacy", "spacy")
                try:
                    cls._MODEL_CACHE[model_name] = spacy.load(
                        model_name, disable=list(cls.DISABLED_COMPONENTS)
                    )
                except OSError as exc:
                    msg = (
                        f"spaCy model '{model_name}' is not installed. Run: "
                        f"python -m spacy download {model_name}"
                    )
                    raise OSError(msg) from exc
            return cls._MODEL_CACHE[model_name]

    def _label_for(self, label: str, types: TypeCatalog) -> str | None:
        for candidate in self.LABEL_MAP.get(label, ()):
            found = types.entity_by_label(candidate)
            if found is not None:
                return found.label
        return None

    def _run(self, text: str, types: TypeCatalog) -> list[RawSpan]:
        nlp = self._nlp or self._shared_nlp(self._model_name)
        hits: list[RawSpan] = []
        for ent in nlp(text).ents:
            if ent.label_ not in self._labels:
                continue
            label = self._label_for(ent.label_, types)
            if label is None:
                continue
            hits.append(
                RawSpan(ent.start_char, ent.end_char, ent.text, label, self._confidence)
            )
        return hits

    async def _predict(self, text: str, types: TypeCatalog) -> list[RawSpan]:
        return await asyncio.to_thread(self._run, text, types)
