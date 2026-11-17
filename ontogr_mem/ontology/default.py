"""The default ontology for a new tenant."""

from functools import cache
from importlib.resources import files

from ontogr_mem.ontology.declaration import OntologyDeclaration
from ontogr_mem.ontology.loader import loads_ontology


@cache
def _parsed() -> OntologyDeclaration:
    text = (
        files("ontogr_mem.ontology")
        .joinpath("default.yaml")
        .read_text(encoding="utf-8")
    )
    return loads_ontology(text, file_format="yaml")


def default_ontology() -> OntologyDeclaration:
    """Return an independent copy of the bundled default ontology."""
    return _parsed().model_copy(deep=True)
