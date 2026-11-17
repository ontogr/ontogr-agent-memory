"""Read a declaration from an OWL ontology in Turtle form."""

import re

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS

from ontogr_mem.ontology.declaration import (
    EntityDeclaration,
    OntologyDeclaration,
    PatternDeclaration,
    RelationDeclaration,
)
from ontogr_mem.ontology.errors import InvalidChange
from ontogr_mem.ontology.models import RelationKind


def declaration_from_turtle(text: str) -> OntologyDeclaration:
    """Read classes and object properties from Turtle text."""
    graph = Graph()
    try:
        graph.parse(data=text, format="turtle")
    except (SyntaxError, ValueError) as exc:
        raise InvalidChange("cannot read the Turtle ontology", hint=str(exc)) from exc
    classes = {
        subject
        for kind in (OWL.Class, RDFS.Class)
        for subject in graph.subjects(RDF.type, kind)
        if isinstance(subject, URIRef)
    }

    def label(iri: URIRef, *, property_label: bool = False) -> str:
        labels = sorted(
            str(value)
            for value in graph.objects(iri, RDFS.label)
            if isinstance(value, Literal)
        )
        value = labels[0] if labels else re.split(r"[#/]", str(iri))[-1]
        value = re.sub(r"\s+", "_", value.strip())
        if property_label:
            value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value).upper()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value):
            raise InvalidChange(
                "cannot make a label from the IRI",
                item=str(iri),
                hint="add an rdfs:label",
            )
        return value

    def metadata(iri: URIRef) -> tuple[str, list[str], bool]:
        comments = sorted(
            str(v) for v in graph.objects(iri, RDFS.comment) if isinstance(v, Literal)
        )
        definitions = sorted(
            str(v)
            for v in graph.objects(iri, SKOS.definition)
            if isinstance(v, Literal)
        )
        aliases = sorted(
            {
                str(v).strip()
                for v in graph.objects(iri, SKOS.altLabel)
                if isinstance(v, Literal)
            }
        )
        deprecated = any(
            isinstance(v, Literal) and v.toPython() is True
            for v in graph.objects(iri, OWL.deprecated)
        )
        return (comments or definitions or [""])[0], aliases, deprecated

    entity_labels = {iri: label(iri) for iri in classes}
    entities = []
    for iri, name in entity_labels.items():
        description, aliases, deprecated = metadata(iri)
        parents = sorted(
            entity_labels[parent]
            for parent in graph.objects(iri, RDFS.subClassOf)
            if isinstance(parent, URIRef) and parent in classes
        )
        entities.append(
            EntityDeclaration(
                label=name,
                description=description,
                aliases=aliases,
                parent=parents[0] if parents else None,
                deprecated=deprecated,
            )
        )
    relations = []
    properties = {
        subject
        for subject in graph.subjects(RDF.type, OWL.ObjectProperty)
        if isinstance(subject, URIRef)
    }
    for iri in properties:
        description, aliases, deprecated = metadata(iri)
        patterns = [
            PatternDeclaration(
                source=entity_labels[source], target=entity_labels[target]
            )
            for source in graph.objects(iri, RDFS.domain)
            if isinstance(source, URIRef) and source in classes
            for target in graph.objects(iri, RDFS.range)
            if isinstance(target, URIRef) and target in classes
        ]
        relations.append(
            RelationDeclaration(
                label=label(iri, property_label=True),
                description=description,
                aliases=aliases,
                kind=RelationKind.SINGLE
                if (iri, RDF.type, OWL.FunctionalProperty) in graph
                else RelationKind.MANY,
                symmetric=(iri, RDF.type, OWL.SymmetricProperty) in graph,
                patterns=patterns,
                deprecated=deprecated,
            )
        )
    return OntologyDeclaration(entities=entities, relations=relations)
