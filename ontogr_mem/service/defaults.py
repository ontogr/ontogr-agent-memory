"""Default entity and relation catalog for a new Tenant."""

from ontogr_mem.ontology.models import AddEntity, AddRelation, ChangeSet, PatternSpec


DEFAULT_CATALOG = ChangeSet(
    base_version=0,
    ops=[
        AddEntity(label="Person"),
        AddEntity(label="Organization"),
        AddEntity(label="Place"),
        AddEntity(label="Thing"),
        AddEntity(label="Event"),
        AddRelation(
            label="WORKS_AT",
            exclusive=True,
            patterns=[PatternSpec(src="Person", tgt="Organization")],
        ),
        AddRelation(
            label="LIVES_IN",
            exclusive=True,
            patterns=[PatternSpec(src="Person", tgt="Place")],
        ),
        AddRelation(
            label="MARRIED_TO",
            exclusive=True,
            patterns=[PatternSpec(src="Person", tgt="Person")],
        ),
        AddRelation(
            label="LOCATED_IN",
            exclusive=True,
            patterns=[PatternSpec(src="Organization", tgt="Place")],
        ),
        AddRelation(
            label="MEMBER_OF",
            patterns=[PatternSpec(src="Person", tgt="Organization")],
        ),
        AddRelation(
            label="KNOWS",
            patterns=[PatternSpec(src="Person", tgt="Person")],
        ),
        AddRelation(label="OWNS", patterns=[PatternSpec(src="Person", tgt="Thing")]),
        AddRelation(label="RELATED_TO"),
    ],
)
