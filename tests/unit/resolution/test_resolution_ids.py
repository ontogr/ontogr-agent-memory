"""Tests for deterministic resolution ids."""

from ontogr_mem.resolution.ids import alias_id, entity_id, mention_id


def test_entity_id_is_the_literal_uuid5_of_the_episode_and_local_id():
    """The entity id is pinned to a literal UUID."""
    assert entity_id("ep1", "e0") == "408424a3-4698-5fec-b7d9-2d221ebd4221"


def test_mention_id_is_the_literal_hash_of_episode_local_id_and_version():
    """The mention id is pinned to a literal hash."""
    assert mention_id("ep1", "e0", "1") == "3b29e9eabbbaf942120ca7b3631212df"


def test_alias_id_is_the_literal_hash_of_the_entity_and_strict_key():
    """The alias id is pinned to a literal hash."""
    eid = entity_id("ep1", "e0")
    assert alias_id(eid, "john smith") == "cd9be9989cb437268503dad06eedab68"


def test_ids_are_stable_across_calls():
    """Repeating a call returns the same id."""
    assert entity_id("ep1", "e0") == entity_id("ep1", "e0")
    assert mention_id("ep1", "e0", "1") == mention_id("ep1", "e0", "1")
    assert alias_id("e1", "acme") == alias_id("e1", "acme")


def test_a_different_version_or_key_gives_a_different_id():
    """Version and key are part of the identity of a mention and an alias."""
    assert mention_id("ep1", "e0", "1") != mention_id("ep1", "e0", "2")
    assert alias_id("e1", "acme") != alias_id("e1", "acme inc")
