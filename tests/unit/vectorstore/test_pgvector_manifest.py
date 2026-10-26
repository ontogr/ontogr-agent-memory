"""Tests for the pgvector index manifest."""

from ontogr_mem.vectorstore.models import SpaceSpec, VectorKind
from ontogr_mem.vectorstore.pgvector.schema import HnswParams, build_manifest


def _specs(dim: int = 3) -> dict[VectorKind, SpaceSpec]:
    """Return a space of every kind with the given dimension."""
    return {
        kind: SpaceSpec(kind, "mock", dim, f"{kind.value}.text", 1)
        for kind in VectorKind
    }


class TestBuildManifest:
    """Behavior of `build_manifest`."""

    def test_every_table_that_holds_a_vector_has_an_hnsw_index(self) -> None:
        """The node, edge and episode tables are all indexed."""
        manifest = build_manifest("m_t1", _specs())

        indexed = {spec.table for spec in manifest if "hnsw" in spec.name}

        assert indexed == {"vec_node", "vec_edge", "vec_episode"}

    def test_the_episode_index_uses_the_operator_class_of_its_dimension(self) -> None:
        """A space above the vector limit is indexed as a half vector."""
        specs = _specs()
        specs[VectorKind.EPISODE] = SpaceSpec(
            VectorKind.EPISODE, "mock", 2500, "episode.text", 1
        )

        manifest = build_manifest("m_t1", specs)
        episode = next(spec for spec in manifest if spec.name == "vec_episode_hnsw")

        assert "halfvec_cosine_ops" in episode.sql

    def test_the_build_parameters_are_part_of_the_index_definition(self) -> None:
        """An index set built with other parameters has other digests."""
        default = {spec.name: spec.digest for spec in build_manifest("m_t1", _specs())}
        other = {
            spec.name: spec.digest
            for spec in build_manifest(
                "m_t1", _specs(), HnswParams(m=32, ef_construction=128)
            )
        }

        assert default["vec_node_hnsw"] != other["vec_node_hnsw"]
