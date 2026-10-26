"""The embedding space registry. A space is its model, dimension and text template."""

from collections.abc import Mapping, Sequence

from ontogr_mem.vectorstore.embedder import EmbeddingGateway
from ontogr_mem.vectorstore.errors import SpaceMismatchError
from ontogr_mem.vectorstore.models import SpaceSpec, VectorKind
from ontogr_mem.vectorstore.protocol import Backend


class SpaceRegistry:
    """Creates missing spaces, then verifies every configured one.

    Args:
        backend: Where the spaces are stored.
        gateway: The embedder, probed once for its length and norm.
    """

    def __init__(self, backend: Backend, gateway: EmbeddingGateway) -> None:
        """Bind the registry to a backend and an embedder."""
        self._backend = backend
        self._gateway = gateway

    async def ensure(self, specs: Mapping[VectorKind, SpaceSpec]) -> None:
        """Create the missing spaces and compare the stored ones.

        Raises:
            SpaceMismatchError: If the probe or a stored space disagrees with the
                configuration.
        """
        await self._probe(specs)
        for kind, spec in specs.items():
            stored = await self._backend.read_space(kind)
            if stored is None:
                await self._backend.write_space(spec)
                stored = await self._backend.read_space(kind)
            if stored is None:
                msg = f"{kind.value}: space could not be stored"
                raise SpaceMismatchError(msg)
            self._compare(kind, spec, stored)

    async def _probe(self, specs: Mapping[VectorKind, SpaceSpec]) -> None:
        length, length_norm = await self._gateway.probe()
        if length != self._gateway.dim:
            msg = f"probe returned {length} values, embedder dim is {self._gateway.dim}"
            raise SpaceMismatchError(msg)
        expected: Sequence[int] = [spec.dim for spec in specs.values()]
        if length not in expected:
            msg = f"probe returned {length} values, expected one of {list(expected)}"
            raise SpaceMismatchError(msg)
        if length_norm == 0.0:
            msg = "probe returned a zero-norm vector"
            raise SpaceMismatchError(msg)

    @staticmethod
    def _compare(kind: VectorKind, want: SpaceSpec, have: SpaceSpec) -> None:
        for field, stored, configured in (
            ("model", have.model, want.model),
            ("dim", have.dim, want.dim),
            ("text_template", have.text_template, want.text_template),
            ("template_version", have.template_version, want.template_version),
        ):
            if stored != configured:
                msg = (
                    f"{kind.value}: stored {field} {stored!r}, "
                    f"configured {configured!r}"
                )
                raise SpaceMismatchError(msg)
