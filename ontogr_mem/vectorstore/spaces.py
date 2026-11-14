"""The embedding space registry. A space is its model, dimension and text template."""

import math
from collections.abc import Mapping, Sequence

from ontogr_mem.vectorstore.embedder import EmbeddingGateway, EmbeddingPurpose
from ontogr_mem.vectorstore.errors import SpaceMismatchError
from ontogr_mem.vectorstore.models import SpaceSpec, VectorKind
from ontogr_mem.vectorstore.protocol import Backend


class SpaceRegistry:
    """Creates missing spaces, then verifies every configured one.

    Args:
        backend: Where the spaces are stored.
        gateway: Provider checked for both purposes, dimensions, and finite norms.
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
        expected: Sequence[int] = [spec.dim for spec in specs.values()]
        purposes: tuple[EmbeddingPurpose, ...] = ("document", "query")
        for purpose in purposes:
            length, length_norm = await self._gateway.probe(purpose=purpose)
            if length != self._gateway.dim:
                msg = (
                    f"{purpose} probe returned {length} values, "
                    f"embedder dim is {self._gateway.dim}"
                )
                raise SpaceMismatchError(msg)
            if length not in expected:
                msg = (
                    f"{purpose} probe returned {length} values, "
                    f"expected one of {list(expected)}"
                )
                raise SpaceMismatchError(msg)
            if not math.isfinite(length_norm) or length_norm <= 0.0:
                msg = f"{purpose} probe returned a non-finite or zero-norm vector"
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
                if field == "model":
                    msg = (
                        f"{kind.value}: Existing memory uses a different embedding "
                        "model or encoding behavior. Changing the model or encoding "
                        "requires re-embedding the source data into a new Tenant. "
                        "Automatic re-embedding is not supported."
                    )
                else:
                    mismatch = {
                        "dim": ("embedding dimension", "embedding dimension"),
                        "text_template": ("text template", "text template"),
                        "template_version": (
                            "text template version",
                            "text template",
                        ),
                    }[field]
                    msg = (
                        f"{kind.value}: Existing memory uses a different "
                        f"{mismatch[0]}. Changing the {mismatch[1]} requires "
                        "re-embedding the source data into a new Tenant. "
                        "Automatic re-embedding is not supported."
                    )
                raise SpaceMismatchError(msg)
