"""Apply the ontology a host gives when it opens memory."""

from ontogr_mem.ontology.declaration import OntologyDeclaration
from ontogr_mem.ontology.default import default_ontology
from ontogr_mem.ontology.errors import InvalidChange
from ontogr_mem.ontology.manager import OntologyManager
from ontogr_mem.ontology.models import Ontology
from ontogr_mem.service.errors import InvalidArgumentError


async def apply_ontology(
    manager: OntologyManager, declaration: OntologyDeclaration | None
) -> Ontology:
    """Apply a declaration, or initialize a new tenant with the default.

    Raises:
        InvalidArgumentError: The declaration does not apply to the stored ontology.
    """
    try:
        if declaration is not None:
            return await manager.declare(declaration)
        head = await manager.catalog()
        if head.version == 0:
            return await manager.declare(default_ontology(), only_if_empty=True)
        return head
    except InvalidChange as exc:
        raise InvalidArgumentError(f"the ontology does not apply: {exc}") from exc
