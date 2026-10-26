"""Forget target selection and validation."""

from dataclasses import dataclass

from ontogr_mem.service.errors import InvalidArgumentError


@dataclass(frozen=True, slots=True)
class ForgetSelector:
    """One target used to resolve a forget plan.

    Attributes:
        episode_ids: Episode ids to forget.
        fact_id: Fact whose evidence episodes are selected.
        subject_ref: Subject reference stored in episode metadata.
        source_system: Source system whose episodes are selected.
    """

    episode_ids: tuple[str, ...] | None = None
    fact_id: str | None = None
    subject_ref: str | None = None
    source_system: str | None = None

    @property
    def kind(self) -> str:
        """Return the name of the selected target field."""
        validated = validate_selector(self)
        for name, value in (
            ("episode_ids", validated.episode_ids),
            ("fact_id", validated.fact_id),
            ("subject_ref", validated.subject_ref),
            ("source_system", validated.source_system),
        ):
            if value is not None:
                return name
        raise InvalidArgumentError("name exactly one forget target")

    @property
    def value(self) -> str | tuple[str, ...]:
        """Return the selected target value."""
        validated = validate_selector(self)
        for value in (
            validated.episode_ids,
            validated.fact_id,
            validated.subject_ref,
            validated.source_system,
        ):
            if value is not None:
                return value
        raise InvalidArgumentError("name exactly one forget target")


def validate_selector(selector: ForgetSelector) -> ForgetSelector:
    """Validate that exactly one target field has a non-blank value.

    Args:
        selector: Target fields supplied by the caller.

    Returns:
        The same selector when it names exactly one target.

    Raises:
        InvalidArgumentError: If the target is missing, ambiguous, or blank.
    """
    values = (
        selector.episode_ids,
        selector.fact_id,
        selector.subject_ref,
        selector.source_system,
    )
    if sum(value is not None for value in values) != 1:
        raise InvalidArgumentError("name exactly one forget target")

    value = next(value for value in values if value is not None)
    if isinstance(value, tuple):
        if not value or any(not item.strip() for item in value):
            raise InvalidArgumentError("forget target must not be blank")
    elif not value.strip():
        raise InvalidArgumentError("forget target must not be blank")
    return selector
