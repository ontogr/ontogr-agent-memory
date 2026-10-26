"""Building the model's request for one episode.

The prompt grows with the number of earlier episodes, so it is fitted under the
gate's input cap before every call: the oldest earlier episodes go first, and
the current episode is never cut.

The attempts loosen the request. The first sends the strict schema, the second
the loose one (providers that reject strict schemas), the third no schema at
all.
"""

from collections.abc import Callable, Sequence

from ontogr_mem.data_models import Episode, ExtractedEntity, TypeView
from ontogr_mem.fact_extraction.prompts import SYSTEM, build_prompt, schema
from ontogr_mem.llm import InputTooLargeError, LLMRequest, ModelSize


def fit_prompt(
    episode: Episode,
    candidates: Sequence[ExtractedEntity],
    view: TypeView | None,
    previous: Sequence[Episode],
    repair: tuple[str, list[str]] | None,
    *,
    max_input_tokens: int,
    instructions: str,
    count_tokens: Callable[[str], int],
) -> str:
    """Render the prompt, dropping the oldest previous episodes to fit.

    The current episode is never truncated. Previous episodes are dropped
    oldest first until the prompt fits.

    Args:
        episode: The episode to extract from.
        candidates: The entity ids the model may use as refs.
        view: The type view the model used, or None.
        previous: Earlier episodes, oldest first.
        repair: The previous invalid reply and its errors, for a repair attempt.
        max_input_tokens: Hard cap on system plus prompt tokens.
        instructions: Extra instructions from settings.
        count_tokens: Counts the tokens of a piece of text.

    Returns:
        The prompt.

    Raises:
        InputTooLargeError: If the prompt does not fit even without previous
            episodes.
    """
    total = 0
    for count in range(len(previous), -1, -1):
        selected = previous[len(previous) - count :]
        prompt = build_prompt(
            episode,
            candidates,
            view,
            selected,
            instructions=instructions,
            repair=repair,
        )
        total = count_tokens(SYSTEM) + count_tokens(prompt)
        if total <= max_input_tokens:
            return prompt
    msg = f"episode needs {total} tokens, cap is {max_input_tokens}"
    raise InputTooLargeError(msg)


def build_request(
    attempt: int,
    prompt: str,
    *,
    model_size: ModelSize,
    max_output_tokens: int,
    with_types: bool,
) -> LLMRequest:
    """Build the request for one attempt.

    Attempt 1 sends the strict schema, attempt 2 the loose one, attempt 3 no
    schema at all.

    Args:
        attempt: The attempt number, starting at zero.
        prompt: The rendered prompt.
        model_size: Tier of model to use.
        max_output_tokens: Cap on generated tokens.
        with_types: Include the dense relation type id in the schema.

    Returns:
        The request.
    """
    return LLMRequest(
        system=SYSTEM,
        prompt=prompt,
        model_size=model_size,
        schema=(
            schema(with_types=with_types, strict=attempt == 0) if attempt < 2 else None
        ),
        max_output_tokens=max_output_tokens,
    )
