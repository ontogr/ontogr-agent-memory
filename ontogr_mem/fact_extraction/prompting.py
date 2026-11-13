"""Request for one attempt of the fact pass.

`build_request` pairs a rendered prompt with the system message, the schema
chosen for the attempt and the output cap.
"""

from ontogr_mem.fact_extraction.prompts import SYSTEM, schema
from ontogr_mem.llm import LLMRequest, ModelSize


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
