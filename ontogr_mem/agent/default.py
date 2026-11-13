"""Build the chat agent with the answer model and memory support."""

from typing import Any

from pydantic_ai import Agent

from ontogr_mem.agent.capability import MemoryCapability
from ontogr_mem.agent.instructions import DEFAULT_INSTRUCTIONS
from ontogr_mem.client import Memory
from ontogr_mem.llm.failures import ModelNotConfiguredError
from ontogr_mem.llm.roles import ModelRole
from ontogr_mem.llm.run import OUTPUT_RETRIES, role_model_settings


def build_default_agent(
    memory: Memory,
    *,
    capability: MemoryCapability | None = None,
    instructions: str | None = None,
) -> Agent[Any, str]:
    """Build a text-answer agent over an open Memory client.

    Args:
        memory: Client whose answer Model role supplies the model and settings.
        capability: Optional memory configuration, using this same client.
        instructions: Replacement answer instructions, or the default text.

    Returns:
        An agent whose caller owns the Memory client's lifetime.

    Raises:
        ModelNotConfiguredError: If no answer model is configured.
        ValueError: If the supplied capability uses a different client.
    """
    model = memory.models.model_for(ModelRole.ANSWER)
    if model is None:
        raise ModelNotConfiguredError("no model configured for answer")
    if capability is not None and capability.memory is not memory:
        raise ValueError("capability must use the supplied Memory client")
    config = memory.models.config_for(ModelRole.ANSWER)
    return Agent(
        model=model,
        output_type=str,
        name="assistant",
        instructions=DEFAULT_INSTRUCTIONS if instructions is None else instructions,
        model_settings=role_model_settings(config, config.max_output_tokens),
        retries={"output": OUTPUT_RETRIES},
        capabilities=[capability or MemoryCapability(memory)],
    )
