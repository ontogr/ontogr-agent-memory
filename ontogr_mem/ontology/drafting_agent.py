"""Typed additive ontology proposals."""

from pydantic_ai import Agent

from ontogr_mem.ontology.proposals import Draft


drafting_agent: Agent[None, Draft] = Agent(
    output_type=Draft,
    instructions="You extend an ontology with additive operations only.",
)
