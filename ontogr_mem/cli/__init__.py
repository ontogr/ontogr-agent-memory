"""The ontogr-mem console command."""

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence

from pydantic import ValidationError

from ontogr_mem.agent import build_default_agent
from ontogr_mem.cli.chat import run_chat
from ontogr_mem.client import Memory
from ontogr_mem.data_models.tenant import validate_tenant_id
from ontogr_mem.errors import AgentMemoryError, MissingExtraError
from ontogr_mem.llm import build_models
from ontogr_mem.llm.failures import ModelNotConfiguredError
from ontogr_mem.llm.roles import ModelRole
from ontogr_mem.vectorstore.builtin import EmbedderUnavailableError


async def _chat(tenant: str) -> None:
    """Keep the client open until chat and its drain have finished."""
    models = build_models()
    try:
        # Check the models before the graph and the embedder start, so a missing
        # setting does not first cost a model download.
        for role in (
            ModelRole.ANSWER,
            ModelRole.ENTITY_EXTRACTION,
            ModelRole.FACT_EXTRACTION,
        ):
            if not models.configured(role):
                raise ModelNotConfiguredError(f"no model configured for {role.value}")
        async with Memory.open(tenant=tenant, models=models) as memory:
            await run_chat(memory, build_default_agent(memory))
    finally:
        await models.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    """Parse a console command and return its process exit status.

    Args:
        argv: Arguments without the executable name, or the process arguments.

    Returns:
        Zero on normal exit, one on a runtime failure, or 130 on Ctrl-C.
        Argparse exits with its standard help and usage statuses.
    """
    parser = argparse.ArgumentParser(prog="ontogr-mem")
    commands = parser.add_subparsers(dest="command", required=True)
    chat = commands.add_parser("chat", help="chat with persistent memory")
    chat.add_argument("--tenant", default="default", type=validate_tenant_id)
    args = parser.parse_args(argv)
    os.environ["PYDANTIC_AI_NO_BANNER"] = "1"
    status = 0
    try:
        asyncio.run(_chat(args.tenant))
    except KeyboardInterrupt:
        status = 130
    except MissingExtraError as error:
        print(str(error), file=sys.stderr)
        status = 1
    except EmbedderUnavailableError as error:
        print(f"{error}\n{error.hint}", file=sys.stderr)
        status = 1
    except ModelNotConfiguredError:
        print(
            "Configure MEMORY_LLM_MODEL, MEMORY_LLM_BASE_URL and "
            "MEMORY_LLM_API_KEY for answers and text extraction.",
            file=sys.stderr,
        )
        status = 1
    except ValidationError:
        print(
            "Invalid memory settings; check the named MEMORY_* variables.",
            file=sys.stderr,
        )
        status = 1
    except AgentMemoryError:
        print(
            "Memory failed; check database, embedding and model settings.",
            file=sys.stderr,
        )
        status = 1
    except Exception:
        print(
            "Chat failed; check the model endpoint and memory services.",
            file=sys.stderr,
        )
        status = 1
    return status


__all__ = ["main"]
