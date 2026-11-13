"""Packaged skill text served by get_skill, and the server instructions."""

import pytest
from fastmcp import Client, FastMCP
from fastmcp.skills import SkillsClientExtension

from ontogr_mem.mcp_server.instructions import INSTRUCTIONS
from ontogr_mem.mcp_server.server import build_server
from ontogr_mem.mcp_server.settings import ALL_TOOLS, McpSettings
from ontogr_mem.mcp_server.skills import SKILLS_ROOT, get_skill
from ontogr_mem.service.errors import InvalidArgumentError
from tests.support.mcp_memory import FakeMemoryService, build_memory


SKILL_NAMES = ["memory-recall", "memory-write"]

SKILL_URIS = [f"skill://{name}/SKILL.md" for name in SKILL_NAMES]


def _server(*, tools: tuple[str, ...] = ALL_TOOLS, skills_extension: bool) -> FastMCP:
    """Build a server over a fake service with the given tools and extension."""
    settings = McpSettings(
        _env_file=None, tools=tools, api_key=None, skills_extension=skills_extension
    )
    return build_server(build_memory(FakeMemoryService()), settings=settings)


class TestGetSkill:
    """get_skill returns packaged SKILL.md text and rejects other names."""

    @pytest.mark.parametrize("name", ["memory-write", "memory-recall"])
    async def test_returns_the_packaged_skill_text(self, name):
        """Each packaged skill returns its SKILL.md file unchanged."""
        expected = (SKILLS_ROOT / name / "SKILL.md").read_text(encoding="utf-8")

        text = await get_skill(name)

        assert text == expected
        assert text.startswith(f"---\nname: {name}")

    @pytest.mark.parametrize(
        "name", ["nope", "../memory-write", "memory-write/../../settings", ""]
    )
    async def test_rejects_unknown_and_traversal_names(self, name):
        """Names other than the two packaged skills raise InvalidArgumentError."""
        with pytest.raises(InvalidArgumentError):
            await get_skill(name)

    def test_instructions_name_both_skills(self):
        """The server instructions point clients at both skill names."""
        assert "memory-write" in INSTRUCTIONS
        assert "memory-recall" in INSTRUCTIONS


class TestSkillsThroughServer:
    """The built server serves the packaged skills as a tool and as an extension."""

    @pytest.mark.parametrize("name", SKILL_NAMES)
    async def test_get_skill_tool_returns_the_packaged_file_bytes(self, name):
        """The get_skill tool returns each packaged SKILL.md byte for byte."""
        expected = (SKILLS_ROOT / name / "SKILL.md").read_bytes()
        server = _server(skills_extension=False)

        async with Client(server) as client:
            result = await client.call_tool("get_skill", {"name": name})

        text = "".join(block.text for block in result.content)
        assert text.encode("utf-8") == expected

    async def test_extension_lists_both_skills(self):
        """With the extension on, a skills client lists the two skill URIs."""
        server = _server(skills_extension=True)

        async with Client(server, extensions=[SkillsClientExtension()]) as client:
            skills = await client.list_skills()

        assert sorted(skill.uri for skill in skills) == SKILL_URIS

    @pytest.mark.parametrize("name", SKILL_NAMES)
    async def test_extension_serves_the_packaged_file_text(self, name):
        """Reading a skill URI returns the text of the packaged SKILL.md."""
        expected = (SKILLS_ROOT / name / "SKILL.md").read_text(encoding="utf-8")
        server = _server(skills_extension=True)

        async with Client(server, extensions=[SkillsClientExtension()]) as client:
            contents = await client.read_resource(f"skill://{name}/SKILL.md")

        assert "".join(content.text for content in contents) == expected

    async def test_extension_off_refuses_skill_listing_but_keeps_the_tool(self):
        """With the extension off, listing skills is refused and get_skill works."""
        name = "memory-write"
        expected = (SKILLS_ROOT / name / "SKILL.md").read_text(encoding="utf-8")
        server = _server(skills_extension=False)

        async with Client(server, extensions=[SkillsClientExtension()]) as client:
            with pytest.raises(RuntimeError, match="does not advertise the Skills"):
                await client.list_skills()
            result = await client.call_tool("get_skill", {"name": name})

        assert "".join(block.text for block in result.content) == expected

    async def test_extension_serves_skills_without_the_get_skill_tool(self):
        """Dropping get_skill from the tools leaves the extension serving skills."""
        tools = tuple(name for name in ALL_TOOLS if name != "get_skill")
        server = _server(tools=tools, skills_extension=True)

        async with Client(server, extensions=[SkillsClientExtension()]) as client:
            tool_names = [tool.name for tool in await client.list_tools()]
            skills = await client.list_skills()

        assert tool_names == [
            "search_memory",
            "get_entity",
            "get_job",
            "write_facts",
            "forget_memory",
        ]
        assert sorted(skill.uri for skill in skills) == SKILL_URIS
