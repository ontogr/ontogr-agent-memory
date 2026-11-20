<h1 align="center"><a href="https://github.com/ontogr/ontogr-agent-memory">OntoGr Agent Memory</a></h1>

<h3 align="center">Durable, traceable memory for AI agents.</h3>

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/ontogr-mem.svg)](https://pypi.org/project/ontogr-mem/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/ontogr-mem/)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ontogr/ontogr-agent-memory)
[![License](https://img.shields.io/github/license/ontogr/ontogr-agent-memory?color=green)](https://github.com/ontogr/ontogr-agent-memory/blob/main/LICENSE)

[![CI](https://img.shields.io/github/actions/workflow/status/ontogr/ontogr-agent-memory/ci.yml?branch=main&label=CI&logo=githubactions)](https://github.com/ontogr/ontogr-agent-memory/actions/workflows/ci.yml)
[![Integration tests](https://img.shields.io/github/actions/workflow/status/ontogr/ontogr-agent-memory/integration.yml?branch=main&label=integration%20tests&logo=pytest)](https://github.com/ontogr/ontogr-agent-memory/actions/workflows/integration.yml)
[![Codecov](https://codecov.io/gh/ontogr/ontogr-agent-memory/branch/main/graph/badge.svg)](https://codecov.io/gh/ontogr/ontogr-agent-memory)

</div>

OntoGr Agent Memory stores what an agent learned, where it learned it, and when
it was true. It ingests source material as episodes, extracts entities and
facts, resolves duplicate entities, reconciles facts on two time axes, and
returns a budgeted context block that is ready for an answering model.

<p align="center">
  <img src="https://raw.githubusercontent.com/ontogr/ontogr-agent-memory/main/img/hero.svg" alt="Source items enter the extraction layer as episodes. Extracted facts go to the temporal and consistency engine, which writes fact versions to the storage layer. Retrieval reads the storage layer and returns a budgeted context block." width="860">
</p>

Install the package with all backends:

```bash
uv add "ontogr-mem[local,neo4j,postgres]"
```

## Pick your path

| Path | What you set up |
| --- | --- |
| [Claude Code](https://ontogr.github.io/ontogr-agent-memory/latest/get-started/claude-code/) | Add the memory plugin, then write and recall a fact. |
| [Other hosts](https://ontogr.github.io/ontogr-agent-memory/latest/get-started/other-hosts/) | Connect Codex, Cursor, or OpenCode to the memory server. |
| [Python app](https://ontogr.github.io/ontogr-agent-memory/latest/get-started/python-app/) | Write and search memory from an async application. |
| [Shared server and Docker](https://ontogr.github.io/ontogr-agent-memory/latest/get-started/shared-server-and-docker/) | Run one server for a machine or a container for a team. |
| [Terminal chat](https://ontogr.github.io/ontogr-agent-memory/latest/get-started/chat/) | Try memory in a terminal before connecting a host. |

The full documentation is at
[ontogr.github.io/ontogr-agent-memory](https://ontogr.github.io/ontogr-agent-memory/).

Read the [contributing guide](https://github.com/ontogr/ontogr-agent-memory/blob/main/CONTRIBUTING.md)
before you submit a pull request. Rules for coding agents are in
[AGENTS.md](https://github.com/ontogr/ontogr-agent-memory/blob/main/AGENTS.md).

OntoGr Agent Memory is licensed under the
[Apache License 2.0](https://github.com/ontogr/ontogr-agent-memory/blob/main/LICENSE).
