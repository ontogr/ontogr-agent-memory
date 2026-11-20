FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535
COPY --from=ghcr.io/astral-sh/uv:0.12.23@sha256:61d393e44e249f2e4b526b6c7ddcecce245946826e608e11c93ad4f5bba55b21 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY ontogr_mem ./ontogr_mem
ENV UV_PYTHON_DOWNLOADS=never UV_NO_CACHE=1 UV_LINK_MODE=copy UV_HTTP_TIMEOUT=120
RUN uv sync --locked --no-dev --no-editable \
        --extra local --extra neo4j --extra postgres \
    && useradd --create-home --uid 10001 memory \
    && mkdir -p /data/state /data/models \
    && chown -R memory:memory /data
USER memory
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYDANTIC_AI_NO_BANNER=1 \
    MEMORY_STATE_URL=sqlite:////data/state/agent_memory_state.sqlite3 \
    MEMORY_DATA_DIR=/data/state \
    FASTEMBED_CACHE_PATH=/data/models \
    MEMORY_SERVER_IDLE_MINUTES=0 \
    MEMORY_MCP_HOST=0.0.0.0 \
    MEMORY_MCP_PORT=8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=6 \
    CMD ["/app/.venv/bin/python", "-c", "import os, urllib.request; key = os.environ.get('MEMORY_MCP_API_KEY'); headers = {'Authorization': f'Bearer {key}'} if key and key.strip() else {}; request = urllib.request.Request('http://127.0.0.1:8000/health', headers=headers); urllib.request.urlopen(request, timeout=4).close()"]
ENTRYPOINT ["ontogr-mem"]
CMD ["server", "run"]
