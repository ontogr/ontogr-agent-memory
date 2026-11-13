FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.23@sha256:61d393e44e249f2e4b526b6c7ddcecce245946826e608e11c93ad4f5bba55b21 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY ontogr_mem ./ontogr_mem
ENV UV_PYTHON_DOWNLOADS=never UV_NO_CACHE=1 UV_LINK_MODE=copy UV_HTTP_TIMEOUT=120
RUN uv sync --locked --no-dev --no-editable \
        --extra falkordb --extra fastembed --extra mcp \
    && useradd --create-home --uid 10001 memory \
    && mkdir -p /data/state /data/models \
    && chown -R memory:memory /data
USER memory
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYDANTIC_AI_NO_BANNER=1 \
    MEMORY_STATE_SQLITE_PATH=/data/state/agent_memory_state.sqlite3 \
    FASTEMBED_CACHE_PATH=/data/models
ENTRYPOINT ["ontogr-mem"]
CMD ["mcp", "--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
