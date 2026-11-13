FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY ontogr_mem ./ontogr_mem
RUN pip install --no-cache-dir ".[falkordb,fastembed,mcp]" \
    && useradd --create-home --uid 10001 memory \
    && mkdir -p /data/state /data/models \
    && chown -R memory:memory /data
USER memory
ENV PYTHONUNBUFFERED=1 \
    PYDANTIC_AI_NO_BANNER=1 \
    MEMORY_STATE_SQLITE_PATH=/data/state/agent_memory_state.sqlite3 \
    FASTEMBED_CACHE_PATH=/data/models
ENTRYPOINT ["ontogr-mem"]
CMD ["mcp", "--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
