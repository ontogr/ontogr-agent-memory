"""An ``LLMClient`` over Ollama Cloud, for the live judge run only.

The key is read from ``OLLAMA_API_KEY`` at call time and is never stored on the
object or written anywhere. ``POST /api/chat`` is called with a bearer token,
``stream`` off, JSON output and temperature 0, so runs are repeatable.
"""

import asyncio
import json
import os
import urllib.error
import urllib.request

from ontogr_mem.llm import LLMRequest, LLMResponse


DEFAULT_MODEL = "gpt-oss:120b"
DEFAULT_URL = "https://ollama.com"


def available() -> bool:
    """Return whether a key is set in the environment."""
    return bool(os.environ.get("OLLAMA_API_KEY"))


class OllamaLLMClient:
    """Calls ``/api/chat`` in a worker thread so the event loop stays free.

    Args:
        model: Model name. Defaults to ``OLLAMA_MODEL`` or the built-in model.
        base_url: Endpoint. Defaults to ``OLLAMA_BASE_URL`` or the cloud service.
        timeout: Seconds to wait for one reply.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        """Read the model and endpoint from arguments or the environment."""
        self.model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
        url = base_url or os.environ.get("OLLAMA_BASE_URL", DEFAULT_URL)
        self._url = url.rstrip("/")
        self._timeout = timeout

    def _post(self, request: LLMRequest) -> LLMResponse:
        body = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.prompt},
            ],
        }
        http_request = urllib.request.Request(
            f"{self._url}/api/chat",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self._timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            msg = f"ollama http {exc.code}"
            raise RuntimeError(msg) from None
        return LLMResponse(
            text=data["message"]["content"],
            input_tokens=int(data.get("prompt_eval_count", 0)),
            output_tokens=int(data.get("eval_count", 0)),
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Run one completion in a worker thread."""
        return await asyncio.to_thread(self._post, request)
