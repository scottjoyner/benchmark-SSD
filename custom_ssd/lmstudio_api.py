from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


class LMStudioAPIError(RuntimeError):
    pass


@dataclass(slots=True)
class CompletionResult:
    text: str
    elapsed_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LMStudioAPI:
    def __init__(self, base_url: str = "http://127.0.0.1:1234", timeout: int = 180) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get_json(self, url: str) -> dict[str, object]:
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise LMStudioAPIError(f"Failed to reach LM Studio at {url}: {exc}") from exc

    def _post_json(self, url: str, payload: dict[str, object]) -> dict[str, object]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise LMStudioAPIError(f"LM Studio HTTP {exc.code} for {url}: {details}") from exc
        except urllib.error.URLError as exc:
            raise LMStudioAPIError(f"Failed to reach LM Studio at {url}: {exc}") from exc

    def list_models(self) -> list[str]:
        payload = self._get_json(f"{self.base_url}/v1/models")
        return [item["id"] for item in payload.get("data", [])]

    def chat_completion(
        self,
        model: str,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> CompletionResult:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        start = time.perf_counter()
        response = self._post_json(f"{self.base_url}/v1/chat/completions", payload)
        elapsed = time.perf_counter() - start
        usage = response.get("usage", {})
        text = response["choices"][0]["message"]["content"]
        completion_tokens = int(usage.get("completion_tokens", 0))
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
        return CompletionResult(
            text=text,
            elapsed_seconds=elapsed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
