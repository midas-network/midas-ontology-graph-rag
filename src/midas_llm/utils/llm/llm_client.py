from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
from enum import Enum
import time
import logging
import httpx

LOGGER = logging.getLogger("midas-llm")


class APIType(Enum):
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"


@dataclass
class LLMResponse:
    content: str
    model: str
    done: bool = True


def send_to_llm(
        prompt: str,
        llm_model: str,
        llm_host: str,
        timeout_seconds: int = 300000,
        api_type: APIType | str = APIType.OLLAMA,
        logger: logging.Logger = LOGGER,
) -> LLMResponse:
    if not llm_host.startswith(("http://", "https://")):
        llm_host = f"http://{llm_host}"
    if isinstance(api_type, str):
        api_type = APIType(api_type.lower())

    logger.debug("SENDING TO LLM | model=%s host=%s api=%s", llm_model, llm_host, api_type.value)

    if api_type == APIType.OPENAI_COMPATIBLE:
        return _send_openai_compatible(prompt, llm_model, llm_host, timeout_seconds, logger)
    else:
        return _send_ollama(prompt, llm_model, llm_host, timeout_seconds, logger)


def _send_openai_compatible(prompt, llm_model, host, timeout_seconds, logger):
    """Send to OpenAI-compatible API (vLLM, NVIDIA NIM, LiteLLM, LocalAI, etc.)."""
    url = f"{host}/v1/chat/completions"
    payload = {
        "model": llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 16384,
    }
    logger.debug("Prompt length: %d chars", len(prompt))
    llm_start = time.time()

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException as e:
        raise TimeoutError(f"LLM call exceeded {timeout_seconds}s.") from e
    except httpx.HTTPStatusError as e:
        logger.error("HTTP error: %s - %s", e.response.status_code, e.response.text[:500])
        raise

    llm_elapsed = time.time() - llm_start
    choices = data.get("choices", [])
    content = choices[0].get("message", {}).get("content", "") if choices else ""
    content = content.strip()

    usage = data.get("usage", {})
    logger.debug("Response in %.2fs (%d chars, %d prompt tokens, %d output tokens)",
                 llm_elapsed, len(content), usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))

    return LLMResponse(content=content, model=llm_model, done=True)


def _send_ollama(prompt, llm_model, host, timeout_seconds, logger):
    """Send to Ollama native API."""
    url = f"{host}/api/chat"
    payload = {
        "model": llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.7, "num_ctx": 8192, "num_predict": 2048},
    }
    logger.debug("Prompt length: %d chars", len(prompt))
    llm_start = time.time()

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException as e:
        raise TimeoutError(f"LLM call exceeded {timeout_seconds}s.") from e
    except httpx.HTTPStatusError as e:
        logger.error("Ollama HTTP error: %s - %s", e.response.status_code, e.response.text[:500])
        raise

    llm_elapsed = time.time() - llm_start
    if "message" in data:
        content = data.get("message", {}).get("content", "")
    else:
        content = data.get("response", "")

    chatml_tokens = ["<|im_start|>", "<|im_end|>", "<|im_start|>assistant", "<|im_end|>assistant"]
    for token in chatml_tokens:
        content = content.replace(token, "")
    content = content.strip()

    logger.debug("Response in %.2fs (%d chars)", llm_elapsed, len(content))
    return LLMResponse(content=content, model=llm_model, done=data.get("done", True))


def test_respond_ok(
        llm_model: str,
        llm_host: str,
        timeout_seconds: int = 60,
        *,
        api_type: APIType | str = APIType.OLLAMA,
        logger: logging.Logger = LOGGER,
) -> Tuple[bool, str]:
    try:
        resp = send_to_llm("Say hello.", llm_model, llm_host, timeout_seconds, api_type=api_type, logger=logger)
        text = resp.content or ""
        return bool(text.strip()), text
    except Exception as exc:
        logger.error("Sanity ping failed: %s", exc)
        return False, f"ERROR: {exc}"