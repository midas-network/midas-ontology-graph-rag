from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple
import time
import logging
import httpx

LOGGER = logging.getLogger("midas-llm")


@dataclass
class LLMResponse:
    """Simple response wrapper."""
    content: str
    model: str
    done: bool = True


def send_to_llm(
    prompt: str,
    llm_model: str,
    ollama_host: str,
    timeout_seconds: int = 300,
    *,
    logger: logging.Logger = LOGGER,
) -> LLMResponse:
    if not ollama_host.startswith(("http://", "https://")):
        ollama_host = f"http://{ollama_host}"

    logger.debug("SENDING TO LLM | model=%s host=%s timeout=%ss", llm_model, ollama_host, timeout_seconds)

    # Use /api/chat endpoint - works with most instruction-tuned models
    url = f"{ollama_host}/api/chat"
    payload = {
        "model": llm_model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_ctx": 8192,
            "num_predict": 2048,
        },
    }

    logger.debug("Sending request to LLM (timeout %d seconds)...", timeout_seconds)
    logger.debug("Prompt length: %d chars", len(prompt))
    llm_start = time.time()

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException as e:
        raise TimeoutError(
            f"LLM call exceeded {timeout_seconds}s. "
            "Check Ollama load/queue, server reachability, or try a smaller model."
        ) from e
    except httpx.HTTPStatusError as e:
        logger.error("Ollama HTTP error: %s - %s", e.response.status_code, e.response.text[:500])
        raise

    llm_elapsed = time.time() - llm_start

    # Handle both /api/generate and /api/chat response formats
    if "message" in data:
        # /api/chat format
        message = data.get("message", {})
        content = message.get("content", "")
    else:
        # /api/generate format
        content = data.get("response", "")

    # Clean up ChatML tokens that may leak into output
    for token in ["<|im_start|>", "<|im_end|>", "</im_start>", "</im_end|>", "<|im_start|>assistant", "<|im_end|>assistant"]:
        content = content.replace(token, "")
    content = content.strip()

    eval_count = data.get("eval_count", 0)
    prompt_eval_count = data.get("prompt_eval_count", 0)

    logger.debug("LLM response received in %.2fs (%d chars, %d prompt tokens, %d output tokens)",
                llm_elapsed, len(content), prompt_eval_count, eval_count)

    # Warn if response seems too fast or empty (but not for expected short responses)
    content_upper = content.upper().strip()
    is_expected_short = content_upper in ("MATCH", "NO_MATCH", "YES", "NO", "OK", "HELLO", "HELLO!", "HI")
    if llm_elapsed < 1.0 and len(content) < 50 and not is_expected_short:
        logger.warning("Response suspiciously fast (%.2fs) with little content (%d chars). Model may be failing.", llm_elapsed, len(content))
        logger.warning("Raw Ollama response: %s", data)

    return LLMResponse(content=content, model=llm_model, done=data.get("done", True))


def test_respond_ok(
    llm_model: str,
    ollama_host: str,
    timeout_seconds: int = 60,
    *,
    logger: logging.Logger = LOGGER,
) -> Tuple[bool, str]:
    """Send a minimal prompt and return (is_ok, raw_response or error).

    Considered OK if the LLM returns any non-empty response (not just 'OK').
    """
    try:
        resp = send_to_llm(
            prompt="Say hello.",
            llm_model=llm_model,
            ollama_host=ollama_host,
            timeout_seconds=timeout_seconds,
            logger=logger,
        )
        text = resp.content or ""
        # Any non-empty response means the LLM is responding
        is_ok = bool(text.strip())
        return is_ok, text
    except Exception as exc:  # noqa: BLE001
        logger.error("Sanity ping failed: %s", exc)
        return False, f"ERROR: {exc}"
        return False, f"ERROR: {exc}"
