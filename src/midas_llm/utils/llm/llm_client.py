from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple
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
    constrained: bool = False  # True if response used constrained decoding


def send_to_llm(
        prompt: str,
        llm_model: str,
        llm_host: str,
        timeout_seconds: int = 300000,
        api_type: APIType | str = APIType.OLLAMA,
        json_schema: dict[str, Any] | None = None,
        logger: logging.Logger = LOGGER,
) -> LLMResponse:
    """Send prompt to LLM and return response.

    Args:
        prompt: The prompt text.
        llm_model: Model identifier (e.g., 'meta/llama-3.1-8b-instruct').
        llm_host: Host URL (e.g., 'http://localhost:8000').
        timeout_seconds: Request timeout.
        api_type: 'ollama' or 'openai_compatible'.
        json_schema: Optional JSON schema for constrained decoding.
            When provided:
              - NIM/OpenAI-compatible: uses response_format.json_schema
              - Ollama: uses the format parameter
            When None: unconstrained text generation (existing behavior).
        logger: Logger instance.

    Returns:
        LLMResponse with content, model name, and whether constrained
        decoding was used.
    """
    if not llm_host.startswith(("http://", "https://")):
        llm_host = f"http://{llm_host}"
    if isinstance(api_type, str):
        api_type = APIType(api_type.lower())

    logger.debug(
        "SENDING TO LLM | model=%s host=%s api=%s constrained=%s",
        llm_model, llm_host, api_type.value, json_schema is not None,
    )

    if api_type == APIType.OPENAI_COMPATIBLE:
        return _send_openai_compatible(
            prompt, llm_model, llm_host, timeout_seconds,
            json_schema=json_schema, logger=logger,
        )
    else:
        return _send_ollama(
            prompt, llm_model, llm_host, timeout_seconds,
            json_schema=json_schema, logger=logger,
        )


def _send_openai_compatible(
    prompt, llm_model, host, timeout_seconds,
    *, json_schema=None, logger=LOGGER,
):
    """Send to OpenAI-compatible API (vLLM, NVIDIA NIM, LiteLLM, etc.).

    If json_schema is provided, enables constrained decoding via
    response_format. NIM and vLLM both support this parameter.
    """
    url = f"{host}/v1/chat/completions"

    payload = {
        "model": llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 16384,
    }

    # ── Constrained decoding via JSON schema ──
    if json_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "midas_extraction",
                "strict": True,
                "schema": json_schema,
            },
        }
        logger.info(
            "Constrained decoding ENABLED (json_schema with %d properties)",
            len(json_schema.get("properties", {})),
        )

    logger.debug("Prompt length: %d chars", len(prompt))
    llm_start = time.time()

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.post(url, json=payload)

            # If NIM rejects json_schema, fall back to json_object
            if resp.status_code == 400 and json_schema is not None:
                error_text = resp.text[:500]
                if "json_schema" in error_text.lower() or "response_format" in error_text.lower():
                    logger.warning(
                        "Server rejected json_schema format, "
                        "falling back to json_object mode: %s",
                        error_text[:200],
                    )
                    payload["response_format"] = {"type": "json_object"}
                    resp = client.post(url, json=payload)

            # If json_object also fails, fall back to unconstrained
            if resp.status_code == 400 and json_schema is not None:
                error_text = resp.text[:500]
                logger.warning(
                    "Server rejected json_object format, "
                    "falling back to unconstrained: %s",
                    error_text[:200],
                )
                payload.pop("response_format", None)
                resp = client.post(url, json=payload)

            resp.raise_for_status()
            data = resp.json()

    except httpx.TimeoutException as e:
        raise TimeoutError(f"LLM call exceeded {timeout_seconds}s.") from e
    except httpx.HTTPStatusError as e:
        logger.error(
            "HTTP error: %s - %s",
            e.response.status_code, e.response.text[:500],
        )
        raise

    llm_elapsed = time.time() - llm_start
    choices = data.get("choices", [])
    content = (
        choices[0].get("message", {}).get("content", "")
        if choices else ""
    )
    content = content.strip()

    # Detect whether constrained decoding was actually used
    used_constrained = (
        json_schema is not None
        and "response_format" in payload
    )

    usage = data.get("usage", {})
    logger.debug(
        "Response in %.2fs (%d chars, %d prompt tokens, %d output tokens, constrained=%s)",
        llm_elapsed, len(content),
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
        used_constrained,
    )

    return LLMResponse(
        content=content,
        model=llm_model,
        done=True,
        constrained=used_constrained,
    )


def _send_ollama(
    prompt, llm_model, host, timeout_seconds,
    *, json_schema=None, logger=LOGGER,
):
    """Send to Ollama native API.

    If json_schema is provided, passes it via the 'format' parameter
    for Ollama's structured output support (v0.5+).
    """
    url = f"{host}/api/chat"

    payload = {
        "model": llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_ctx": 8192,
            "num_predict": 2048,
        },
    }

    # ── Constrained decoding via Ollama format parameter ──
    if json_schema is not None:
        payload["format"] = json_schema
        logger.info(
            "Ollama constrained decoding ENABLED (format schema with %d properties)",
            len(json_schema.get("properties", {})),
        )

    logger.debug("Prompt length: %d chars", len(prompt))
    llm_start = time.time()

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.post(url, json=payload)

            # If Ollama rejects the schema (old version), fall back
            if resp.status_code == 400 and json_schema is not None:
                error_text = resp.text[:500]
                logger.warning(
                    "Ollama rejected format schema (upgrade to v0.5+), "
                    "falling back to format='json': %s",
                    error_text[:200],
                )
                payload["format"] = "json"
                resp = client.post(url, json=payload)

            # If format='json' also fails, fall back to unconstrained
            if resp.status_code == 400 and json_schema is not None:
                logger.warning(
                    "Ollama rejected format='json', "
                    "falling back to unconstrained."
                )
                payload.pop("format", None)
                resp = client.post(url, json=payload)

            resp.raise_for_status()
            data = resp.json()

    except httpx.TimeoutException as e:
        raise TimeoutError(f"LLM call exceeded {timeout_seconds}s.") from e
    except httpx.HTTPStatusError as e:
        logger.error(
            "Ollama HTTP error: %s - %s",
            e.response.status_code, e.response.text[:500],
        )
        raise

    llm_elapsed = time.time() - llm_start

    if "message" in data:
        content = data.get("message", {}).get("content", "")
    else:
        content = data.get("response", "")

    chatml_tokens = [
        "<|im_start|>", "<|im_end|>",
        "<|im_start|>assistant", "<|im_end|>assistant",
    ]
    for token in chatml_tokens:
        content = content.replace(token, "")
    content = content.strip()

    used_constrained = json_schema is not None and "format" in payload

    logger.debug(
        "Response in %.2fs (%d chars, constrained=%s)",
        llm_elapsed, len(content), used_constrained,
    )

    return LLMResponse(
        content=content,
        model=llm_model,
        done=data.get("done", True),
        constrained=used_constrained,
    )


def test_respond_ok(
        llm_model: str,
        llm_host: str,
        timeout_seconds: int = 60,
        *,
        api_type: APIType | str = APIType.OLLAMA,
        logger: logging.Logger = LOGGER,
) -> Tuple[bool, str]:
    try:
        resp = send_to_llm(
            "Say hello.", llm_model, llm_host, timeout_seconds,
            api_type=api_type, logger=logger,
        )
        text = resp.content or ""
        return bool(text.strip()), text
    except Exception as exc:
        logger.error("Sanity ping failed: %s", exc)
        return False, f"ERROR: {exc}"