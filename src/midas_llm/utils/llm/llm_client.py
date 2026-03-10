from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal, Tuple
from enum import Enum
import time
import logging
import httpx

LOGGER = logging.getLogger("midas-llm")
OpenAIResponseFormatMode = Literal["json_schema", "json_object", "none"]
_OPENAI_RESPONSE_FORMAT_MODE_BY_HOST: dict[str, OpenAIResponseFormatMode] = {}


class APIType(Enum):
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"


@dataclass
class LLMResponse:
    content: str
    model: str
    done: bool = True
    constrained: bool = False  # True if response used constrained decoding
    request_duration_s: float | None = None
    total_duration_s: float | None = None
    load_duration_s: float | None = None
    prompt_eval_duration_s: float | None = None
    eval_duration_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    usage: dict[str, Any] | None = None


def _ns_to_seconds(value: Any) -> float | None:
    """Convert provider nanosecond duration fields to seconds."""
    if isinstance(value, (int, float)):
        return float(value) / 1_000_000_000.0
    return None


def _response_format_error(text: str) -> bool:
    """Detect OpenAI-compatible errors about unsupported response_format."""
    lowered = text.lower()
    hints = ("response_format", "json_schema", "json_object")
    return any(hint in lowered for hint in hints)


def _set_openai_response_format_mode(
    host: str,
    mode: OpenAIResponseFormatMode,
    *,
    logger: logging.Logger,
    reason: str,
) -> None:
    """Cache the best-known response_format mode for this host."""
    host_key = host.rstrip("/")
    previous = _OPENAI_RESPONSE_FORMAT_MODE_BY_HOST.get(host_key)
    _OPENAI_RESPONSE_FORMAT_MODE_BY_HOST[host_key] = mode
    if previous != mode:
        logger.warning(
            "OpenAI-compatible host %s response_format mode set to %s (%s)",
            host_key, mode, reason,
        )


def _set_openai_response_format(
    payload: dict[str, Any],
    mode: OpenAIResponseFormatMode,
    json_schema: dict[str, Any],
) -> None:
    if mode == "json_schema":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "midas_extraction",
                "strict": True,
                "schema": json_schema,
            },
        }
    elif mode == "json_object":
        payload["response_format"] = {"type": "json_object"}
    else:
        payload.pop("response_format", None)


def send_to_llm(
        prompt: str,
        llm_model: str,
        llm_host: str,
        timeout_seconds: int = 300000,
        api_type: APIType | str = APIType.OLLAMA,
        json_schema: dict[str, Any] | None = None,
        allow_json_schema: bool = False,
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
              - NIM/OpenAI-compatible: uses response_format.json_object by default
                (or response_format.json_schema if allow_json_schema=True)
              - Ollama: uses the format parameter
            When None: unconstrained text generation (existing behavior).
        allow_json_schema: For OpenAI-compatible APIs, only attempt
            response_format.type="json_schema" when True. Default False
            uses response_format.type="json_object".
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
            json_schema=json_schema, allow_json_schema=allow_json_schema, logger=logger,
        )
    else:
        return _send_ollama(
            prompt, llm_model, llm_host, timeout_seconds,
            json_schema=json_schema, logger=logger,
        )


def _send_openai_compatible(
    prompt, llm_model, host, timeout_seconds,
    *, json_schema=None, allow_json_schema=False, logger=LOGGER,
):
    """Send to OpenAI-compatible API (vLLM, NVIDIA NIM, LiteLLM, etc.).

    If json_schema is provided, uses response_format with fallback-aware mode
    selection. By default this starts with json_object; json_schema is only
    attempted when allow_json_schema=True.
    """
    url = f"{host}/v1/chat/completions"

    payload = {
        "model": llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 16384,
    }

    response_mode: OpenAIResponseFormatMode = "none"

    # ── Constrained decoding via JSON schema/json_object (fallback-aware) ──
    if json_schema is not None:
        host_key = host.rstrip("/")
        cached_mode = _OPENAI_RESPONSE_FORMAT_MODE_BY_HOST.get(host_key)
        if allow_json_schema:
            response_mode = cached_mode or "json_schema"
        else:
            if cached_mode == "none":
                response_mode = "none"
            else:
                response_mode = "json_object"
        _set_openai_response_format(payload, response_mode, json_schema)
        logger.info(
            "Constrained decoding requested (mode=%s, json_schema_enabled=%s, %d schema properties)",
            response_mode,
            allow_json_schema,
            len(json_schema.get("properties", {})),
        )

    logger.debug("Prompt length: %d chars", len(prompt))
    llm_start = time.time()

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.post(url, json=payload)

            # If host rejects json_schema, cache and fall back to json_object.
            if (
                resp.status_code == 400
                and json_schema is not None
                and allow_json_schema
                and response_mode == "json_schema"
            ):
                error_text = resp.text[:500]
                if _response_format_error(error_text):
                    logger.warning(
                        "Server rejected json_schema format, "
                        "falling back to json_object mode: %s",
                        error_text[:200],
                    )
                    response_mode = "json_object"
                    _set_openai_response_format_mode(
                        host,
                        response_mode,
                        logger=logger,
                        reason="json_schema rejected by server",
                    )
                    _set_openai_response_format(payload, response_mode, json_schema)
                    resp = client.post(url, json=payload)

            # If host also rejects json_object, cache and fall back to unconstrained.
            if (
                resp.status_code == 400
                and json_schema is not None
                and response_mode == "json_object"
            ):
                error_text = resp.text[:500]
                if _response_format_error(error_text):
                    logger.warning(
                        "Server rejected json_object format, "
                        "falling back to unconstrained: %s",
                        error_text[:200],
                    )
                    response_mode = "none"
                    _set_openai_response_format_mode(
                        host,
                        response_mode,
                        logger=logger,
                        reason="json_object rejected by server",
                    )
                    _set_openai_response_format(payload, response_mode, json_schema)
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
        and response_mode != "none"
    )

    usage_raw = data.get("usage", {})
    usage = usage_raw if isinstance(usage_raw, dict) else {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    reasoning_tokens = None
    completion_details = usage.get("completion_tokens_details")
    if isinstance(completion_details, dict):
        reasoning_tokens = completion_details.get("reasoning_tokens")
    logger.debug(
        "Response in %.2fs (%d chars, %d prompt tokens, %d output tokens, constrained=%s)",
        llm_elapsed, len(content),
        prompt_tokens or 0,
        completion_tokens or 0,
        used_constrained,
    )

    return LLMResponse(
        content=content,
        model=llm_model,
        done=True,
        constrained=used_constrained,
        request_duration_s=llm_elapsed,
        prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
        completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
        reasoning_tokens=reasoning_tokens if isinstance(reasoning_tokens, int) else None,
        usage=usage if usage else None,
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

    usage = {
        "prompt_tokens": data.get("prompt_eval_count"),
        "completion_tokens": data.get("eval_count"),
    }

    return LLMResponse(
        content=content,
        model=llm_model,
        done=data.get("done", True),
        constrained=used_constrained,
        request_duration_s=llm_elapsed,
        total_duration_s=_ns_to_seconds(data.get("total_duration")),
        load_duration_s=_ns_to_seconds(data.get("load_duration")),
        prompt_eval_duration_s=_ns_to_seconds(data.get("prompt_eval_duration")),
        eval_duration_s=_ns_to_seconds(data.get("eval_duration")),
        prompt_tokens=data.get("prompt_eval_count")
        if isinstance(data.get("prompt_eval_count"), int) else None,
        completion_tokens=data.get("eval_count")
        if isinstance(data.get("eval_count"), int) else None,
        usage=usage,
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
