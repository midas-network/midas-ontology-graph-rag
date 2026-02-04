from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Tuple
import time
import logging

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

LOGGER = logging.getLogger("midas-llm")


def send_to_llm(
    prompt: str,
    llm_model: str,
    ollama_host: str,
    timeout_seconds: int = 300,
    *,
    logger: logging.Logger = LOGGER,
) -> Any:
    if not ollama_host.startswith(("http://", "https://")):
        ollama_host = f"http://{ollama_host}"

    logger.info("SENDING TO LLM | model=%s host=%s timeout=%ss", llm_model, ollama_host, timeout_seconds)

    llm = ChatOllama(
        model=llm_model,
        temperature=0,
        base_url=ollama_host,
    )

    logger.info("Sending request to LLM (timeout %d seconds)...", timeout_seconds)
    llm_start = time.time()

    def _call():
        return llm.invoke([HumanMessage(content=prompt)])

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_call)
        try:
            response = future.result(timeout=timeout_seconds)
        except FuturesTimeoutError as e:  # noqa: PERF203
            future.cancel()
            raise TimeoutError(
                f"LLM call exceeded {timeout_seconds}s. "
                "Check Ollama load/queue, server reachability, or try a smaller model."
            ) from e

    llm_elapsed = time.time() - llm_start
    logger.info("LLM response received in %.2fs", llm_elapsed)
    return response


def test_respond_ok(
    llm_model: str,
    ollama_host: str,
    timeout_seconds: int = 60,
    *,
    logger: logging.Logger = LOGGER,
) -> Tuple[bool, str]:
    """Send a minimal prompt_text and return (is_ok, raw_response or error)."""
    try:
        resp = send_to_llm(
            prompt="Respond with exactly: OK",
            llm_model=llm_model,
            ollama_host=ollama_host,
            timeout_seconds=timeout_seconds,
            logger=logger,
        )
        text = getattr(resp, "content", "") or ""
        is_ok = text.strip() == "OK"
        return is_ok, text
    except Exception as exc:  # noqa: BLE001
        logger.error("Sanity ping failed: %s", exc)
        return False, f"ERROR: {exc}"
