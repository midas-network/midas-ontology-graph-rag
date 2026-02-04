import httpx
import logging

LOGGER = logging.getLogger("midas-llm")


def probe_llm_host(host: str, timeout: float = 5.0, *, logger: logging.Logger = LOGGER) -> None:
    """Best-effort probe of the Ollama host for model availability."""
    logger.info("Probing LLM host: %s", host)
    try:
        resp = httpx.get(f"{host}/api/tags", timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        models = [m.get("name") for m in data.get("models", [])] if isinstance(data, dict) else []
        logger.info("LLM host reachable. Models: %s", ', '.join(models) if models else 'None reported')
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM host probe failed: %s", exc)
