import httpx
import logging
import os
import socket
from typing import List, Tuple

LOGGER = logging.getLogger("midas-llm")


def _normalize_host(host: str) -> str:
    host = (host or "").rstrip("/")
    if host and not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host


def _append_candidate(hosts: List[str], seen: set, host: str) -> None:
    host = _normalize_host(host)
    if host and host not in seen:
        seen.add(host)
        hosts.append(host)


def candidate_ollama_hosts(initial_host: str | None = None) -> List[str]:
    """Build an ordered list of candidate Ollama hosts/ports to probe."""
    candidates: List[str] = []
    seen: set[str] = set()

    env_host = os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_API_BASE")
    env_port = os.getenv("OLLAMA_PORT")
    hostname = socket.gethostname()
    fqdn = socket.getfqdn()

    for host in (initial_host, env_host):
        _append_candidate(candidates, seen, host)  # prefer explicit inputs

    # Respect custom port if provided
    ports = [env_port] if env_port else []
    if "11434" not in ports:
        ports.append("11434")

    for port in ports:
        for base in ("localhost", "127.0.0.1", hostname, fqdn):
            _append_candidate(candidates, seen, f"{base}:{port}")

    return candidates


def probe_llm_host(host: str, timeout: float = 5.0, *, logger: logging.Logger = LOGGER) -> Tuple[bool, List[str]]:
    """Best-effort probe of the Ollama host for model availability."""
    host = _normalize_host(host)
    logger.info("Probing LLM host: %s", host)
    try:
        resp = httpx.get(f"{host}/api/tags", timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        models = [m.get("name") for m in data.get("models", [])] if isinstance(data, dict) else []
        logger.info("LLM host reachable. Models: %s", ', '.join(models) if models else 'None reported')
        return True, models
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM host probe failed: %s", exc)
        return False, []


def autodetect_llm_host(initial_host: str | None = None, timeout: float = 3.0, *, logger: logging.Logger = LOGGER) -> str | None:
    """Probe likely hosts and return the first reachable one."""
    candidates = candidate_ollama_hosts(initial_host)
    for host in candidates:
        ok, _ = probe_llm_host(host, timeout=timeout, logger=logger)
        if ok:
            if initial_host and _normalize_host(initial_host) != host:
                logger.info("Selected reachable LLM host %s (initial %s was unreachable)", host, initial_host)
            return host
    logger.warning("No reachable LLM host found after probing candidates: %s", ", ".join(candidates))
    return _normalize_host(initial_host) if initial_host else None


def discover_ollama_host(initial_host: str | None = None, timeout: float = 3.0, *, logger: logging.Logger = LOGGER) -> str | None:
    """Compatibility shim; use autodetect_llm_host."""
    return autodetect_llm_host(initial_host, timeout=timeout, logger=logger)
