# midas_api.py
from __future__ import annotations

import logging
import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional, TypedDict, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

__all__ = [
    "MidasClient",
    "MidasApiError",
    "MidasAuthError",
    "MidasNotFound",
]

log = logging.getLogger(__name__)


# -------- Exceptions ---------------------------------------------------------

class MidasApiError(RuntimeError):
    """Base exception for MIDAS API errors."""
    def __init__(self, message: str, status: Optional[int] = None, payload: Any = None):
        super().__init__(message)
        self.status = status
        self.payload = payload


class MidasAuthError(MidasApiError):
    """Authentication/authorization errors (401/403)."""
    pass


class MidasNotFound(MidasApiError):
    """Resource not found (404)."""
    pass


# -------- Types --------------------------------------------------------------

JSON = Union[dict, list, str, int, float, bool, None]

class PaperMetadata(TypedDict, total=False):
    id: int
    title: str
    authors: List[str]
    doi: Optional[str]
    published: Optional[str]
    # Add more fields as you learn the schema


# -------- Client -------------------------------------------------------------

class MidasClient:
    """
    Minimal, extensible client for the MIDAS API.

    Base URL defaults to: https://members.midasnetwork.us
    Authentication: apiKey passed as query param (per your example).

    Usage:
        client = MidasClient(api_key=os.environ["MIDAS_API_KEY"])
        ids = client.get_new_paper_ids("2025-01-01")
    """

    def __init__(
        self,
        api_key: str = os.getenv("MIDAS_API_KEY"),
        *,
        base_url: str = "https://members.midasnetwork.us",
        timeout: float = 15.0,
        max_retries: int = 3,
        backoff_factor: float = 0.6,
        user_agent: str = "midas-api-python/0.1",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Session with retries (handles 429/5xx & network errors)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

        retry = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "POST", "PUT", "DELETE", "PATCH"),
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    # ----- Core HTTP helper --------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[JSON] = None,
        timeout: Optional[float] = None,
    ) -> JSON:
        """
        Low-level request helper that appends apiKey and handles errors.
        """
        if params is None:
            params = {}
        # Always send the apiKey as query param (per your example)
        params = {**params, "apiKey": self.api_key}

        url = f"{self.base_url}/{path.lstrip('/')}"
        to = timeout or self.timeout

        resp = self.session.request(method, url, params=params, json=json, timeout=to)

        # Handle common error classes
        if 200 <= resp.status_code < 300:
            if resp.headers.get("Content-Type", "").startswith("application/json"):
                return resp.json()
            # Fallback: text body if not JSON
            return resp.text

        # Map specific error types
        if resp.status_code in (401, 403):
            raise MidasAuthError(
                f"Auth error {resp.status_code} calling {url}",
                status=resp.status_code,
                payload=_safe_json(resp),
            )
        if resp.status_code == 404:
            raise MidasNotFound(
                f"Not found calling {url}",
                status=resp.status_code,
                payload=_safe_json(resp),
            )

        # Try to surface server message
        raise MidasApiError(
            f"HTTP {resp.status_code} calling {url}: {resp.text[:500]}",
            status=resp.status_code,
            payload=_safe_json(resp),
        )

    # ----- Public API methods ------------------------------------------------

    def get_new_paper_ids(
        self,
        release_date: Union[str, date, datetime],
    ) -> List[int]:
        """
        Return an array of paper IDs (numbers) released since a given date.

        Mirrors:
        GET /midascontacts/query/papers/ids/new?apiKey=...&releaseDate=YYYY-MM-DD

        :param release_date: date string "YYYY-MM-DD", datetime, or date
        """
        rd = _coerce_date(release_date)
        data = self._request(
            "GET",
            "/midascontacts/query/papers/ids/new",
            params={"releaseDate": rd},
        )
        # Expecting a JSON array of numbers
        if isinstance(data, list) and all(isinstance(x, int) for x in data):
            return data
        raise MidasApiError(
            "Unexpected response format for get_new_paper_ids; expected List[int].",
            payload=data,
        )

    def get_paper(self, paper_id: Union[int, str]) -> Dict[str, Any]:
        """
        Fetch a specific paper's JSON record by ID.

        Mirrors:
        GET /midascontacts/query/papers/{paper_id}?apiKey=...

        Example:
            client.get_paper(24231)

        Returns:
            A dict representing the paper's metadata.
        """
        data = self._request(
            "GET",
            f"/midascontacts/query/papers/{paper_id}",
            # Optional: you can set headers here, though the accept header
            # is typically already handled by requests
        )

        if not isinstance(data, dict):
            raise MidasApiError(
                f"Unexpected response format for paper {paper_id}; expected dict.",
                payload=data,
            )

        return data

# -------- Utilities ----------------------------------------------------------

def _coerce_date(d: Union[str, date, datetime]) -> str:
    if isinstance(d, str):
        # Trust caller; optional validation could be added here
        return d
    if isinstance(d, datetime):
        return d.date().isoformat()
    if isinstance(d, date):
        return d.isoformat()
    raise TypeError("release_date must be str, date, or datetime")

def _safe_json(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return {"text": resp.text[:1000]}


def get_paper_data(paper_id, api_key):
    """Retrieve and parse paper data from the MIDAS API."""
    result = {}
    client = MidasClient(api_key=api_key)
    paper_data = client.get_paper(paper_id)
    result['paper_title'] = paper_data.get("title", "")
    result['paper_abstract'] = paper_data['paperAbstract']['paperAbstractText'][0]['value']
    result['paper_keywords'] = [d[0]['name'] for d in [list(x.values()) for x in paper_data["paperPubMedKeywords"]]]
    result['paper_meshterms'] = [d[0]['name'] for d in [list(x.values()) for x in paper_data["paperPubMedMeshTerms"]]]
    return result


# -------- CLI (optional) ----------------------------------------------------


if __name__ == "__main__":
    _main()
