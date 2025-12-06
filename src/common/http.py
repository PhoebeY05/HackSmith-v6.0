from typing import Any

import requests


def post_json(url: str, payload: dict) -> Any:
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()
