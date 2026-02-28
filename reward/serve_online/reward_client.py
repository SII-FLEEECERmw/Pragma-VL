from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import requests
import base64
import io
from PIL import Image

def compute_rewards(payload, server_url, server_timeout):
    t0 = time.time()
    resp = requests.post(server_url, json=payload, timeout=server_timeout)
    elapsed = time.time() - t0

    try:
        data = resp.json()
    except Exception:
        data = {"raw_text": resp.text}

    if resp.status_code != 200:
        raise RuntimeError(f"reward server status={resp.status_code}, dt={elapsed:.3f}s, body={data}")
    if "rewards" not in data:
        raise RuntimeError(f"reward server response missing 'rewards', dt={elapsed:.3f}s, body={data}")
    if len(data["rewards"]) != len(payload['items']):
        raise RuntimeError(
            f"reward len mismatch: {len(data['rewards'])} != {len(payload['items'])}, dt={elapsed:.3f}s, body={data}"
        )
    return [float(x) for x in data["rewards"]]



@dataclass(frozen=True)
class RewardServerConfig:
    """Configuration for an external reward server."""
    url: str
    timeout_s: float = 60.0
    system_prompt: str = "You are a helpful assistant."
    replace_image_token: bool = True


ExternalComputeFn = Callable[[List[Dict[str, Any]], str, float], List[float]]


class RewardServerClient:
    """
    Stateless reward server client.

    Design notes:
    - All methods are static/class methods so they can be used in both:
      1) VERL RewardManager
      2) offline quality evaluation scripts
    - No internal caching/state to keep it safe under multiprocessing.
    """

    @staticmethod
    def compute_rewards(
        items: List[Dict[str, Any]],
        server: RewardServerConfig,
        external_compute_score: Optional[ExternalComputeFn] = None,
    ) -> List[float]:
        """
        Compute rewards for a batch by calling the online reward server.

        Args:
            items:
                List of dict items. Each dict should match your server schema,
                e.g. {"question": ..., "response": ..., "image_base64": ...}.
            server:
                Reward server configuration.
            external_compute_score:
                Optional function hook to override HTTP call. Signature:
                fn(items, url, timeout_s) -> rewards

        Returns:
            List[float]: reward values, same length as items.

        Raises:
            RuntimeError: if server returns non-200 or malformed response.
        """
        if not items:
            return []

        if external_compute_score is not None:
            rewards = external_compute_score(payloads, server.url, server.timeout_s)
            return RewardServerClient._validate_rewards(rewards, len(items), meta="external_compute_score")

        payload = {
            "items": items,
            "system_prompt": server.system_prompt,
            "replace_image_token": server.replace_image_token,
        }

        t0 = time.time()
        resp = requests.post(server.url, json=payload, timeout=server.timeout_s)
        elapsed = time.time() - t0

        data = RewardServerClient._safe_json(resp)
        if resp.status_code != 200:
            raise RuntimeError(
                f"reward server status={resp.status_code}, dt={elapsed:.3f}s, url={server.url}, body={data}"
            )

        if "rewards" not in data:
            raise RuntimeError(
                f"reward server response missing 'rewards', dt={elapsed:.3f}s, url={server.url}, body={data}"
            )

        rewards = data["rewards"]
        return RewardServerClient._validate_rewards(rewards, len(items), meta=f"http dt={elapsed:.3f}s")

    @staticmethod
    def _safe_json(resp: requests.Response) -> Dict[str, Any]:
        """Parse JSON response safely; fallback to raw text for debugging."""
        try:
            return resp.json()
        except Exception:
            return {"raw_text": resp.text}

    @staticmethod
    def _validate_rewards(rewards: Any, expected_len: int, meta: str) -> List[float]:
        """Validate reward list and cast to float."""
        if not isinstance(rewards, list):
            raise RuntimeError(f"reward type invalid: {type(rewards)}, meta={meta}")

        if len(rewards) != expected_len:
            raise RuntimeError(f"reward len mismatch: {len(rewards)} != {expected_len}, meta={meta}")

        try:
            return [float(x) for x in rewards]
        except Exception as e:
            raise RuntimeError(f"reward cast to float failed, meta={meta}, err={e}, rewards={rewards}") from e



def _dummy_image_b64() -> str:
    img = Image.new("RGB", (256, 256), (120, 200, 80))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def main() -> None:
    server = RewardServerConfig(
        url="http://127.0.0.1:8300/reward",
        timeout_s=120,
        system_prompt="You are a helpful assistant.",
        replace_image_token=True,
    )

    items: List[Dict[str, Any]] = [
        {
            "question": "Describe the image. <image>",
            "response": "It looks like a solid-color square image.",
            "image_base64": _dummy_image_b64(),
        },
        {
            "question": "What is 2+2?",
            "response": "2+2=4.",
            "image_base64": None,
        },
    ]

    rewards = RewardServerClient.compute_rewards(items, server)
    for i, r in enumerate(rewards):
        print(i, r)


if __name__ == "__main__":
    main()
