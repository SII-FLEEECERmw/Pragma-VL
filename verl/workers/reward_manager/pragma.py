# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications Copyright 2025 Fudan University / Antgroup / Ming Wen
# Changes: refactor into class, add batching, add reward server client, etc.

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications Copyright 2025 Fudan University / Antgroup / Ming Wen
# Changes: refactor into class, add batching, add reward server client, etc.

from __future__ import annotations

import base64
import io
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple, Union

import requests
import torch
from PIL import Image

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from reward.serve_online.eval_processor import EvalAIAnswerProcessor


@dataclass(frozen=True)
class _ServerConfig:
    """Configuration for calling the online reward service."""
    url: str
    timeout_s: int
    system_prompt: str
    replace_image_token: bool


@register("pragma")
class PragmaRewardManager(AbstractRewardManager):
    """
    Reward manager that queries an online reward server.

    Expected input:
      - data.batch["prompts"], data.batch["responses"], data.batch["attention_mask"]
      - data.non_tensor_batch["extra_info"][i]["question"]
      - (optional) data.non_tensor_batch["extra_info"][i]["gt_answer"]
      - (optional) data.non_tensor_batch["original_image"][i] is a PIL.Image

    Output:
      - reward tensor [B] aligned to the first dimension of data.batch["responses"].
    """

    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score=None,  # optional external hook
        reward_fn_key: str = "data_source",
        reward_batch_size: int = 16,
        server_url: str = "http://127.0.0.1:8300/reward",
        timeout: int = 300,
        system_prompt: str = "You are a helpful assistant.",
        replace_image_token: bool = True,
        enable_gt_adjust: bool = True,
        gt_match_bonus_ratio: float = 0.6,
        gt_mismatch_penalty_ratio: float = 0.3,
        print_examples: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = int(num_examine)
        self.reward_fn_key = reward_fn_key

        self.reward_batch_size = int(reward_batch_size)
        self._server = _ServerConfig(
            url=server_url,
            timeout_s=int(timeout),
            system_prompt=system_prompt,
            replace_image_token=bool(replace_image_token),
        )

        self._eval_ai = EvalAIAnswerProcessor()

        self._enable_gt_adjust = bool(enable_gt_adjust)
        self._gt_match_bonus_ratio = float(gt_match_bonus_ratio)
        self._gt_mismatch_penalty_ratio = float(gt_mismatch_penalty_ratio)

        self._print_examples = bool(print_examples)
        self._printed = 0

        # If provided, must be:
        # compute_score(items: List[dict], server_url: str, timeout: int) -> List[float]
        self._external_compute_score = compute_score

    # -------------------------------------------------------------------------
    # Image / text helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _get_optional_image(non_tensor_batch: Dict[str, Any], idx: int) -> Optional[Image.Image]:
        
        images = non_tensor_batch.get("original_image")
        if idx < 0 or idx >= len(images):
            return None
        return images[idx]['image'][0]

    def _decode_response_text(
        self,
        prompts: torch.Tensor,
        responses_row: torch.Tensor,
        attention_mask_row: torch.Tensor,
    ) -> str:
        """
        Decode response tokens into text.

        Assumption (matches your original logic):
          - prompt_length = prompts.shape[-1]
          - valid response length is derived from attention_mask after prompt part
          - responses_row contains response token ids (not including prompt ids)
        """
        prompt_length = prompts.shape[-1]
        valid_resp_len = int(attention_mask_row[prompt_length:].sum().item())
        token_ids = responses_row[:valid_resp_len]
        return self.tokenizer.decode(token_ids)

    # -------------------------------------------------------------------------
    # Reward server client (IMPORTANT: keep as a dedicated method)
    # -------------------------------------------------------------------------
    def _compute_reward(self, items: List[Dict[str, Any]]) -> List[float]:
        """
        Compute rewards for a micro-batch by calling the online reward server.

        Notes:
        - This method is intentionally isolated to avoid being treated as a local
          reward model by VERL's internal "reward model loop" detection logic.
        """
        payload = {
            "items": items,
            "system_prompt": self._server.system_prompt,
            "replace_image_token": self._server.replace_image_token,
        }

        if self._external_compute_score is not None:
            return list(self._external_compute_score(payload, self._server.url, self._server.timeout_s))

        t0 = time.time()
        resp = requests.post(self._server.url, json=payload, timeout=self._server.timeout_s)
        elapsed = time.time() - t0

        try:
            data = resp.json()
        except Exception:
            data = {"raw_text": resp.text}

        if resp.status_code != 200:
            raise RuntimeError(f"reward server status={resp.status_code}, dt={elapsed:.3f}s, body={data}")
        if "rewards" not in data:
            raise RuntimeError(f"reward server response missing 'rewards', dt={elapsed:.3f}s, body={data}")
        if len(data["rewards"]) != len(items):
            raise RuntimeError(
                f"reward len mismatch: {len(data['rewards'])} != {len(items)}, dt={elapsed:.3f}s, body={data}"
            )

        return [float(x) for x in data["rewards"]]

    # -------------------------------------------------------------------------
    # Optional GT-based adjustment
    # -------------------------------------------------------------------------
    def _adjust_by_gt(
        self,
        rewards: List[float],
        gt_answers: List[Optional[str]],
        ai_answers: List[Optional[str]],
    ) -> List[float]:
        if not self._enable_gt_adjust or not rewards:
            return rewards

        span = float(max(rewards) - min(rewards))
        if span <= 0:
            return rewards

        adjusted = list(rewards)
        for i, (gt, ai) in enumerate(zip(gt_answers, ai_answers)):
            if gt is None or ai is None:
                continue
            if gt == ai:
                adjusted[i] += self._gt_match_bonus_ratio * span
            else:
                adjusted[i] -= self._gt_mismatch_penalty_ratio * span
        return adjusted

    def _maybe_print_one_example(self, items: Sequence[Dict[str, Any]], rewards: Sequence[float]) -> None:
        if not self._print_examples or self._printed >= self.num_examine or not items:
            return

        ex = items[0]
        print("\n[RewardExample]")
        print("Q:", str(ex.get("question", ""))[:300])
        print("A:", str(ex.get("response", ""))[:300])
        print("has_image:", ex.get("image_base64") is not None)
        print("reward:", float(rewards[0]))
        self._printed += 1

    # -------------------------------------------------------------------------
    # DataProto conversion
    # -------------------------------------------------------------------------
    def _build_micro_batch_items(
        self,
        mb: DataProto,
    ) -> Tuple[List[Dict[str, Any]], List[Optional[str]], List[Optional[str]]]:
        """
        Convert a micro DataProto batch into server request items.
        Returns:
          items, gt_parsed_list, ai_parsed_list
        """
        prompts = mb.batch["prompts"]
        responses = mb.batch["responses"]
        attn = mb.batch["attention_mask"]

        extra_info_list = mb.non_tensor_batch.get("extra_info")
        items: List[Dict[str, Any]] = []
        gt_parsed: List[Optional[str]] = []
        ai_parsed: List[Optional[str]] = []

        for i in range(len(mb)):
            question = str(extra_info_list[i].get("question", ""))
            gt = extra_info_list[i].get("gt_answer")

            resp_text = self._decode_response_text(prompts, responses[i], attn[i])

            gt_p = self._eval_ai(gt) if gt is not None else None
            ai_p = self._eval_ai(resp_text) if resp_text is not None else None
            gt_parsed.append(gt_p)
            ai_parsed.append(ai_p)

            item: Dict[str, Any] = {"question": question, "response": resp_text}

            img = self._get_optional_image(mb.non_tensor_batch, i)
            if img is not None:
                item["image_base64"] = self._image_to_base64(img)

            items.append(item)

        return items, gt_parsed, ai_parsed

    # -------------------------------------------------------------------------
    # Main entry
    # -------------------------------------------------------------------------
    def __call__(self, data: DataProto, return_dict: bool = False) -> Union[torch.Tensor, Dict[str, Any]]:
        # If rewards are already provided, return them directly.
        if "rm_scores" in data.batch:
            # rm_scores: [B] -> scatter to token-level tensor with same shape as responses
            bsz = len(data)
            device = data.batch["responses"].device
            rewards = data.batch["rm_scores"].to(device=device, dtype=torch.float32)

            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32, device=device)

            prompts = data.batch["prompts"]
            attn = data.batch["attention_mask"]
            prompt_length = prompts.shape[-1]

            for i in range(bsz):
                valid_resp_len = int(attn[i][prompt_length:].sum().item())
                if valid_resp_len > 0:
                    reward_tensor[i, valid_resp_len - 1] = rewards[i]

            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra = {k: data.non_tensor_batch[k] for k in reward_extra_keys}
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra}
            return reward_tensor

        bsz = len(data)
        device = data.batch["responses"].device

        # token-level reward tensor (same shape as responses)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32, device=device)
        prompts = data.batch["prompts"]
        attn = data.batch["attention_mask"]
        prompt_length = prompts.shape[-1]

        for start in range(0, bsz, self.reward_batch_size):
            end = min(start + self.reward_batch_size, bsz)
            mb = data[start:end]

            items, gt_parsed, ai_parsed = self._build_micro_batch_items(mb)

            outcome_rewards = self._compute_reward(items)
            outcome_rewards = self._adjust_by_gt(outcome_rewards, gt_parsed, ai_parsed)

            self._maybe_print_one_example(items, outcome_rewards)

            # scatter each outcome reward to (valid_response_length - 1)
            for j, r in enumerate(outcome_rewards):
                i = start + j
                valid_resp_len = int(attn[i][prompt_length:].sum().item())
                if valid_resp_len > 0:
                    reward_tensor[i, valid_resp_len - 1] = float(r)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": None}
        return reward_tensor
