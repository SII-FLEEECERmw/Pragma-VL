#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from reward.model.modeling_qwen_2_5_par_reward import RewardModelQwenForCausalLM


# -----------------------------------------------------------------------------
# CLI / Config
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ServerConfig:
    model_path: str
    host: str
    port: int

    device: str
    dtype: str
    trust_remote_code: bool
    attn_impl: str

    max_length: int
    padding: str
    truncation: bool
    micro_batch_size: int

    log_file: str
    log_rotation: str
    workers: int


def parse_args() -> ServerConfig:
    import argparse

    p = argparse.ArgumentParser(description="Reward Server (Qwen2.5 PAR Reward)")

    # I/O
    p.add_argument("--model-path", required=True, help="HF model path or local dir")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8300)

    # runtime
    p.add_argument("--device", default="cuda:1", help='e.g. "cuda", "cuda:0", "cpu"')
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable trust_remote_code",
    )
    p.add_argument("--attn-impl", default="flash_attention_2")

    # tokenizer/processor
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--padding", default="max_length")
    p.add_argument("--truncation", action="store_true", default=True)
    p.add_argument("--no-truncation", dest="truncation", action="store_false")

    # batching
    p.add_argument("--micro-batch-size", type=int, default=16)

    # logging
    p.add_argument("--log-file", default="reward_qwen_server.log")
    p.add_argument("--log-rotation", default="500 MB")

    # uvicorn
    p.add_argument("--workers", type=int, default=1)

    ns = p.parse_args()
    return ServerConfig(
        model_path=ns.model_path,
        host=ns.host,
        port=ns.port,
        device=ns.device,
        dtype=ns.dtype,
        trust_remote_code=ns.trust_remote_code,
        attn_impl=ns.attn_impl,
        max_length=ns.max_length,
        padding=ns.padding,
        truncation=ns.truncation,
        micro_batch_size=ns.micro_batch_size,
        log_file=ns.log_file,
        log_rotation=ns.log_rotation,
        workers=ns.workers,
    )


# -----------------------------------------------------------------------------
# Request/Response schemas
# -----------------------------------------------------------------------------
class RewardItem(BaseModel):
    question: str
    response: str
    image_base64: Optional[str] = Field(default=None, description="Optional base64-encoded image bytes")


class RewardRequest(BaseModel):
    items: List[RewardItem]
    system_prompt: str = "You are a helpful assistant."
    replace_image_token: bool = True


class RewardResponse(BaseModel):
    rewards: List[float]
    batch_size: int


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def resolve_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def decode_image_base64(image_b64: str) -> Image.Image:
    try:
        raw = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image_base64: {e}") from e


def chunks(n: int, chunk_size: int) -> List[Tuple[int, int]]:
    return [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]


# -----------------------------------------------------------------------------
# Inference Service
# -----------------------------------------------------------------------------
class RewardService:
    def __init__(self, cfg: ServerConfig) -> None:
        self.cfg = cfg
        self.model: Optional[RewardModelQwenForCausalLM] = None
        self.tokenizer = None
        self.processor = None

    def ensure_loaded(self) -> None:
        if self.model is not None:
            return

        if self.cfg.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("device is cuda*, but CUDA is not available.")

        torch_dtype = resolve_dtype(self.cfg.dtype)
        logger.info(f"Loading reward model from: {self.cfg.model_path}")

        self.model = RewardModelQwenForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.cfg.model_path,
            torch_dtype=torch_dtype,
            attn_implementation=self.cfg.attn_impl,
            trust_remote_code=self.cfg.trust_remote_code,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_path)
        self.processor = AutoProcessor.from_pretrained(self.cfg.model_path)

        self.model = self.model.to(self.cfg.device)
        self.model.eval()
        logger.info(f"Model ready on device={self.cfg.device}, dtype={torch_dtype}")

    def build_chat_text(self, system_prompt: str, question: str, response: str, replace_image_token: bool) -> str:
        assert self.tokenizer is not None
        assert self.processor is not None

        q = question
        if replace_image_token and "<image>" in q:
            image_token = getattr(self.processor, "image_token", "<image>")
            q = image_token.join(q.split("<image>"))

        conv = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
            {"role": "assistant", "content": response},
        ]
        return self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)

    def prepare_inputs(
        self,
        items: List[RewardItem],
        system_prompt: str,
        replace_image_token: bool,
        with_image: bool,
    ) -> Dict[str, torch.Tensor]:
        assert self.processor is not None

        texts: List[str] = []
        images: Optional[List[Image.Image]] = [] if with_image else None

        for it in items:
            if with_image:
                if not it.image_base64:
                    raise ValueError("with_image group contains item without image_base64")
                images.append(decode_image_base64(it.image_base64))
            else:
                if it.image_base64:
                    raise ValueError("no-image group contains item with image_base64")

            texts.append(self.build_chat_text(system_prompt, it.question, it.response, replace_image_token))

        encoded = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=self.cfg.padding,
            truncation=self.cfg.truncation,
            max_length=self.cfg.max_length,
        )
        return {k: v.to(self.cfg.device) for k, v in encoded.items() if torch.is_tensor(v)}

    def run_predict(self, inputs: Dict[str, torch.Tensor]) -> List[float]:
        assert self.model is not None

        amp_dtype = resolve_dtype(self.cfg.dtype)
        use_amp = self.cfg.device.startswith("cuda") and amp_dtype in (torch.float16, torch.bfloat16)

        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    _, reward_logits = self.model.predict(**inputs)
            else:
                _, reward_logits = self.model.predict(**inputs)

        if reward_logits.dim() == 2 and reward_logits.size(-1) == 1:
            reward_logits = reward_logits.squeeze(-1)

        return reward_logits.detach().float().cpu().tolist()

    def predict_rewards(self, req: RewardRequest) -> List[float]:
        n = len(req.items)
        if n == 0:
            return []

        rewards: List[Optional[float]] = [None] * n
        with_img_idx, no_img_idx = [], []

        for i, it in enumerate(req.items):
            (with_img_idx if it.image_base64 else no_img_idx).append(i)

        def run_group(indexes: List[int], with_image: bool) -> None:
            if not indexes:
                return

            group_items = [req.items[i] for i in indexes]
            for s, e in chunks(len(group_items), self.cfg.micro_batch_size):
                mb_items = group_items[s:e]
                mb_indexes = indexes[s:e]

                inputs = self.prepare_inputs(
                    items=mb_items,
                    system_prompt=req.system_prompt,
                    replace_image_token=req.replace_image_token,
                    with_image=with_image,
                )
                mb_rewards = self.run_predict(inputs)

                if len(mb_rewards) != len(mb_indexes):
                    raise RuntimeError("reward logits batch size mismatch")

                for orig_i, r in zip(mb_indexes, mb_rewards):
                    rewards[orig_i] = float(r)

        run_group(with_img_idx, with_image=True)
        run_group(no_img_idx, with_image=False)

        if any(r is None for r in rewards):
            raise RuntimeError("Some rewards are missing after inference")

        return [float(r) for r in rewards]


# -----------------------------------------------------------------------------
# App factory
# -----------------------------------------------------------------------------
def create_app(cfg: ServerConfig) -> FastAPI:
    logger.add(cfg.log_file, rotation=cfg.log_rotation, encoding="utf-8")

    service = RewardService(cfg)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        service.ensure_loaded()
        yield

    app = FastAPI(title="Reward Server (Qwen2.5 PAR Reward)", lifespan=lifespan)

    @app.post("/reward", response_model=RewardResponse)
    def reward(req: RewardRequest) -> RewardResponse:
        try:
            service.ensure_loaded()
            rewards = service.predict_rewards(req)
            return RewardResponse(rewards=rewards, batch_size=len(rewards))
        except Exception as e:
            logger.exception("Failed to compute reward.")
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


def main() -> None:
    cfg = parse_args()
    app = create_app(cfg)
    logger.info(f"Starting Reward Server on {cfg.host}:{cfg.port}")
    uvicorn.run(app, host=cfg.host, port=cfg.port, workers=cfg.workers)


if __name__ == "__main__":
    main()
