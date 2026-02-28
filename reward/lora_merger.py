#!/usr/bin/env python3
"""
LoRA Merger Utility
===================

Merge a PEFT/LoRA adapter (saved in a HuggingFace Trainer checkpoint) into a base
Reward model, and export a standalone inference-ready model directory.

This script is designed for the training setup where:
- The base model class is `RewardModelQwenForCausalLM`.
- LoRA adapters are saved under `checkpoint-*` folders (via HF Trainer / PEFT).
- Tokenizer and Processor (vision/text preprocessor) should be saved together
  with the merged model for inference.

Typical usage
-------------
1) Merge from a specific checkpoint:
    python lora_merger.py \
      --base_model /path/to/base_model \
      --checkpoint_or_adapter /path/to/output_dir/checkpoint-2583 \
      --out_dir /path/to/output_dir/merged_model \
      --dtype bfloat16

2) Pick latest checkpoint automatically:
    python lora_merger.py \
      --base_model /path/to/base_model \
      --checkpoint_or_adapter /path/to/output_dir \
      --auto_latest \
      --out_dir /path/to/output_dir/merged_model
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Optional

import torch
from loguru import logger
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer

from reward.model.modeling_qwen_2_5_par_reward import RewardModelQwenForCausalLM


# -----------------------------------------------------------------------------
# Configuration / helpers
# -----------------------------------------------------------------------------

def resolve_dtype(name: str) -> torch.dtype:
    """Map a human-readable dtype name to torch.dtype."""
    n = name.lower()
    if n in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if n in {"fp16", "float16"}:
        return torch.float16
    if n in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def require_dir(path: str, what: str) -> None:
    """Fail fast if `path` is not an existing directory."""
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{what} is not a directory: {path}")


def find_latest_checkpoint(output_dir: str) -> str:
    """
    Find the latest `checkpoint-*` directory in an output directory.
    Assumes checkpoints are named like `checkpoint-1234`.
    """
    require_dir(output_dir, "output_dir")

    candidates = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* found under: {output_dir}")

    def _step(p: str) -> int:
        name = os.path.basename(p)
        try:
            return int(name.split("-")[-1])
        except ValueError:
            return -1

    candidates.sort(key=_step)
    latest = candidates[-1]
    if _step(latest) < 0:
        raise RuntimeError(f"Found checkpoint dirs but none match checkpoint-<int>: {candidates}")
    return latest


def infer_assets_source(base_model: str, adapter_or_ckpt: str) -> str:
    """
    Choose where to load tokenizer/processor from.

    - Prefer `adapter_or_ckpt` if it already contains tokenizer/processor files.
      This is useful if you saved updated special tokens/chat_template.
    - Otherwise fall back to `base_model`.
    """
    preferred_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "preprocessor_config.json",
    ]

    if os.path.isdir(adapter_or_ckpt) and any(os.path.exists(os.path.join(adapter_or_ckpt, f)) for f in preferred_files):
        return adapter_or_ckpt
    return base_model


def save_tokenizer_and_processor(src_dir: str, out_dir: str, trust_remote_code: bool) -> None:
    """Save tokenizer and processor to `out_dir`."""
    logger.info(f"Saving tokenizer/processor from: {src_dir}")

    tokenizer = AutoTokenizer.from_pretrained(src_dir, trust_remote_code=trust_remote_code, use_fast=False)
    tokenizer.save_pretrained(out_dir)

    # Processor is optional depending on model type; handle gracefully.
    try:
        processor = AutoProcessor.from_pretrained(src_dir, trust_remote_code=trust_remote_code)
        processor.save_pretrained(out_dir)
    except Exception as e:
        logger.warning(f"AutoProcessor not saved (not found or not supported): {e}")


# -----------------------------------------------------------------------------
# Main logic
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class MergeArgs:
    base_model: str
    adapter_or_checkpoint: str
    out_dir: str

    device: str
    dtype: torch.dtype
    attn_implementation: str

    trust_remote_code: bool
    safe_serialization: bool
    max_shard_size: str


def merge_lora(args: MergeArgs) -> None:
    """
    Merge LoRA adapter weights into base model and export a standalone model.

    Steps:
      1) Load base model from `args.base_model`.
      2) Load LoRA adapter from `args.adapter_or_checkpoint` and attach to base.
      3) Merge weights (merge_and_unload).
      4) Save merged model to `args.out_dir`.
      5) Save tokenizer + processor to `args.out_dir`.
    """
    require_dir(args.base_model, "base_model")
    require_dir(args.adapter_or_checkpoint, "adapter_or_checkpoint")
    os.makedirs(args.out_dir, exist_ok=True)

    logger.info("==== LoRA Merger ====")
    logger.info(f"Base model         : {args.base_model}")
    logger.info(f"Adapter/checkpoint : {args.adapter_or_checkpoint}")
    logger.info(f"Output dir         : {args.out_dir}")
    logger.info(f"Device             : {args.device}")
    logger.info(f"Dtype              : {args.dtype}")
    logger.info(f"Attention impl     : {args.attn_implementation}")
    logger.info(f"Safe serialization : {args.safe_serialization}")
    logger.info(f"Max shard size     : {args.max_shard_size}")

    # 1) Load base model
    logger.info("Loading base model...")
    base_model = RewardModelQwenForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    base_model.eval()

    if args.device != "cpu":
        base_model = base_model.to(args.device)

    # 2) Attach adapter
    logger.info("Attaching LoRA adapter...")
    lora_model = PeftModel.from_pretrained(
        base_model,
        args.adapter_or_checkpoint,
        is_trainable=False,
    )

    # 3) Merge adapter weights into base weights
    logger.info("Merging adapter into base weights (merge_and_unload)...")
    merged = lora_model.merge_and_unload()
    merged.eval()

    # 4) Save merged model (move to CPU to reduce GPU memory pressure)
    logger.info("Saving merged model to disk...")
    merged = merged.to("cpu")
    merged.save_pretrained(
        args.out_dir,
        safe_serialization=args.safe_serialization,
        max_shard_size=args.max_shard_size,
    )

    # 5) Save tokenizer + processor
    assets_src = infer_assets_source(args.base_model, args.adapter_or_checkpoint)
    save_tokenizer_and_processor(assets_src, args.out_dir, args.trust_remote_code)

    logger.info("Merge complete.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base reward model.")

    parser.add_argument("--base_model", type=str, required=True, help="Base pretrained model directory/repo.")
    parser.add_argument(
        "--checkpoint_or_adapter",
        type=str,
        required=True,
        help="Path to a checkpoint-* dir, adapter dir, or an output_dir containing checkpoint-*.",
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save the merged model.")

    parser.add_argument(
        "--auto_latest",
        action="store_true",
        help="If checkpoint_or_adapter is an output_dir, automatically pick the latest checkpoint-*.",
    )

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="bfloat16", help="bfloat16|float16|float32")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")

    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", action="store_false", dest="trust_remote_code")

    parser.add_argument("--safe_serialization", action="store_true", default=True)
    parser.add_argument("--unsafe_serialization", action="store_false", dest="safe_serialization")
    parser.add_argument("--max_shard_size", type=str, default="2GB")

    return parser


def resolve_checkpoint_path(path: str, auto_latest: bool) -> str:
    """
    Resolve the adapter/checkpoint path.

    If `auto_latest` is enabled and `path` is not a `checkpoint-*` folder, treat it as
    an output_dir and pick the latest checkpoint.
    """
    require_dir(path, "checkpoint_or_adapter")

    base = os.path.basename(os.path.abspath(path))
    if auto_latest and not base.startswith("checkpoint-"):
        return find_latest_checkpoint(path)

    return path


def main() -> None:
    parser = build_arg_parser()
    cli = parser.parse_args()

    adapter_or_ckpt = resolve_checkpoint_path(cli.checkpoint_or_adapter, cli.auto_latest)
    if cli.auto_latest:
        logger.info(f"Resolved checkpoint: {adapter_or_ckpt}")

    args = MergeArgs(
        base_model=cli.base_model,
        adapter_or_checkpoint=adapter_or_ckpt,
        out_dir=cli.out_dir,
        device=cli.device,
        dtype=resolve_dtype(cli.dtype),
        attn_implementation=cli.attn_implementation,
        trust_remote_code=cli.trust_remote_code,
        safe_serialization=cli.safe_serialization,
        max_shard_size=cli.max_shard_size,
    )

    merge_lora(args)


if __name__ == "__main__":
    main()
