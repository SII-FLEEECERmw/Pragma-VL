from __future__ import annotations

import logging
import os
import pathlib
from typing import Any

import torch
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoProcessor, AutoTokenizer, Trainer

from reward.data.dataloader import make_supervised_data_module 
from reward.model.modeling_qwen_2_5_par_reward import ( 
    RewardModelQwenForCausalLM,
)
from reward.argument import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
)

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------

local_rank: int | None = None


def rank0_print(*args: Any) -> None:
    """Print only on rank 0."""
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str) -> None:
    """
    Save model weights in a HF-Trainer-friendly way.

    Notes:
        - With DeepSpeed enabled, defer to trainer.save_model.
        - Otherwise, move state dict to CPU before saving to reduce GPU memory pressure.
    """
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa: SLF001


def set_model(model_args: ModelArguments, model: torch.nn.Module) -> torch.nn.Module:
    """
    Configure trainable parameters.

    Behavior (kept unchanged):
        1) Freeze all parameters.
        2) If not using LoRA, selectively unfreeze full parameters.
        3) If using LoRA:
           - Inject LoRA into target modules.
           - Only enable LoRA params for selected submodules (LLM / Vision).
           - Always enable reward heads (multiheads / metavoter).
           - Optionally enable full finetuning for vision merger.
    """
    # 1) Freeze all parameters.
    for p in model.parameters():
        p.requires_grad = False

    # 2) Full-parameter tuning path (no LoRA).
    if not model_args.use_lora:
        rank0_print("Using Full-parameter fine-tuning.")

        if model_args.tune_mm_vision:
            for _, p in model.visual.named_parameters():
                p.requires_grad = True

        if model_args.tune_mm_mlp:
            for _, p in model.visual.merger.named_parameters():
                p.requires_grad = True

        if model_args.tune_mm_llm:
            for _, p in model.model.named_parameters():
                p.requires_grad = True

        for _, p in model.multiheads.named_parameters():
            p.requires_grad = True
        for _, p in model.metavoter.named_parameters():
            p.requires_grad = True

        return model

    # 3) LoRA path.
    rank0_print("Using LoRA for parameter-efficient fine-tuning.")

    tune_llm = bool(model_args.tune_mm_llm)
    tune_vision = bool(model_args.tune_mm_vision)

    if (not tune_llm) and (not tune_vision):
        rank0_print(
            "LoRA is enabled, but neither LLM nor Vision is selected. Only train reward heads."
        )
        for _, p in model.multiheads.named_parameters():
            p.requires_grad = True
        for _, p in model.metavoter.named_parameters():
            p.requires_grad = True
        return model

    # Parse LoRA targets.
    lora_target_modules = [
        x.strip()
        for x in model_args.lora_target_modules.split(",")
        if x.strip()
    ]

    # Build LoRA config (NO layers_pattern; keep as-is).
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(model, lora_config)

    # 4) Only unfreeze LoRA params in selected submodules.
    def _is_lora_param(name: str) -> bool:
        return ("lora_A" in name) or ("lora_B" in name) or ("lora_" in name)

    allow_prefixes: list[str] = []
    if tune_llm:
        allow_prefixes.append("base_model.model.model.")   # LLM parameters
    if tune_vision:
        allow_prefixes.append("base_model.model.visual.")  # Vision parameters

    for name, p in model.named_parameters():
        if not _is_lora_param(name):
            continue
        p.requires_grad = any(name.startswith(pref) for pref in allow_prefixes)

    # 5) Reward heads are always trainable.
    def _unfreeze_by_attr(root: Any, attr: str) -> bool:
        if hasattr(root, attr):
            for _, p in getattr(root, attr).named_parameters():
                p.requires_grad = True
            return True
        return False

    _unfreeze_by_attr(model, "multiheads")
    _unfreeze_by_attr(model, "metavoter")

    # Fallback paths for different PEFT wrappers.
    if hasattr(model, "base_model"):
        _unfreeze_by_attr(model.base_model, "multiheads")
        _unfreeze_by_attr(model.base_model, "metavoter")
        if hasattr(model.base_model, "model"):
            _unfreeze_by_attr(model.base_model.model, "multiheads")
            _unfreeze_by_attr(model.base_model.model, "metavoter")

    # 6) Optionally finetune full vision merger (NOT LoRA).
    if model_args.tune_mm_mlp:
        rank0_print("Additionally fine-tuning the full vision merger.")

        if hasattr(model, "visual") and hasattr(model.visual, "merger"):
            for _, p in model.visual.merger.named_parameters():
                p.requires_grad = True

        elif hasattr(model, "base_model"):
            bm = model.base_model
            if hasattr(bm, "visual") and hasattr(bm.visual, "merger"):
                for _, p in bm.visual.merger.named_parameters():
                    p.requires_grad = True
            if (
                hasattr(bm, "model")
                and hasattr(bm.model, "visual")
                and hasattr(bm.model.visual, "merger")
            ):
                for _, p in bm.model.visual.merger.named_parameters():
                    p.requires_grad = True

    return model


def _enable_gradient_checkpointing_hooks(model: torch.nn.Module) -> None:
    """Ensure input embeddings require grads when gradient checkpointing is enabled."""
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return

    def make_inputs_require_grad(_module: torch.nn.Module, _input: Any, output: torch.Tensor) -> None:
        output.requires_grad_(True)

    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


def train(attn_implementation: str = "flash_attention_2") -> None:
    """Main training entry."""
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print("model_args:\n", model_args)
    print("-----------")
    print("data_args:\n", data_args)
    print("-----------")
    print("training_args:\n", training_args)

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    model = RewardModelQwenForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
    )

    # Image processor is needed by the dataset/collator.
    data_args.image_processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path
    ).image_processor
    data_args.model_type = "reward_model_qwen"

    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        _enable_gradient_checkpointing_hooks(model)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    model = set_model(model_args, model)

    # Debug print: show trainable params on rank 0 only.
    if torch.distributed.get_rank() == 0:
        model.print_trainable_parameters()

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, processing_class=tokenizer, args=training_args, **data_module)

    has_checkpoint = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if has_checkpoint:
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # If LoRA is used, optionally merge and export a final inference model.
    if model_args.use_lora and trainer.is_world_process_zero():
        rank0_print("\n\nTraining complete. Merging LoRA layers for final inference model...")

        merged_output_dir = os.path.join(training_args.output_dir, "merged_model")
        os.makedirs(merged_output_dir, exist_ok=True)
        rank0_print(f"Merged model will be saved to: {merged_output_dir}")

        merged_model = trainer.model.to("cpu").merge_and_unload()
        rank0_print("Saving the merged model...")
        merged_model.save_pretrained(merged_output_dir, safe_serialization=True)

        trainer.tokenizer.save_pretrained(merged_output_dir)
        if hasattr(data_args, "image_processor"):
            data_args.image_processor.save_pretrained(merged_output_dir)

        rank0_print(f"🎉 Successfully merged and saved the final model to {merged_output_dir}")


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
