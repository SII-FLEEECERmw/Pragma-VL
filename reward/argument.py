import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Union


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA for parameter-efficient tuning."}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA rank."}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha."}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout."}
    )
    lora_target_modules: str = field(
        default="q_proj,v_proj",
        metadata={"help": "Comma-separated list of module names to apply LoRA to (e.g., 'q_proj,v_proj,k_proj,o_proj')."}
    )

@dataclass
class DataArguments:
  source_dir: str = field(default="")
  dataset_sources: str = field(default="")
  max_pixels: int = field(default=28 * 28 * 576)
  min_pixels: int = field(default=28 * 28 * 16)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
