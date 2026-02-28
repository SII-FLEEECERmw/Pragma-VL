from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch
from datasets import concatenate_datasets, load_from_disk
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer

from .data_utils import get_rope_index_25, preprocess_qwen_2_visual


class LazyHybridLossRewardDataset(Dataset):
    """
    Lazily indexed reward dataset supporting mixed loss types:
      - BT / BT+MSE (paired chosen vs rejected)
      - MSE (single response regression)

    The dataset builds a global shuffled index across multiple on-disk datasets, and
    performs minimal I/O at __getitem__ time by fetching only the required row.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: Any) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.get_rope_index = get_rope_index_25

        dataset_configs = self._parse_dataset_sources(self.data_args.dataset_sources)

        print("Building lazy dataset index...")
        self.datasets, self.source_map = self._build_lazy_index(
            self.data_args.source_dir, dataset_configs
        )

        self.virtual_indices = self._create_virtual_indices()
        print(f"Lazy index built. Total virtual samples: {len(self.virtual_indices)}")

    @staticmethod
    def _parse_dataset_sources(raw_sources: str) -> List[Dict[str, Any]]:
        """Parse JSON string from args (with tolerant quote stripping)."""
        cleaned = raw_sources.strip("'\"")
        return json.loads(cleaned)

    def _build_lazy_index(
        self, source_dir: str, dataset_configs: List[Dict[str, Any]]
    ) -> Tuple[List[Dataset], List[Dict[str, Any]]]:
        """
        Scan disk and create lazy dataset views (no full in-memory load).
        Applies per-source sampling_rate via Dataset.select (still lazy).
        """
        datasets_list: List[Dataset] = []
        source_map: List[Dict[str, Any]] = []

        for cfg in dataset_configs:
            source_path = os.path.join(source_dir, cfg["name"])
            source_type = cfg["type"]
            sampling_rate = cfg.get("sampling_rate", 1.0)

            print(
                f"Indexing source: {source_path} (type: {source_type}, rate: {sampling_rate})"
            )

            if not os.path.exists(source_path):
                print(f"  Warning: Path not found, skipping: {source_path}")
                continue

            category_datasets: List[Dataset] = []
            categories = sorted(
                d
                for d in os.listdir(source_path)
                if os.path.isdir(os.path.join(source_path, d))
            )

            for category in categories:
                category_path = os.path.join(source_path, category, "train")
                try:
                    ds = load_from_disk(category_path, keep_in_memory=False)
                    category_datasets.append(ds)
                except Exception as e:
                    print(
                        f"  Warning: Failed to load index for category '{category}'. Error: {e}"
                    )

            if not category_datasets:
                continue

            combined_dataset = concatenate_datasets(category_datasets)

            if sampling_rate < 1.0:
                num_samples = int(len(combined_dataset) * sampling_rate)
                sampled_indices = random.sample(range(len(combined_dataset)), num_samples)
                combined_dataset = combined_dataset.select(sampled_indices)
                print(f"  -> Created a lazy view with {len(combined_dataset)} samples.")

            datasets_list.append(combined_dataset)
            source_map.append({"type": source_type, "size": len(combined_dataset)})

        return datasets_list, source_map

    def _create_virtual_indices(self) -> List[Tuple[int, int]]:
        """
        Create a global shuffled index across all sources.

        Each element is (dataset_idx, sample_idx_in_dataset).
        """
        indices: List[Tuple[int, int]] = []
        for dataset_idx, src in enumerate(self.source_map):
            size = src["size"]
            indices.extend((dataset_idx, sample_idx) for sample_idx in range(size))

        random.shuffle(indices)
        return indices

    def __len__(self) -> int:
        return len(self.virtual_indices)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        """
        Resolve virtual index to (dataset_idx, sample_idx), fetch the row, and preprocess it.
        """
        dataset_idx, sample_idx = self.virtual_indices[i]
        sample_metadata = self.datasets[dataset_idx][sample_idx]
        sample_metadata["source_type"] = self.source_map[dataset_idx]["type"]
        return self._process_sample(sample_metadata)

    def _process_sample(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch sample preprocessing based on its source_type."""
        source_type = metadata["source_type"]
        if source_type in {"bt", "bt_mse"}:
            return self._process_bt_sample(metadata)
        if source_type == "mse":
            return self._process_mse_sample(metadata)
        raise ValueError(f"Unknown source type: {source_type}")

    def _process_bt_sample(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess a BT (paired) sample."""
        question = metadata["question"]
        image = metadata.get("image")

        chosen_proc = self._process_single_response(
            question, metadata["chosen_answer"], image
        )
        rejected_proc = self._process_single_response(
            question, metadata["rejected_answer"], image
        )

        try:
            labels = torch.tensor(
                [
                    [
                        metadata["chosen_score_helpness"],
                        metadata["chosen_score_harmlessness"],
                        metadata["chosen_weighted_score"],
                    ],
                    [
                        metadata["rejected_score_helpness"],
                        metadata["rejected_score_harmlessness"],
                        metadata["rejected_weighted_score"],
                    ],
                ],
                dtype=torch.float32,
            )
            return {
                "chosen": chosen_proc,
                "rejected": rejected_proc,
                "labels": labels,
                "source_type": metadata["source_type"],
            }
        except Exception:
            print(metadata)
            breakpoint()
            raise

    def _process_mse_sample(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess an MSE (single) sample."""
        question = metadata["question"]
        image = metadata.get("image")

        response_proc = self._process_single_response(question, metadata["answer"], image)
        labels = torch.tensor(
            [
                metadata["score_helpness"],
                metadata["score_harmlessness"],
                metadata["weighted_score"],
            ],
            dtype=torch.float32,
        )
        return {
            "response": response_proc,
            "labels": labels,
            "source_type": metadata["source_type"],
        }

    def _process_single_response(self, question: str, answer: str, image: Any) -> Dict[str, Any]:
        """
        Convert a single (question, answer, image?) into tokenized inputs + RoPE indices.

        Returns:
            dict with:
              - input_ids: [L]
              - position_ids: [3, L]
              - pixel_values: Tensor | None
              - image_grid_thw: [3] | None
        """
        pixel_values, image_grid_thw = None, None
        if image is not None:
            pixel_values, image_grid_thw = self._process_image(image)

        conversations = [
            {
                "from": "human",
                "value": f"{question}\n<image>" if image is not None else question,
            },
            {"from": "gpt", "value": answer},
        ]

        grid_thw_for_text = (
            [
                image_grid_thw.prod().item()
                // (self.data_args.image_processor.merge_size**2)
            ]
            if image_grid_thw is not None
            else []
        )

        text_data = preprocess_qwen_2_visual(
            [conversations],
            self.tokenizer,
            grid_thw_image=grid_thw_for_text,
        )

        input_ids = torch.tensor(text_data["input_ids"], dtype=torch.long).unsqueeze(0)  # [1, L]

        position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            input_ids,
            image_grid_thw=image_grid_thw.unsqueeze(0) if image_grid_thw is not None else None,
        )
        # position_ids: [3, 1, L]

        return {
            "input_ids": input_ids.squeeze(0),       # [L]
            "position_ids": position_ids.squeeze(1), # [3, L]
            "pixel_values": pixel_values,            # Tensor | None
            "image_grid_thw": image_grid_thw,        # [3] | None
        }

    def _process_image(self, image: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run vision processor and return (pixel_values, image_grid_thw)."""
        processor = self.data_args.image_processor
        image = image.convert("RGB")

        visual_processed = processor.preprocess(image, return_tensors="pt")
        pixel_values = visual_processed["pixel_values"]
        image_grid_thw = visual_processed["image_grid_thw"]

        image_tensor = pixel_values[0] if isinstance(pixel_values, list) else pixel_values
        grid_thw = image_grid_thw[0]
        return image_tensor, grid_thw


@dataclass
class StandardRewardDataCollator:
    """
    Collator for non-packed training:
      - BT expands to (chosen, rejected)
      - MSE expands to a single response

    Text is padded to [B, Lmax]. Images are concatenated across the batch
    (dynamic resolution expected to vary only on dim0).
    """

    tokenizer: PreTrainedTokenizer
    pad_to_multiple_of: int | None = None

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        LOSS_TYPE_MSE = 0
        LOSS_TYPE_BT = 1
        LOSS_TYPE_BT_MSE = 2

        responses: List[Dict[str, Any]] = []
        labels: List[List[float]] = []

        for inst in instances:
            st = inst["source_type"]
            if st == "bt":
                responses += [inst["chosen"], inst["rejected"]]
                labels += [
                    [
                        LOSS_TYPE_BT,
                        float(inst["labels"][0][0]),
                        float(inst["labels"][0][1]),
                        float(inst["labels"][0][2]),
                    ],
                    [
                        LOSS_TYPE_BT,
                        float(inst["labels"][1][0]),
                        float(inst["labels"][1][1]),
                        float(inst["labels"][1][2]),
                    ],
                ]
            elif st == "bt_mse":
                responses += [inst["chosen"], inst["rejected"]]
                labels += [
                    [
                        LOSS_TYPE_BT_MSE,
                        float(inst["labels"][0][0]),
                        float(inst["labels"][0][1]),
                        float(inst["labels"][0][2]),
                    ],
                    [
                        LOSS_TYPE_BT_MSE,
                        float(inst["labels"][1][0]),
                        float(inst["labels"][1][1]),
                        float(inst["labels"][1][2]),
                    ],
                ]
            elif st == "mse":
                responses.append(inst["response"])
                s = inst["labels"]
                labels.append([LOSS_TYPE_MSE, float(s[0]), float(s[1]), float(s[2])])
            else:
                raise ValueError(f"Unknown source_type: {st}")

        if not responses:
            return {}

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer.pad_token_id is None")

        lengths = [int(r["input_ids"].numel()) for r in responses]
        max_len = max(lengths)

        if self.pad_to_multiple_of is not None:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        batch_size = len(responses)
        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        position_ids = torch.zeros((3, batch_size, max_len), dtype=torch.long)

        for i, r in enumerate(responses):
            ids = r["input_ids"].to(torch.long)      # [L]
            pos = r["position_ids"].to(torch.long)   # [3, L]
            seq_len = ids.numel()

            input_ids[i, :seq_len] = ids
            attention_mask[i, :seq_len] = 1

            if pos.ndim == 1:
                pos = pos.unsqueeze(0).expand(3, -1)
            position_ids[:, i, :seq_len] = pos

        batch: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": torch.tensor(labels, dtype=torch.float32),
        }

        pixel_values_list: List[torch.Tensor] = []
        image_grid_thw_list: List[torch.Tensor] = []
        for r in responses:
            pv = r.get("pixel_values")
            g = r.get("image_grid_thw")
            if pv is not None:
                pixel_values_list.append(pv)
                image_grid_thw_list.append(g)

        if pixel_values_list:
            batch["pixel_values"] = torch.cat(pixel_values_list, dim=0)
            batch["image_grid_thw"] = torch.stack(image_grid_thw_list, dim=0).to(torch.long)

        return batch


def make_supervised_data_module(tokenizer: PreTrainedTokenizer, data_args: Any) -> Dict[str, Any]:
    """Factory for Trainer: dataset + collator."""
    train_dataset = LazyHybridLossRewardDataset(tokenizer=tokenizer, data_args=data_args)
    data_collator = StandardRewardDataCollator(tokenizer=tokenizer)
    return {
        "train_dataset": train_dataset,
        "eval_dataset": None,
        "data_collator": data_collator,
    }
