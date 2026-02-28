from __future__ import annotations

import copy
import re
from typing import Dict, List, Optional, Tuple

import torch
from transformers.tokenization_utils import PreTrainedTokenizer

# -----------------------------------------------------------------------------
# Global constants
# -----------------------------------------------------------------------------

IGNORE_INDEX = -100


# -----------------------------------------------------------------------------
# Dataset utilities
# -----------------------------------------------------------------------------

def parse_sampling_rate(dataset_name: str) -> float:
    """Parse sampling rate from a dataset name, e.g. `my_data%80` -> 0.8."""
    match = re.search(r"%(\d+)$", dataset_name)
    return int(match.group(1)) / 100.0 if match else 1.0


# -----------------------------------------------------------------------------
# Text preprocessing
# -----------------------------------------------------------------------------

def preprocess_qwen_2_visual(
    sources: List[List[Dict]],
    tokenizer: PreTrainedTokenizer,
    grid_thw_image: List[int] = [],
) -> Dict[str, torch.Tensor]:
    """
    Format conversational data into Qwen2-VL chat-template token ids.

    Notes:
        - This function intentionally does NOT pad; padding should be done in a DataCollator.
        - Image placeholders `<image>` (user side) are expanded into vision tokens based on `grid_thw_image`.
    """
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."

    # Avoid mutating the incoming tokenizer instance.
    tokenizer = copy.deepcopy(tokenizer)

    # Ensure a chat template exists (required by apply_chat_template).
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        )

    visual_replicate_index_image = 0
    current_input_ids: List[int] = []
    current_targets: List[int] = []

    for source in sources:
        # Add system message tokens.
        system_formatted = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}],
            add_generation_prompt=False,
            tokenize=False,
        )
        system_tokens = tokenizer.encode(system_formatted, add_special_tokens=False)
        current_input_ids.extend(system_tokens)
        current_targets.extend([IGNORE_INDEX] * len(system_tokens))

        full_conv = [{"role": "system", "content": system_message}]

        for conv in source:
            role = roles.get(conv.get("from", conv.get("role")))
            content = conv.get("value", conv.get("content", ""))

            # Expand `<image>` placeholders in user messages.
            if role == "user" and "<image>" in content:
                parts = content.split("<image>")
                new_parts = []

                for _ in range(len(parts) - 1):
                    new_parts.append(parts[_])

                    if visual_replicate_index_image < len(grid_thw_image):
                        replacement = (
                            "<|vision_start|>"
                            + ("<|image_pad|>" * grid_thw_image[visual_replicate_index_image])
                            + "<|vision_end|>"
                        )
                        visual_replicate_index_image += 1
                    else:
                        # If placeholders exceed actual images, keep as-is.
                        replacement = "<image>"

                    new_parts.append(replacement)

                new_parts.append(parts[-1])
                content = "".join(new_parts)

            full_conv.append({"role": role, "content": content})

        conv_formatted = tokenizer.apply_chat_template(
            full_conv, add_generation_prompt=False, tokenize=False
        )
        conv_tokens = tokenizer.encode(conv_formatted, add_special_tokens=False)
        current_input_ids.extend(conv_tokens)

    # Return raw token ids (no padding).
    return {"input_ids": current_input_ids}


# -----------------------------------------------------------------------------
# RoPE index computation
# -----------------------------------------------------------------------------

def get_rope_index_25(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D RoPE index based on image/video temporal-height-width layout.

    Returns:
        position_ids: (3, batch_size, sequence_length)
        mrope_position_deltas: (batch_size, 1)
    """
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652

    mrope_position_deltas = []

    has_vision = input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None)
    if has_vision:
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)

        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)

        for i, ids in enumerate(total_input_ids):
            ids = ids[attention_mask[i] == 1]

            vision_start_indices = torch.argwhere(ids == vision_start_token_id).squeeze(1)
            vision_tokens = ids[vision_start_indices + 1]

            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()

            input_tokens = ids.tolist()
            llm_pos_ids_list: list = []

            st = 0
            remain_images, remain_videos = image_nums, video_nums

            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1

                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1

                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    second_per_grid_t = (
                        second_per_grid_ts[video_index] if second_per_grid_ts is not None else 1.0
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                llm_grid_t = t.item()
                llm_grid_h = h.item() // spatial_merge_size
                llm_grid_w = w.item() // spatial_merge_size

                text_len = ed - st
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0

                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                # Keep the original behavior (note the constant `2`).
                time_tensor = expanded_range * second_per_grid_t * 2
                t_index = time_tensor.long().flatten()

                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )

                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )

                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)

        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        return position_ids, mrope_position_deltas

    position_ids = (
        torch.arange(input_ids.shape[1], device=input_ids.device)
        .view(1, 1, -1)
        .expand(3, input_ids.shape[0], -1)
    )
    mrope_position_deltas = torch.zeros(
        [input_ids.shape[0], 1],
        device=input_ids.device,
        dtype=input_ids.dtype,
    )
    return position_ids, mrope_position_deltas
