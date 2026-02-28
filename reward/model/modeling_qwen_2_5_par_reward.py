from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLTextModel
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VisionTransformerPretrainedModel,
)

# -----------------------------------------------------------------------------
# Loss functions
# -----------------------------------------------------------------------------


def LossPair_BT(
    scores_chosen: torch.Tensor, scores_rejected: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bradley-Terry pairwise loss and accuracy for reward comparison."""
    reward_diff = scores_chosen - scores_rejected
    losses = -F.logsigmoid(reward_diff)
    total_loss = losses.mean()
    accuracies = (reward_diff > 0).float().mean(dim=0)
    return total_loss, accuracies


def LossPair_CERanking(reward: torch.Tensor) -> torch.Tensor:
    """Cross-entropy ranking loss with the first item as the positive target."""
    target = torch.zeros(reward.shape[0], dtype=torch.long, device=reward.device)
    loss_list = F.cross_entropy(reward, target, reduction="none")
    return loss_list.mean()


def LossPoint_MSE(scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
    """Pointwise MSE loss for reward regression."""
    target_scores = target_scores.to(device=scores.device, dtype=scores.dtype)
    return F.mse_loss(scores, target_scores)


def LossPoint_MAE(scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
    """Pointwise MAE loss for reward regression."""
    target_scores = target_scores.to(device=scores.device, dtype=scores.dtype)
    return F.l1_loss(scores, target_scores)


def LossCausalLM_CE(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Standard causal LM cross-entropy with label shift."""
    logits = logits.float()
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    vocab_size = shift_logits.shape[-1]
    return F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------


class MultiHeadMLP(nn.Module):
    """A small MLP head per objective; returns concatenated scalar rewards."""

    def __init__(self, hidden_size: int, num_objectives: int = 2) -> None:
        super().__init__()
        self.reward_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                )
                for _ in range(num_objectives)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        scores = [head(hidden_states) for head in self.reward_heads]
        return torch.cat(scores, dim=-1)


class RMSNorm(nn.Module):
    """Simple RMSNorm with a learnable scale parameter."""

    def __init__(self, dims: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dims))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x) * self.weight


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


class RewardModelQwenConfig(Qwen2_5_VLConfig):
    model_type = "reward_model_qwen"


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class RewardModelQwenForCausalLM(Qwen2_5_VLPreTrainedModel):
    config_class = RewardModelQwenConfig

    def __init__(self, config: RewardModelQwenConfig) -> None:
        super().__init__(config)
        config.model_type = "reward_model_qwen"

        # Match upstream design: vision uses vision_config; text uses text_config.
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(
            config.vision_config
        )
        self.model = Qwen2_5_VLTextModel._from_config(config.text_config)

        hidden_size = config.text_config.hidden_size
        self.multiheads = MultiHeadMLP(hidden_size)
        self.metavoter = nn.Sequential(
            nn.Linear(hidden_size, 256),
            RMSNorm(256, eps=1e-5),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.rope_deltas = None

        self.metavoter.apply(self._init_weights)
        self.multiheads.apply(self._init_weights)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize newly-added heads/norms while leaving base model untouched."""
        if isinstance(module, nn.Linear):
            std = getattr(self.config, "initializer_range", 0.02)
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    # -------------------------------------------------------------------------
    # Upstream-compatible utilities: placeholder masks + image/video features
    # -------------------------------------------------------------------------

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return expanded masks for <image>/<video> placeholder tokens."""
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id

        special_image_mask = input_ids == image_token_id
        special_video_mask = input_ids == video_token_id

        n_image_tokens = special_image_mask.sum()
        image_mask = (
            special_image_mask.unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        if image_features is not None and inputs_embeds[image_mask].numel() != image_features.numel():
            raise ValueError(
                "Image features and image tokens do not match: "
                f"tokens: {n_image_tokens}, features {image_features.shape}"
            )

        n_video_tokens = special_video_mask.sum()
        video_mask = (
            special_video_mask.unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        if video_features is not None and inputs_embeds[video_mask].numel() != video_features.numel():
            raise ValueError(
                "Video features and video tokens do not match: "
                f"tokens: {n_video_tokens}, features {video_features.shape}"
            )

        return image_mask, video_mask

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, ...]:
        """Compute and split image embeddings per image (dynamic resolution)."""
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (
            image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2
        ).tolist()
        return torch.split(image_embeds, split_sizes)

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, ...]:
        """Compute and split video embeddings per video (dynamic resolution)."""
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
        split_sizes = (
            video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2
        ).tolist()
        return torch.split(video_embeds, split_sizes)

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate 3D RoPE position ids for Qwen2.5-VL (vision + text) sequences.

        Returns:
            position_ids: (3, batch_size, seq_len)
            mrope_position_deltas: (batch_size, 1)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        mrope_position_deltas = []

        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
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
                llm_pos_ids_list: list[torch.Tensor] = []

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
                            second_per_grid_ts[video_index]
                            if second_per_grid_ts is not None
                            else 1.0
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

                    time_tensor = (
                        expanded_range
                        * second_per_grid_t
                        * self.config.vision_config.tokens_per_second
                    )
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

            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas

        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
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

    # -------------------------------------------------------------------------
    # forward/predict (non-packed mode only)
    # -------------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Qwen2_5_VLCausalLMOutputWithPast:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        if attention_mask.ndim != 2:
            raise ValueError(
                "Only non-packed mode supported. attention_mask must be [B, L], "
                f"got {tuple(attention_mask.shape)}"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("pixel_values is not None but image_grid_thw is None")

            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            if video_grid_thw is None:
                raise ValueError(
                    "pixel_values_videos is not None but video_grid_thw is None"
                )

            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        attention_mask = attention_mask.to(inputs_embeds.device)

        if position_ids is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]  # [B, L, H]
        logits_multi_reward_all = self.multiheads(hidden_states)  # [B, L, 2]
        logits_meta_all = self.metavoter(hidden_states)  # [B, L, 1]

        lengths = attention_mask.long().sum(dim=1)  # [B]
        end_pos = lengths - 1
        bidx = torch.arange(hidden_states.size(0), device=hidden_states.device)

        multi_reward_logits = logits_multi_reward_all[bidx, end_pos]  # [B, 2]
        reward_logits = logits_meta_all[bidx, end_pos]  # [B, 1]
        logits = torch.cat([multi_reward_logits, reward_logits], dim=-1)  # [B, 3]

        loss = None
        if labels is not None:
            loss_type_ids = labels[:, 0].long()
            target_scores = labels[:, 1:]  # [B, 3]

            total_loss = torch.tensor(0.0, device=logits.device)

            mse_mask = (loss_type_ids == 0) | (loss_type_ids == 2)
            if mse_mask.any():
                total_loss = total_loss + 0.5 * LossPoint_MSE(
                    logits[mse_mask], target_scores[mse_mask]
                )

            bt_indices = torch.where((loss_type_ids == 1) | (loss_type_ids == 2))[0]
            if len(bt_indices) % 2 != 0:
                raise ValueError(
                    "BT samples must come in chosen/rejected pairs in the batch."
                )
            if len(bt_indices) > 0:
                pairs = bt_indices.view(-1, 2)
                chosen_idx, rejected_idx = pairs[:, 0], pairs[:, 1]
                bt_loss, _ = LossPair_BT(
                    reward_logits[chosen_idx], reward_logits[rejected_idx]
                )
                total_loss = total_loss + 0.5 * bt_loss

            loss = total_loss

        if not return_dict:
            raise NotImplementedError

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

    @torch.inference_mode()
    def predict(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        if attention_mask.ndim != 2:
            raise ValueError(
                "predict() only supports non-packed mode. attention_mask must be [B, L], "
                f"got {tuple(attention_mask.shape)}"
            )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("pixel_values is not None but image_grid_thw is None")

            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        attention_mask = attention_mask.to(inputs_embeds.device)

        if position_ids is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]  # [B, L, H]
        logits_multi_reward_all = self.multiheads(hidden_states)  # [B, L, 2]
        logits_meta_all = self.metavoter(hidden_states)  # [B, L, 1]

        lengths = attention_mask.long().sum(dim=1)
        end_pos = lengths - 1
        bidx = torch.arange(hidden_states.size(0), device=hidden_states.device)

        multi_reward_logits = logits_multi_reward_all[bidx, end_pos]  # [B, 2]
        reward_logits = logits_meta_all[bidx, end_pos]  # [B, 1]
        return multi_reward_logits, reward_logits


AutoConfig.register("reward_model_qwen", RewardModelQwenConfig)
AutoModelForCausalLM.register(RewardModelQwenConfig, RewardModelQwenForCausalLM)
