#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("dot")
@dataclass
class DOTConfig(PreTrainedConfig):
    """Configuration for DOT (Decoder-Only Transformer) policy."""

    # Input / output structure.
    n_obs_steps: int = 3
    train_horizon: int = 20
    inference_horizon: int = 20
    lookback_obs_steps: int = 10
    lookback_aug: int = 5

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Architecture.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    pre_norm: bool = True
    lora_rank: int = 20
    merge_lora: bool = False

    dim_model: int = 128
    n_heads: int = 8
    dim_feedforward: int = 512
    n_decoder_layers: int = 8
    rescale_shape: tuple[int, int] = (96, 96)

    # Augmentation.
    crop_scale: float = 0.8
    state_noise: float = 0.01
    noise_decay: float = 0.999995

    # Training and loss computation.
    dropout: float = 0.1

    # Weighting and inference.
    alpha: float = 0.75
    train_alpha: float = 0.9
    predict_every_n: int = 1
    return_every_n: int = 1

    # Training preset
    optimizer_lr: float = 1.0e-4
    optimizer_min_lr: float = 1.0e-4
    optimizer_lr_cycle_steps: int = 300000
    optimizer_weight_decay: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

        if self.predict_every_n > self.inference_horizon:
            raise ValueError(
                f"predict_every_n ({self.predict_every_n}) must be <= inference_horizon ({self.inference_horizon})."
            )
        if self.return_every_n > self.inference_horizon:
            raise ValueError(
                f"return_every_n ({self.return_every_n}) must be <= inference_horizon ({self.inference_horizon})."
            )
        if self.predict_every_n > self.inference_horizon // self.return_every_n:
            raise ValueError(
                f"predict_every_n ({self.predict_every_n}) must be <= "
                f"inference_horizon // return_every_n ({self.inference_horizon // self.return_every_n})."
            )
        if self.train_horizon < self.inference_horizon:
            raise ValueError(
                f"train_horizon ({self.train_horizon}) must be >= inference_horizon ({self.inference_horizon})."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.optimizer_min_lr,
            num_warmup_steps=0,
            num_decay_steps=self.optimizer_lr_cycle_steps,
        )

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    @property
    def observation_delta_indices(self) -> list[int] | None:
        far_past_obs = list(
            range(
                -self.lookback_aug - self.lookback_obs_steps,
                self.lookback_aug + 1 - self.lookback_obs_steps,
            )
        )
        recent_obs = list(range(2 - self.n_obs_steps, 1))

        return far_past_obs + recent_obs

    @property
    def action_delta_indices(self) -> list[int]:
        far_past_actions = list(
            range(
                -self.lookback_aug - self.lookback_obs_steps,
                self.lookback_aug + 1 - self.lookback_obs_steps,
            )
        )
        recent_actions = list(range(2 - self.n_obs_steps, self.train_horizon))

        return far_past_actions + recent_actions

    @property
    def reward_delta_indices(self) -> None:
        return None
