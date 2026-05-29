#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""CNN Behavioral Cloning Policy

A simple policy that uses a ResNet18 backbone to compute image features and has fully connected
layers to predict action chunks.
"""

from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn

from lerobot.policies.cnn_bc.configuration_cnn_bc import CNNBCConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def _build_cnn_backbone(name: str, weights: str | None) -> tuple[nn.Module, int]:
    """Build a torchvision CNN backbone with its classification head stripped.

    Returns the backbone (producing a pooled feature vector ``(B, embed_dim)`` or a
    pre-pool feature map that is flattened by the caller) and the embedding dimension.
    """
    model = torchvision.models.get_model(name, weights=weights)

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        embed_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Linear):
            embed_dim = classifier.in_features
        elif isinstance(classifier, nn.Sequential):
            last_linear = None
            for m in classifier:
                if isinstance(m, nn.Linear):
                    last_linear = m
            if last_linear is None:
                raise ValueError(
                    f"Could not find a Linear layer in classifier of backbone '{name}'."
                )
            embed_dim = last_linear.in_features
        else:
            raise ValueError(
                f"Unsupported classifier type {type(classifier).__name__} for backbone '{name}'."
            )
        model.classifier = nn.Identity()
    else:
        raise ValueError(
            f"Cannot strip classification head from torchvision backbone '{name}': "
            "no `fc` or `classifier` attribute found."
        )

    return model, embed_dim


class CNNBCPolicy(PreTrainedPolicy):
    """CNN Behavioral Cloning Policy.

    Uses a ResNet18 backbone to extract image features and feeds them (together with optional robot
    proprioceptive state) through fully connected layers to predict a chunk of future actions.
    """

    config_class = CNNBCConfig
    name = "cnn_bc"

    def __init__(self, config: CNNBCConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = CNNBCNet(config)
        self.reset()

    def get_optim_params(self) -> dict:
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

        # State history: each entry is a (state_dim,) tensor from a previous step.
        state_dim = self.config.robot_state_feature.shape[0] if self.config.robot_state_feature else 0
        if self.config.state_history_size > 0 and state_dim > 0:
            self._state_history: deque[Tensor] = deque(
                [torch.zeros(state_dim) for _ in range(self.config.state_history_size)],
                maxlen=self.config.state_history_size,
            )
        else:
            self._state_history = deque(maxlen=0)

        # Action history: each entry is a (action_dim,) tensor (first action of a past chunk).
        action_dim = self.config.action_feature.shape[0]
        if self.config.action_history_size > 0:
            self._action_history: deque[Tensor] = deque(
                [torch.zeros(action_dim) for _ in range(self.config.action_history_size)],
                maxlen=self.config.action_history_size,
            )
        else:
            self._action_history = deque(maxlen=0)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        Manages an internal action queue: when the queue is empty the policy is queried for
        a full action chunk and the queue is populated from that chunk.

        State history is updated at every call so that the most recent X states are available
        the next time the model is queried.  Action history is updated each time a new chunk is
        predicted (the first action of the chunk is stored).
        """
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(self._build_batch_with_history(batch))[
                :, : self.config.n_action_steps
            ]
            # actions: (B, n_action_steps, action_dim) → enqueue as (n_action_steps, B, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

            # Store the first action of this chunk in the action history (B=1 assumed at inference).
            if self.config.action_history_size > 0:
                self._action_history.append(actions[0, 0].detach())

        # Record the current state so subsequent model queries can use it as context.
        if self._state_history.maxlen and self.config.robot_state_feature is not None:
            self._state_history.append(batch[OBS_STATE][0].detach())

        return self._action_queue.popleft()

    def _build_batch_with_history(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Return a copy of *batch* enriched with flattened history tensors.

        Keys added (when the corresponding history is configured and non-empty):
            - ``"observation.state_history"``: ``(1, state_history_size * state_dim)``
            - ``"observation.action_history"``: ``(1, action_history_size * action_dim)``
        """
        if not self._state_history and not self._action_history:
            return batch  # nothing to add

        batch = dict(batch)  # shallow copy – do not mutate the caller's dict
        device = next(iter(batch.values())).device

        if self._state_history:
            state_hist = torch.stack(list(self._state_history)).to(device)  # (size, state_dim)
            batch["observation.state_history"] = state_hist.flatten().unsqueeze(0)  # (1, size*state_dim)

        if self._action_history:
            action_hist = torch.stack(list(self._action_history)).to(device)  # (size, action_dim)
            batch["observation.action_history"] = action_hist.flatten().unsqueeze(0)  # (1, size*action_dim)

        return batch

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        if self.config.image_features:
            batch = dict(batch)  # shallow copy
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        return self.model(batch)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)  # shallow copy
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat = self.model(batch)

        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        return l1_loss, {"l1_loss": l1_loss.item()}


class CNNBCNet(nn.Module):
    """Core neural network for CNN Behavioral Cloning.

    Architecture:
        1. For each camera image: ResNet18 backbone → global average pooling → flat feature vector.
        2. Concatenate all image feature vectors (and optional robot state).
        3. Optionally append flattened state history and action history vectors.
        4. Fully connected layers (Linear → ReLU → Dropout) to predict a flat action chunk.
        5. Reshape output to (batch, chunk_size, action_dim).
    """

    def __init__(self, config: CNNBCConfig):
        super().__init__()
        self.config = config

        if config.image_features:
            self.backbone, backbone_out_channels = _build_cnn_backbone(
                config.vision_backbone, config.pretrained_backbone_weights
            )
        else:
            backbone_out_channels = 0

        # Determine FC head input dimension.
        num_cameras = len(config.image_features) if config.image_features else 0
        input_dim = backbone_out_channels * num_cameras
        if config.robot_state_feature is not None:
            input_dim += config.robot_state_feature.shape[0]

        # Add history dimensions.
        state_dim = config.robot_state_feature.shape[0] if config.robot_state_feature else 0
        if config.state_history_size > 0 and state_dim > 0:
            input_dim += config.state_history_size * state_dim
        action_dim = config.action_feature.shape[0]
        if config.action_history_size > 0:
            input_dim += config.action_history_size * action_dim

        # Build fully connected head.
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                ]
            )
            prev_dim = hidden_dim

        action_dim = config.action_feature.shape[0]
        layers.append(nn.Linear(prev_dim, config.chunk_size * action_dim))

        self.fc = nn.Sequential(*layers)
        self._action_dim = action_dim
        self._chunk_size = config.chunk_size

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Args:
            batch: Dictionary containing:
                - ``"observation.images"`` (list of Tensor, each ``(B, C, H, W)``): camera images.
                - ``"observation.state"`` (Tensor ``(B, state_dim)``): robot proprioceptive state
                  (optional, only if configured).
                - ``"observation.state_history"`` (Tensor ``(B, state_history_size * state_dim)``):
                  flattened past states (optional; zeros used when absent).
                - ``"observation.action_history"`` (Tensor ``(B, action_history_size * action_dim)``):
                  flattened past actions (optional; zeros used when absent).

        Returns:
            Tensor of shape ``(B, chunk_size, action_dim)`` with predicted action chunks.
        """
        features: list[Tensor] = []

        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                x = self.backbone(img)  # (B, C_out)
                if x.dim() == 4:
                    x = x.flatten(1)
                features.append(x)

        if self.config.robot_state_feature is not None:
            features.append(batch[OBS_STATE])

        # History context — use zeros when not provided (e.g. during training).
        if self.config.state_history_size > 0 and self.config.robot_state_feature is not None:
            state_dim = self.config.robot_state_feature.shape[0]
            hist_dim = self.config.state_history_size * state_dim
            if "observation.state_history" in batch:
                features.append(batch["observation.state_history"])
            else:
                ref = features[0]
                features.append(torch.zeros(ref.shape[0], hist_dim, device=ref.device, dtype=ref.dtype))

        if self.config.action_history_size > 0:
            hist_dim = self.config.action_history_size * self._action_dim
            if "observation.action_history" in batch:
                features.append(batch["observation.action_history"])
            else:
                ref = features[0]
                features.append(torch.zeros(ref.shape[0], hist_dim, device=ref.device, dtype=ref.dtype))

        x = torch.cat(features, dim=-1)  # (B, input_dim)
        x = self.fc(x)  # (B, chunk_size * action_dim)
        return x.reshape(x.shape[0], self._chunk_size, self._action_dim)
