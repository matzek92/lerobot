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

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import PREV_REMAINING_TRAJ, ACTPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE


def make_act_config(*, use_prev_remaining_traj: bool, n_action_steps: int = 1) -> ACTConfig:
    return ACTConfig(
        chunk_size=4,
        n_action_steps=n_action_steps,
        use_vae=False,
        dim_model=16,
        n_heads=4,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        input_features={OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(3,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        use_prev_remaining_traj=use_prev_remaining_traj,
    )


def test_act_default_does_not_use_prev_remaining_traj(monkeypatch):
    policy = ACTPolicy(make_act_config(use_prev_remaining_traj=False))
    seen_prev_remaining = None

    def fake_forward(batch):
        nonlocal seen_prev_remaining
        seen_prev_remaining = PREV_REMAINING_TRAJ in batch
        actions = torch.zeros((1, policy.config.chunk_size, policy.config.action_feature.shape[0]))
        return actions, (None, None)

    monkeypatch.setattr(policy.model, "forward", fake_forward)
    policy.predict_action_chunk({OBS_ENV_STATE: torch.zeros(1, 3)})

    assert seen_prev_remaining is False
    assert not hasattr(policy.model, "prev_remaining_action_input_proj")


def test_act_prev_remaining_traj_used_in_training_forward(monkeypatch):
    policy = ACTPolicy(make_act_config(use_prev_remaining_traj=True))
    seen_prev_remaining_trajs = []
    predicted_chunk = torch.tensor(
        [[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]], dtype=torch.float32
    )

    def fake_forward(batch):
        seen_prev_remaining_trajs.append(batch[PREV_REMAINING_TRAJ].clone())
        return predicted_chunk.clone(), (None, None)

    monkeypatch.setattr(policy.model, "forward", fake_forward)

    batch = {
        OBS_ENV_STATE: torch.zeros(1, 3),
        ACTION: torch.zeros(1, policy.config.chunk_size, policy.config.action_feature.shape[0]),
        "action_is_pad": torch.zeros(1, policy.config.chunk_size, dtype=torch.bool),
    }
    policy.forward(batch)

    assert len(seen_prev_remaining_trajs) == 2
    expected_first = torch.zeros_like(predicted_chunk)
    expected_second = torch.tensor([[[2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [0.0, 0.0]]], dtype=torch.float32)
    torch.testing.assert_close(seen_prev_remaining_trajs[0], expected_first)
    torch.testing.assert_close(seen_prev_remaining_trajs[1], expected_second)


def test_act_prev_remaining_traj_used_in_inference_and_tracks_remaining(monkeypatch):
    policy = ACTPolicy(make_act_config(use_prev_remaining_traj=True, n_action_steps=1))
    seen_prev_remaining_trajs = []
    predicted_chunks = [
        torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]], dtype=torch.float32),
        torch.tensor([[[9.0, 90.0], [8.0, 80.0], [7.0, 70.0], [6.0, 60.0]]], dtype=torch.float32),
    ]

    def fake_forward(batch):
        seen_prev_remaining_trajs.append(batch[PREV_REMAINING_TRAJ].clone())
        return predicted_chunks[len(seen_prev_remaining_trajs) - 1].clone(), (None, None)

    monkeypatch.setattr(policy.model, "forward", fake_forward)

    observation = {OBS_ENV_STATE: torch.zeros(1, 3)}
    policy.select_action(observation)
    policy.select_action(observation)

    expected_first = torch.zeros_like(predicted_chunks[0])
    expected_second = torch.tensor([[[2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [0.0, 0.0]]], dtype=torch.float32)
    torch.testing.assert_close(seen_prev_remaining_trajs[0], expected_first)
    torch.testing.assert_close(seen_prev_remaining_trajs[1], expected_second)
    torch.testing.assert_close(policy._prev_action_chunk, predicted_chunks[-1])


def test_act_prev_remaining_traj_reset_clears_cached_chunk(monkeypatch):
    policy = ACTPolicy(make_act_config(use_prev_remaining_traj=True, n_action_steps=1))
    seen_prev_remaining_trajs = []

    def fake_forward(batch):
        seen_prev_remaining_trajs.append(batch[PREV_REMAINING_TRAJ].clone())
        actions = torch.ones((1, policy.config.chunk_size, policy.config.action_feature.shape[0]))
        return actions, (None, None)

    monkeypatch.setattr(policy.model, "forward", fake_forward)

    observation = {OBS_ENV_STATE: torch.zeros(1, 3)}
    policy.select_action(observation)
    policy.reset()
    policy.select_action(observation)

    expected = torch.zeros((1, policy.config.chunk_size, policy.config.action_feature.shape[0]))
    torch.testing.assert_close(seen_prev_remaining_trajs[0], expected)
    torch.testing.assert_close(seen_prev_remaining_trajs[1], expected)
