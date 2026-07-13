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
"""Tests for ACT gripper-movement-based dynamic recomputation."""

from collections import deque
from unittest.mock import MagicMock, patch

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    use_gripper_recompute: bool = True,
    gripper_state_dim_idx: int = -1,
    gripper_recompute_threshold: float = 0.05,
    n_action_steps: int = 5,
    chunk_size: int = 10,
    temporal_ensemble_coeff: float | None = None,
) -> ACTConfig:
    """Return a minimal ACTConfig suitable for unit tests (state input, no images)."""
    cfg = ACTConfig()
    cfg.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
    }
    cfg.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
    }
    cfg.use_gripper_recompute = use_gripper_recompute
    cfg.gripper_state_dim_idx = gripper_state_dim_idx
    cfg.gripper_recompute_threshold = gripper_recompute_threshold
    cfg.n_action_steps = n_action_steps
    cfg.chunk_size = chunk_size
    cfg.temporal_ensemble_coeff = temporal_ensemble_coeff
    cfg.device = "cpu"
    return cfg


def _make_policy(cfg: ACTConfig):
    """Return a lightweight stand-in for ACTPolicy that exercises only the queue logic."""
    from lerobot.policies.act.modeling_act import ACTPolicy

    with patch("lerobot.policies.act.modeling_act.ACT"):
        policy = ACTPolicy.__new__(ACTPolicy)
        # Initialise nn.Module bookkeeping so eval() and parameter traversal work.
        torch.nn.Module.__init__(policy)
        policy.config = cfg
        policy.reset()
    return policy


def _make_batch(state: list[float]) -> dict[str, torch.Tensor]:
    """Build a minimal batch dict with a single observation state."""
    return {OBS_STATE: torch.tensor([state], dtype=torch.float32)}  # shape (1, state_dim)


def _stub_predict_action_chunk(policy, actions_2d: list[list[float]]) -> None:
    """Make ``predict_action_chunk`` return a fixed chunk of actions (batch=1)."""
    tensor = torch.tensor([actions_2d], dtype=torch.float32)  # (1, n_steps, action_dim)
    policy.predict_action_chunk = MagicMock(return_value=tensor)


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


def test_config_default_values():
    """use_gripper_recompute defaults to False, leaving standard behaviour intact."""
    cfg = ACTConfig()
    assert cfg.use_gripper_recompute is False
    assert cfg.gripper_state_dim_idx == -1
    assert cfg.gripper_recompute_threshold == 0.05


def test_config_rejects_gripper_recompute_with_temporal_ensemble():
    """Combining use_gripper_recompute with temporal_ensemble_coeff must raise."""
    with pytest.raises(ValueError, match="temporal_ensemble_coeff"):
        ACTConfig(
            use_gripper_recompute=True,
            temporal_ensemble_coeff=0.01,
            n_action_steps=1,  # required by temporal ensembling
        )


def test_validate_features_rejects_missing_state():
    """validate_features must raise when observation.state is absent but gripper recompute is on."""
    cfg = ACTConfig()
    # Provide env_state so the first check passes, but omit observation.state
    cfg.input_features = {
        OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
    }
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,))}
    cfg.use_gripper_recompute = True

    with pytest.raises(ValueError, match="observation.state"):
        cfg.validate_features()


# ---------------------------------------------------------------------------
# Behavioural tests
# ---------------------------------------------------------------------------


def test_reset_clears_gripper_state():
    """After reset, _prev_gripper_state should be None."""
    cfg = _make_config()
    policy = _make_policy(cfg)
    # Simulate a stored gripper state from a previous episode
    policy._prev_gripper_state = torch.tensor([0.5])
    policy.reset()
    assert policy._prev_gripper_state is None


def test_no_recompute_below_threshold():
    """Queue is not cleared when the gripper change is below the threshold."""
    cfg = _make_config(
        use_gripper_recompute=True,
        gripper_state_dim_idx=-1,
        gripper_recompute_threshold=0.10,
        n_action_steps=5,
        chunk_size=10,
    )
    policy = _make_policy(cfg)

    dummy_actions = [[0.1, 0.2, 0.3, 0.4]] * cfg.n_action_steps

    # Set up a single mock that will be reused across both calls
    _stub_predict_action_chunk(policy, dummy_actions)

    # First call – fills the queue
    batch_1 = _make_batch([0.1, 0.2, 0.3, 0.0])
    policy.select_action(batch_1)
    assert len(policy._action_queue) == cfg.n_action_steps - 1  # one action consumed
    assert policy.predict_action_chunk.call_count == 1

    # Second call – small gripper change (0.04 < 0.10 threshold), queue not cleared
    batch_2 = _make_batch([0.1, 0.2, 0.3, 0.04])
    policy.select_action(batch_2)

    # predict_action_chunk should NOT have been called again
    assert policy.predict_action_chunk.call_count == 1  # still 1
    assert len(policy._action_queue) == cfg.n_action_steps - 2  # another action consumed from queue


def test_recompute_triggered_above_threshold():
    """Queue is cleared and recomputation occurs when gripper change exceeds threshold."""
    cfg = _make_config(
        use_gripper_recompute=True,
        gripper_state_dim_idx=-1,
        gripper_recompute_threshold=0.05,
        n_action_steps=5,
        chunk_size=10,
    )
    policy = _make_policy(cfg)

    dummy_actions = [[0.1, 0.2, 0.3, 0.4]] * cfg.n_action_steps
    _stub_predict_action_chunk(policy, dummy_actions)

    # First call – fills the queue
    batch_1 = _make_batch([0.1, 0.2, 0.3, 0.0])
    policy.select_action(batch_1)
    assert len(policy._action_queue) == cfg.n_action_steps - 1
    assert policy.predict_action_chunk.call_count == 1

    # Second call – large gripper change (0.2 >> 0.05 threshold) → queue cleared, recompute
    batch_2 = _make_batch([0.1, 0.2, 0.3, 0.2])
    policy.select_action(batch_2)

    # predict_action_chunk must have been called a second time
    assert policy.predict_action_chunk.call_count == 2
    # After recompute, (n_action_steps - 1) items remain (one was just popped)
    assert len(policy._action_queue) == cfg.n_action_steps - 1


def test_gripper_recompute_updates_prev_state():
    """_prev_gripper_state is updated to the latest observed gripper value."""
    cfg = _make_config(
        use_gripper_recompute=True,
        gripper_state_dim_idx=2,  # use index 2 explicitly
        gripper_recompute_threshold=0.05,
    )
    policy = _make_policy(cfg)

    dummy_actions = [[0.1, 0.2, 0.3, 0.4]] * cfg.n_action_steps
    _stub_predict_action_chunk(policy, dummy_actions)

    batch = _make_batch([0.5, 0.6, 0.77, 0.8])
    policy.select_action(batch)

    expected = torch.tensor([0.77])
    assert torch.allclose(policy._prev_gripper_state, expected)


def test_no_gripper_recompute_when_disabled():
    """When use_gripper_recompute=False, queue behaviour is unchanged even for large gripper change."""
    cfg = _make_config(
        use_gripper_recompute=False,
        gripper_recompute_threshold=0.05,
        n_action_steps=5,
        chunk_size=10,
    )
    policy = _make_policy(cfg)

    dummy_actions = [[0.1, 0.2, 0.3, 0.4]] * cfg.n_action_steps
    _stub_predict_action_chunk(policy, dummy_actions)

    batch_1 = _make_batch([0.1, 0.2, 0.3, 0.0])
    policy.select_action(batch_1)
    assert policy.predict_action_chunk.call_count == 1

    # Large gripper change – but feature is disabled, so no recompute
    batch_2 = _make_batch([0.1, 0.2, 0.3, 0.9])
    policy.select_action(batch_2)

    assert policy.predict_action_chunk.call_count == 1  # only the initial fill


def test_gripper_recompute_skipped_when_obs_state_absent():
    """If OBS_STATE is absent from batch, recompute logic is silently skipped."""
    cfg = _make_config(use_gripper_recompute=True)
    policy = _make_policy(cfg)

    dummy_actions = [[0.1, 0.2, 0.3, 0.4]] * cfg.n_action_steps
    _stub_predict_action_chunk(policy, dummy_actions)

    # Batch without OBS_STATE key
    batch_no_state = {ACTION: torch.zeros(1, 4)}
    policy.select_action(batch_no_state)  # should not raise

    assert policy._prev_gripper_state is None  # never set

