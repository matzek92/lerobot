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
"""Tests for the Octo policy adapter.

These tests validate OctoPolicy without requiring JAX or Octo to be installed.
The JAX-based OctoModel is replaced by a lightweight mock so that the adapter
logic (observation conversion, history buffering, action queuing, factory
integration, save/load) can be exercised in a standard Python environment.
"""

import json
import tempfile
import types
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.octo.configuration_octo import OctoConfig
from lerobot.policies.octo.modeling_octo import OctoPolicy, _get_batch_size, _tensor_to_octo_image
from lerobot.policies.octo.processor_octo import make_octo_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ACTION_DIM = 7
ACTION_HORIZON = 4
BATCH_SIZE = 1


def _make_config(**kwargs) -> OctoConfig:
    """Return an OctoConfig with no checkpoint path (so no JAX is loaded)."""
    defaults = dict(
        octo_checkpoint="",
        primary_image_key="observation.images.top",
        task_text="pick up the spoon",
        n_action_steps=ACTION_HORIZON,
        action_horizon=ACTION_HORIZON,
        window_size=2,
        image_size=(256, 256),
        seed=0,
    )
    defaults.update(kwargs)
    return OctoConfig(**defaults)


def _make_batch(batch_size: int = BATCH_SIZE) -> dict[str, torch.Tensor]:
    """Create a minimal observation batch as the preprocessor would produce it."""
    return {
        "observation.images.top": torch.rand(batch_size, 3, 84, 84),
        OBS_STATE: torch.rand(batch_size, 6),
    }


def _make_mock_octo_model(action_dim: int = ACTION_DIM) -> MagicMock:
    """Return a minimal mock of OctoModel that returns random actions."""
    model = MagicMock()
    model.dataset_statistics = {
        "bridge_dataset": {
            "action": {
                "mean": np.zeros(action_dim),
                "std": np.ones(action_dim),
            }
        }
    }
    model.create_tasks.return_value = {"language_instruction": ["pick up the spoon"]}
    # sample_actions returns (batch, action_horizon, action_dim)
    model.sample_actions.return_value = np.random.rand(BATCH_SIZE, ACTION_HORIZON, action_dim).astype(
        np.float32
    )
    return model


def _make_policy_with_mock_model(**cfg_kwargs) -> OctoPolicy:
    """Create an OctoPolicy and inject a mock OctoModel."""
    cfg = _make_config(**cfg_kwargs)
    policy = OctoPolicy(cfg)
    policy._octo_model = _make_mock_octo_model()

    # Provide a real JAX-like PRNGKey split that returns two arrays
    def _mock_split(rng):
        return rng, rng

    # Patch jax inside the modeling_octo module so no real jax is needed
    mock_jax = types.SimpleNamespace(
        random=types.SimpleNamespace(
            PRNGKey=lambda seed: np.array([0, seed], dtype=np.uint32),
            split=lambda rng: (rng, rng),
        )
    )
    policy._rng = np.array([0, 0], dtype=np.uint32)
    # Patch jax at the modeling level so predict_action_chunk can run
    with patch.dict("sys.modules", {"jax": mock_jax}):
        pass  # just verifying it's importable; actual patching happens in tests

    return policy


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------


class TestOctoConfig:
    def test_default_values(self):
        cfg = OctoConfig()
        assert cfg.type == "octo"
        assert cfg.octo_checkpoint == "hf://rail-berkeley/octo-small-1.5"
        assert cfg.window_size == 1
        assert cfg.n_action_steps == 1
        assert cfg.action_horizon == 4
        assert cfg.image_size == (256, 256)
        assert cfg.argmax is False
        assert cfg.seed == 42
        assert cfg.unnorm_key is None

    def test_custom_values(self):
        cfg = OctoConfig(
            octo_checkpoint="hf://rail-berkeley/octo-base-1.5",
            window_size=2,
            n_action_steps=2,
            action_horizon=4,
            task_text="grasp the cup",
            unnorm_key="bridge_dataset",
        )
        assert cfg.octo_checkpoint == "hf://rail-berkeley/octo-base-1.5"
        assert cfg.window_size == 2
        assert cfg.task_text == "grasp the cup"
        assert cfg.unnorm_key == "bridge_dataset"

    def test_validation_n_action_steps_too_large(self):
        with pytest.raises(ValueError, match="n_action_steps"):
            OctoConfig(n_action_steps=10, action_horizon=4)

    def test_validation_window_size_zero(self):
        with pytest.raises(ValueError, match="window_size"):
            OctoConfig(window_size=0)

    def test_delta_indices(self):
        cfg = OctoConfig(action_horizon=4)
        assert cfg.action_delta_indices == [0, 1, 2, 3]
        assert cfg.observation_delta_indices is None
        assert cfg.reward_delta_indices is None

    def test_normalization_mapping_empty(self):
        cfg = OctoConfig()
        # Octo handles normalization internally; the mapping must be empty so that
        # LeRobot's NormalizerProcessorStep is not applied.
        assert cfg.normalization_mapping == {}

    def test_optimizer_preset_returns(self):
        cfg = OctoConfig()
        opt = cfg.get_optimizer_preset()
        assert opt is not None

    def test_scheduler_preset_none(self):
        cfg = OctoConfig()
        assert cfg.get_scheduler_preset() is None

    def test_validate_features_does_not_raise(self):
        cfg = OctoConfig()
        cfg.validate_features()  # should be a no-op


# ---------------------------------------------------------------------------
# Factory integration tests
# ---------------------------------------------------------------------------


class TestOctoFactory:
    def test_get_policy_class(self):
        cls = get_policy_class("octo")
        assert cls is OctoPolicy
        assert cls.name == "octo"
        assert cls.config_class is OctoConfig

    def test_make_policy_config(self):
        cfg = make_policy_config("octo")
        assert isinstance(cfg, OctoConfig)
        assert cfg.type == "octo"

    def test_make_policy_config_with_kwargs(self):
        cfg = make_policy_config("octo", octo_checkpoint="hf://rail-berkeley/octo-base-1.5")
        assert cfg.octo_checkpoint == "hf://rail-berkeley/octo-base-1.5"

    def test_make_pre_post_processors_from_factory(self):
        cfg = _make_config()
        pre, post = make_pre_post_processors(cfg)
        assert pre is not None
        assert post is not None


# ---------------------------------------------------------------------------
# Processor tests
# ---------------------------------------------------------------------------


class TestOctoProcessor:
    def test_make_processors_returns_tuple(self):
        cfg = _make_config()
        pre, post = make_octo_pre_post_processors(cfg)
        assert pre is not None
        assert post is not None

    def test_preprocessor_has_correct_steps(self):
        cfg = _make_config()
        pre, _ = make_octo_pre_post_processors(cfg)
        step_names = [type(s).__name__ for s in pre.steps]
        assert "AddBatchDimensionProcessorStep" in step_names
        assert "DeviceProcessorStep" in step_names
        # NormalizerProcessorStep must NOT be added (Octo handles normalization)
        assert "NormalizerProcessorStep" not in step_names

    def test_postprocessor_has_device_step(self):
        cfg = _make_config()
        _, post = make_octo_pre_post_processors(cfg)
        step_names = [type(s).__name__ for s in post.steps]
        assert "DeviceProcessorStep" in step_names

    def test_dataset_stats_ignored(self):
        """Passing dataset_stats should not cause an error (they are intentionally ignored)."""
        cfg = _make_config()
        fake_stats = {"observation.images.top": {"mean": torch.zeros(3), "std": torch.ones(3)}}
        pre, post = make_octo_pre_post_processors(cfg, dataset_stats=fake_stats)
        assert pre is not None


# ---------------------------------------------------------------------------
# Image conversion utilities
# ---------------------------------------------------------------------------


class TestTensorToOctoImage:
    def test_shape_and_dtype_float(self):
        """Float (B, C, H, W) → uint8 (B, H, W, C) at target_size."""
        img = torch.rand(2, 3, 84, 84)
        out = _tensor_to_octo_image(img, (256, 256))
        assert out.shape == (2, 256, 256, 3)
        assert out.dtype == np.uint8

    def test_values_in_range(self):
        """Float values in [0, 1] should map to uint8 in [0, 255]."""
        img = torch.ones(1, 3, 84, 84)
        out = _tensor_to_octo_image(img, (256, 256))
        assert int(out.max()) == 255

        img_zero = torch.zeros(1, 3, 84, 84)
        out_zero = _tensor_to_octo_image(img_zero, (256, 256))
        assert int(out_zero.max()) == 0

    def test_no_resize_when_same_size(self):
        """When the input already matches target_size, no resize should be needed."""
        img = torch.rand(1, 3, 256, 256)
        out = _tensor_to_octo_image(img, (256, 256))
        assert out.shape == (1, 256, 256, 3)

    def test_batch_preserved(self):
        img = torch.rand(4, 3, 64, 64)
        out = _tensor_to_octo_image(img, (128, 128))
        assert out.shape[0] == 4


class TestGetBatchSize:
    def test_from_tensor(self):
        batch = {"a": torch.zeros(3, 10)}
        assert _get_batch_size(batch) == 3

    def test_empty_batch_returns_one(self):
        assert _get_batch_size({}) == 1

    def test_non_tensor_values_ignored(self):
        batch = {"text": "hello", "tensor": torch.zeros(5, 7)}
        assert _get_batch_size(batch) == 5


# ---------------------------------------------------------------------------
# OctoPolicy instantiation and state
# ---------------------------------------------------------------------------


class TestOctoPolicyInstantiation:
    def test_policy_name_and_config_class(self):
        assert OctoPolicy.name == "octo"
        assert OctoPolicy.config_class is OctoConfig

    def test_no_model_loaded_when_checkpoint_empty(self):
        cfg = _make_config(octo_checkpoint="")
        policy = OctoPolicy(cfg)
        assert policy._octo_model is None

    def test_reset_clears_queues(self):
        cfg = _make_config()
        policy = OctoPolicy(cfg)
        # Manually populate the queues
        policy._action_queue.extend([torch.zeros(1, ACTION_DIM)] * 3)
        policy._obs_history["image_primary"] = deque([np.zeros((1, 256, 256, 3))] * 2)
        policy.reset()
        assert len(policy._action_queue) == 0
        assert len(policy._obs_history) == 0

    def test_inference_without_model_raises(self):
        cfg = _make_config(octo_checkpoint="")
        policy = OctoPolicy(cfg)
        batch = _make_batch()
        with pytest.raises(RuntimeError, match="Octo model not loaded"):
            policy.predict_action_chunk(batch)

    def test_forward_raises_not_implemented(self):
        cfg = _make_config()
        policy = OctoPolicy(cfg)
        with pytest.raises(NotImplementedError):
            policy.forward(_make_batch())

    def test_get_optim_params_raises_not_implemented(self):
        cfg = _make_config()
        policy = OctoPolicy(cfg)
        with pytest.raises(NotImplementedError):
            policy.get_optim_params()


# ---------------------------------------------------------------------------
# Observation history / building
# ---------------------------------------------------------------------------


class TestOctoObservationBuilding:
    def test_single_frame_history(self):
        """First call: 1 frame available, window=2 → pad mask has one False."""
        cfg = _make_config(window_size=2, primary_image_key="observation.images.top")
        policy = OctoPolicy(cfg)
        batch = _make_batch()
        obs = policy._build_octo_observation(batch, "observation.images.top", None)

        assert "image_primary" in obs
        assert obs["image_primary"].shape == (BATCH_SIZE, 2, 256, 256, 3)
        # Only the newest slot should be valid
        assert obs["timestep_pad_mask"].shape == (BATCH_SIZE, 2)
        np.testing.assert_array_equal(obs["timestep_pad_mask"][0], [False, True])

    def test_full_history(self):
        """After window_size calls the pad mask should be all True."""
        cfg = _make_config(window_size=2, primary_image_key="observation.images.top")
        policy = OctoPolicy(cfg)
        batch = _make_batch()
        # First call fills one slot
        policy._build_octo_observation(batch, "observation.images.top", None)
        # Second call fills both slots
        obs2 = policy._build_octo_observation(batch, "observation.images.top", None)
        np.testing.assert_array_equal(obs2["timestep_pad_mask"][0], [True, True])

    def test_wrist_image_included_when_key_present(self):
        cfg = _make_config(
            window_size=1,
            primary_image_key="observation.images.top",
            wrist_image_key="observation.images.wrist",
        )
        policy = OctoPolicy(cfg)
        batch = {
            "observation.images.top": torch.rand(1, 3, 64, 64),
            "observation.images.wrist": torch.rand(1, 3, 64, 64),
        }
        obs = policy._build_octo_observation(
            batch, "observation.images.top", "observation.images.wrist"
        )
        assert "image_primary" in obs
        assert "image_wrist" in obs

    def test_wrist_image_absent_when_key_missing(self):
        cfg = _make_config(window_size=1, wrist_image_key="observation.images.wrist")
        policy = OctoPolicy(cfg)
        batch = {"observation.images.top": torch.rand(1, 3, 64, 64)}
        obs = policy._build_octo_observation(batch, "observation.images.top", "observation.images.wrist")
        # wrist key not in batch → no "image_wrist" in obs
        assert "image_wrist" not in obs

    def test_resolve_primary_key_from_config(self):
        cfg = _make_config(primary_image_key="observation.images.top")
        policy = OctoPolicy(cfg)
        batch = {"observation.images.top": torch.rand(1, 3, 64, 64)}
        assert policy._resolve_primary_image_key(batch) == "observation.images.top"

    def test_resolve_primary_key_auto_detect(self):
        cfg = _make_config(primary_image_key=None)
        policy = OctoPolicy(cfg)
        batch = {"observation.images.cam": torch.rand(1, 3, 64, 64)}
        assert policy._resolve_primary_image_key(batch) == "observation.images.cam"

    def test_resolve_primary_key_none_when_no_images(self):
        cfg = _make_config(primary_image_key=None)
        policy = OctoPolicy(cfg)
        batch = {OBS_STATE: torch.rand(1, 6)}
        assert policy._resolve_primary_image_key(batch) is None


# ---------------------------------------------------------------------------
# Task building
# ---------------------------------------------------------------------------


class TestOctoTaskBuilding:
    def _make_policy(self, task_text="pick up the spoon"):
        cfg = _make_config(task_text=task_text)
        policy = OctoPolicy(cfg)
        policy._octo_model = _make_mock_octo_model()
        return policy

    def test_task_text_from_config(self):
        policy = self._make_policy("do the thing")
        task = policy._build_octo_task(_make_batch())
        policy._octo_model.create_tasks.assert_called_once_with(texts=["do the thing"])

    def test_task_text_repeated_for_batch(self):
        policy = self._make_policy("grasp")
        task = policy._build_octo_task(_make_batch(batch_size=3))
        policy._octo_model.create_tasks.assert_called_once_with(texts=["grasp", "grasp", "grasp"])


# ---------------------------------------------------------------------------
# Predict action chunk (with mocked JAX)
# ---------------------------------------------------------------------------


def _make_mock_jax() -> MagicMock:
    """Build a minimal JAX mock that satisfies the imports in predict_action_chunk."""
    rng_val = np.array([0, 0], dtype=np.uint32)
    mock_jax = MagicMock()
    mock_jax.random.split.return_value = (rng_val, rng_val)
    return mock_jax


def _jax_sys_modules_patch(mock_jax=None):
    """Return a patch.dict context that injects a mock jax into sys.modules."""
    import sys

    if mock_jax is None:
        mock_jax = _make_mock_jax()
    # jax.random must also be patched as a sub-module
    return patch.dict(sys.modules, {"jax": mock_jax, "jax.random": mock_jax.random})


class TestPredictActionChunk:
    def _make_policy(self, unnorm_key=None, argmax=False) -> OctoPolicy:
        """Create policy with mock octo model (no real JAX needed)."""
        cfg = _make_config(unnorm_key=unnorm_key, argmax=argmax)
        policy = OctoPolicy(cfg)
        policy._octo_model = _make_mock_octo_model()
        policy._rng = np.array([0, 0], dtype=np.uint32)
        return policy

    def test_output_shape(self):
        policy = self._make_policy()
        batch = _make_batch()
        with _jax_sys_modules_patch():
            actions = policy.predict_action_chunk(batch)

        assert isinstance(actions, torch.Tensor)
        assert actions.shape == (BATCH_SIZE, ACTION_HORIZON, ACTION_DIM)
        assert actions.dtype == torch.float32

    def test_no_unnorm_when_key_none(self):
        policy = self._make_policy(unnorm_key=None)
        batch = _make_batch()
        with _jax_sys_modules_patch():
            policy.predict_action_chunk(batch)

        _, call_kwargs = policy._octo_model.sample_actions.call_args
        assert call_kwargs.get("unnormalization_statistics") is None

    def test_unnorm_stats_passed_when_key_set(self):
        policy = self._make_policy(unnorm_key="bridge_dataset")
        batch = _make_batch()
        with _jax_sys_modules_patch():
            policy.predict_action_chunk(batch)

        _, call_kwargs = policy._octo_model.sample_actions.call_args
        assert call_kwargs.get("unnormalization_statistics") is not None

    def test_bad_unnorm_key_raises(self):
        policy = self._make_policy(unnorm_key="nonexistent_dataset")
        batch = _make_batch()
        with _jax_sys_modules_patch():
            with pytest.raises(KeyError, match="nonexistent_dataset"):
                policy.predict_action_chunk(batch)


# ---------------------------------------------------------------------------
# Select action & action queue
# ---------------------------------------------------------------------------


class TestSelectAction:
    def _make_policy(self) -> OctoPolicy:
        cfg = _make_config(n_action_steps=2, action_horizon=ACTION_HORIZON)
        policy = OctoPolicy(cfg)
        policy._octo_model = _make_mock_octo_model()
        policy._rng = np.array([0, 0], dtype=np.uint32)
        return policy

    def test_select_action_shape(self):
        policy = self._make_policy()
        batch = _make_batch()
        with _jax_sys_modules_patch():
            action = policy.select_action(batch)

        assert action.shape == (BATCH_SIZE, ACTION_DIM)

    def test_action_queue_consumed(self):
        """Two consecutive calls should use the same chunk (n_action_steps=2)."""
        policy = self._make_policy()
        batch = _make_batch()
        with _jax_sys_modules_patch():
            _ = policy.select_action(batch)
            call_count_after_first = policy._octo_model.sample_actions.call_count
            _ = policy.select_action(batch)
            call_count_after_second = policy._octo_model.sample_actions.call_count

        # The model should have been called exactly once (chunk refilled only on first call)
        assert call_count_after_first == 1
        assert call_count_after_second == 1  # second action came from the same chunk

    def test_queue_refill_on_empty(self):
        """After the queue is drained, the next call should re-query the model."""
        policy = self._make_policy()
        batch = _make_batch()
        with _jax_sys_modules_patch():
            # Drain queue completely (n_action_steps=2)
            policy.select_action(batch)
            policy.select_action(batch)
            assert policy._octo_model.sample_actions.call_count == 1

            # Third call should trigger a new model query
            policy.select_action(batch)
            assert policy._octo_model.sample_actions.call_count == 2


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_pretrained_writes_config(self, tmp_path):
        cfg = _make_config()
        policy = OctoPolicy(cfg)
        policy.save_pretrained(str(tmp_path))
        config_file = tmp_path / "config.json"
        assert config_file.exists(), "config.json should be written"

    def test_save_pretrained_no_safetensors(self, tmp_path):
        cfg = _make_config()
        policy = OctoPolicy(cfg)
        policy.save_pretrained(str(tmp_path))
        # Octo weights are not stored as safetensors
        safetensors_file = tmp_path / "model.safetensors"
        assert not safetensors_file.exists(), "safetensors should NOT be written for OctoPolicy"

    def test_from_pretrained_loads_config(self, tmp_path):
        cfg = _make_config()
        policy = OctoPolicy(cfg)
        policy.save_pretrained(str(tmp_path))

        loaded = OctoPolicy.from_pretrained(str(tmp_path), config=cfg)
        assert loaded.config.type == "octo"
        assert loaded.config.task_text == cfg.task_text
        assert loaded._octo_model is None  # no checkpoint → model not loaded
