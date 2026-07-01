# Octo Policy

[Octo](https://octo-models.github.io/) is a generalist robotic policy from UC Berkeley trained on 800k robot trajectories from the Open X-Embodiment dataset. This LeRobot integration wraps the JAX-based Octo model in a `PreTrainedPolicy`-compatible adapter so you can load and run it through the standard LeRobot APIs.

> **Training / fine-tuning is not supported through this adapter.** Use the official Octo training scripts at [octo-models/octo](https://github.com/octo-models/octo) to fine-tune, then load the resulting checkpoint here for inference.

---

## Installation

Octo and JAX are **not** included in LeRobot's default dependencies. Install them first:

```bash
# 1. Clone and install Octo from source
git clone https://github.com/octo-models/octo
cd octo
pip install -e .
pip install -r requirements.txt

# 2. Install JAX (pick ONE of the following)
# CPU only:
pip install "jax[cpu]"
# NVIDIA GPU (CUDA 12):
pip install --upgrade "jax[cuda12_pip]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Then install the LeRobot `octo` optional extra (pulls in the correct version pins):

```bash
pip install "lerobot[octo]"
# or from source:
pip install -e ".[octo]"
```

---

## Available Checkpoints

Both checkpoints are hosted on the Hugging Face Hub and are automatically downloaded on first use:

| Model       | HF path                                   | Parameters | Speed (RTX 4090) |
|-------------|-------------------------------------------|-----------|------------------|
| Octo-Small  | `hf://rail-berkeley/octo-small-1.5`      | 27 M      | 17 it/s          |
| Octo-Base   | `hf://rail-berkeley/octo-base-1.5`       | 93 M      | 13 it/s          |

---

## Quick Start

### Direct instantiation

```python
from lerobot.policies.octo import OctoConfig, OctoPolicy

config = OctoConfig(
    octo_checkpoint="hf://rail-berkeley/octo-small-1.5",
    primary_image_key="observation.images.top",   # key in your observation dict
    task_text="pick up the spoon",
    unnorm_key="bridge_dataset",                   # dataset statistics for unnormalization
    n_action_steps=4,                             # actions to execute per chunk
    window_size=2,                                # observation history length
)

policy = OctoPolicy(config)
policy.eval()

# obs_batch: dict with PyTorch tensors, e.g.
# {"observation.images.top": torch.Tensor (B, C, H, W)}
action = policy.select_action(obs_batch)   # returns (B, action_dim) tensor
```

### Via the LeRobot factory

```python
from lerobot.policies.factory import make_policy, make_policy_config
from lerobot.datasets import LeRobotDataset

dataset = LeRobotDataset("lerobot/aloha_sim_insertion_human")

config = make_policy_config(
    "octo",
    octo_checkpoint="hf://rail-berkeley/octo-small-1.5",
    primary_image_key="observation.images.top",
    task_text="pick up the spoon",
    unnorm_key="bridge_dataset",
    n_action_steps=4,
)

policy = make_policy(config, ds_meta=dataset.meta)
```

---

## Configuration Reference

All options are fields of `OctoConfig`. Commonly used ones:

| Field | Default | Description |
|-------|---------|-------------|
| `octo_checkpoint` | `"hf://rail-berkeley/octo-small-1.5"` | HuggingFace Hub path or local directory |
| `window_size` | `1` | Observation history frames passed to the model (Octo was pretrained with `2`) |
| `n_action_steps` | `1` | Actions to execute before re-querying the model |
| `action_horizon` | `4` | Total action chunk size predicted per call |
| `task_text` | `""` | Default language instruction |
| `unnorm_key` | `None` | Key in `model.dataset_statistics` to unnormalize actions (e.g. `"bridge_dataset"`) |
| `primary_image_key` | `None` | Observation dict key for the primary camera (auto-detected if `None`) |
| `wrist_image_key` | `None` | Observation dict key for the optional wrist camera |
| `image_size` | `(256, 256)` | Images are resized to this (H, W) before being passed to Octo |
| `argmax` | `False` | Use deterministic argmax instead of diffusion sampling |
| `seed` | `42` | Random seed for JAX PRNG |

### Finding valid `unnorm_key` values

```python
from octo.model.octo_model import OctoModel
model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
print(list(model.dataset_statistics.keys()))
# e.g. ['bridge_dataset', 'fractal20220817_data', ...]
```

If `unnorm_key=None`, the model returns z-score normalized actions (mean 0, std 1) which may not correspond to physical robot units.

---

## Observation Format

LeRobot's eval loop passes observations as a Python dict of PyTorch tensors. The adapter expects image tensors of shape `(B, C, H, W)` (float, values in [0, 1] or uint8 in [0, 255]).

Internally the adapter:
1. Converts each image tensor to `(B, H, W, C)` uint8 NumPy arrays
2. Resizes to `config.image_size` (default 256 × 256)
3. Maintains a rolling history buffer of length `window_size`
4. Constructs the Octo observation dict `{"image_primary": ..., "timestep_pad_mask": ...}`

When using a **wrist camera**, set `wrist_image_key` to the corresponding observation key:

```python
config = OctoConfig(
    primary_image_key="observation.images.top",
    wrist_image_key="observation.images.wrist",
    ...
)
```

---

## Action Format

`select_action` returns a PyTorch float tensor of shape `(B, action_dim)`. The `action_dim` depends on the robot embodiment / dataset. For BridgeData v2 (WidowX) it is **7** `[x, y, z, yaw, pitch, roll, gripper]`.

Action chunking: the model predicts `action_horizon` steps per call (default 4). The first `n_action_steps` are queued and returned one at a time, minimising re-inference overhead.

---

## Saving and Loading

```python
# Save the LeRobot config (Octo JAX weights are referenced by path, not copied)
policy.save_pretrained("/path/to/my_octo_config")

# Load back
from lerobot.policies.octo import OctoPolicy
policy = OctoPolicy.from_pretrained("/path/to/my_octo_config")
```

> The Octo JAX checkpoint itself is **not** stored inside the LeRobot save directory — only `config.json` is written. The checkpoint path is recorded in `config.octo_checkpoint` and re-loaded on instantiation.

---

## Limitations

- **Inference only.** Training and fine-tuning are delegated to the Octo-native scripts.
- **No safetensors weights.** The model weights live in a JAX/Orbax checkpoint, not a `model.safetensors` file.
- **JAX is required** at inference time. The policy raises `ImportError` with clear instructions if JAX/Octo are not installed.
- The default pre-trained checkpoints (`octo-small-1.5`, `octo-base-1.5`) are best matched with datasets in the Open X-Embodiment mixture. For other robots, fine-tune the checkpoint first.
