## DQN Plays Pong

Lightweight PyTorch implementation of a DQN agent that learns to play Atari Pong via Gymnasium and ALE. Includes reward shaping, frame preprocessing, TensorBoard logging, and scripts to train or replay a saved model.

### Features
- Dueling DQN with residual blocks for stable value estimation.
- Atari preprocessing: grayscale, resize to 84x84, and 4-frame stacking.
- Experience replay with epsilon-greedy exploration and target network updates.
- Reward shaping tuned for Pong and TensorBoard metrics out of the box.
- Simple train/eval scripts plus `Makefile` shortcut.

### Project Structure
- `core/` — agent, model, layers, replay buffer, and env wrappers.
- `scripts/train.py` — full training loop, logging, reward shaping, model save.
- `scripts/play.py` — load a saved model and watch it play Pong.
- `runs/` — TensorBoard logs (created at runtime).
- `models/` — saved weights (`dqn_pong_model.pth`, created after training).

### Quickstart
**Prerequisites**
- Python 3.10+
- Atari ROM license acceptance (handled by `gymnasium[accept-rom-license,atari]`)
- Optional: GPU with CUDA for faster training

**Setup with uv (recommended)**
```bash
uv sync           # installs deps into .venv
uv run python scripts/train.py
```

**Setup with pip (alternative)**
```bash
python -m venv .venv && source .venv/bin/activate
pip install ale-py gymnasium[accept-rom-license,atari] numpy opencv-python tensorboard torch
python scripts/train.py
```

### Usage
- Train: `make train` or `uv run python scripts/train.py`
- Watch play (needs a saved model at `models/dqn_pong_model.pth`):  
  `uv run python scripts/play.py`
- View logs: `tensorboard --logdir runs/`

### Configuration
Key hyperparameters live in `core/config.py` (buffer size, epsilon schedule, target update freq, total frames, etc.). Adjust `MODEL_SAVE_PATH` if you want to store checkpoints elsewhere.

### Notes
- Training is compute-intensive; expect long runs without a GPU.
- The environment renders by default for visibility; disable or switch render modes if running headless.

### License
See `LICENSE` (MIT).

