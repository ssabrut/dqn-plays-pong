### DQN Plays Pong — Project Overview

This repository implements a Deep Q-Network (DQN) agent to play Atari Pong using `gymnasium` (ALE) and PyTorch. It includes common Atari preprocessing wrappers, a replay buffer, a convolutional Q-network, and a training loop with TensorBoard logging.

### Repository Structure
- `scripts/train.py`: Training entrypoint; sets up env, logs to TensorBoard, performs reward shaping, optimizes DQN, and periodically updates target network.
- `scripts/play.py`: Loads a saved model and plays a few episodes greedily to demonstrate performance.
- `core/agent.py`: DQN agent logic: action selection with epsilon-greedy, learning step, epsilon decay, and target network sync.
- `core/model.py`: CNN Q-network mapping stacked frames to Q-values for actions.
- `core/replay_buffer.py`: Experience replay buffer with `(state, action, reward, next_state, done)` tuples.
- `core/wrappers.py`: Atari preprocessing (grayscale, resize to 84×84, and frame stacking of k=4).
- `core/config.py`: Environment name, device, hyperparameters, and training settings.

### Training Highlights
- **Environment**: `ALE/Pong-v5` registered via `ale_py`. Observations are grayscale, resized 84×84, with 4-frame stacks.
- **Logging**: TensorBoard `SummaryWriter` logs rewards, shaped rewards, epsilon, and moving averages in `runs/pong_experiment_<timestamp>`.
- **Target Network**: Synced every `TARGET_UPDATE_FREQ` frames.
- **Optimization**: Standard DQN with replay, discount `GAMMA`, learning rate `LEARNING_RATE`, batch size `BATCH_SIZE`.

### Reward Shaping Strategy
The raw Pong reward (+1/-1 on points) is sparse for learning. The training script applies shaped rewards:
- Score a point: `+5.0`
- Lose a point: `-5.0`
- No-op action: `-0.02` (discourage inactivity / local optima)
- Other steps: `+0.01` (small living reward)

This shaping addresses early stagnation and the “do-nothing at edges” local optimum observed in experiments. Adjust shaping cautiously to avoid diverging from the true objective.

### Running Training
Install dependencies (Python >= 3.10), then run:

```bash
python -m scripts.train
```

TensorBoard:

```bash
tensorboard --logdir runs
```

### Playing With a Trained Agent
Ensure a model is saved at the configured path (`core/config.py` → `MODEL_SAVE_PATH`), then:

```bash
python -m scripts.play
```

### Key Hyperparameters (see `core/config.py`)
- `GAMMA = 0.995`
- `LEARNING_RATE = 1e-4`
- `BATCH_SIZE = 32`
- `BUFFER_SIZE = 100_000`
- `EPSILON_START = 1.0`, `EPSILON_END = 0.02`, `EPSILON_DECAY = 100_000`
- `TARGET_UPDATE_FREQ = 10_000`
- `TOTAL_FRAMES = 1_000_000`

### Notes and Tips
- Training Pong can take hours on CPU and significantly less on GPU; the code selects `cuda` if available.
- ROM license acceptance is handled via `gymnasium[accept-rom-license, atari]`.
- If the agent over-moves after learning (missing easy balls), consider tuning `TARGET_UPDATE_FREQ`, `LEARNING_RATE`, and `GAMMA` per your experiments.


