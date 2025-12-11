## System Architecture

This project trains and evaluates a DQN agent to play Atari Pong. The system is organized into modular components that handle environment preprocessing, experience storage, model inference, training, and evaluation.

### High-Level Flow
1. **Environment setup** (`scripts/train.py`, `scripts/play.py`): registers ALE environments, applies wrappers, and creates the Gymnasium env with human render.
2. **Preprocessing pipeline** (`core/wrappers.py`): grayscale -> resize to 84x84 -> stack 4 frames, exposing observations shaped `(4, 84, 84)`.
3. **Agent loop** (`core/agent.py`): epsilon-greedy action selection, reward shaping (in `train.py`), replay storage, learning, and target network updates.
4. **Model** (`core/model.py`, `core/layers.py`): dueling DQN with residual blocks for stable value estimation.
5. **Persistence & logging**: TensorBoard logging to `runs/`, model checkpoint to `models/dqn_pong_model.pth`.

### Components
- **Environment wrappers (`core/wrappers.py`)**
  - `GrayScaleObservation`: converts RGB frames to single-channel grayscale.
  - `ResizeObservation`: resizes frames to `(84, 84)`.
  - `FrameStack`: stacks the last `k=4` frames to encode temporal information.
- **Replay Buffer (`core/replay_buffer.py`)**
  - Fixed-capacity deque storing `(state, action, reward, next_state, done)` tuples.
  - Random sampling breaks correlation between consecutive steps for stable training.
- **DQN Model (`core/model.py`, `core/layers.py`)**
  - Conv stack: `Conv(32, 8x8, s4)` → `Conv(64, 4x4, s2)` → `Conv(128, 3x3, s1)` with batch norm and ReLU.
  - Four `ResidualBlock`s maintain gradient flow and representation depth.
  - Dueling heads: separate value and advantage streams combined into Q-values.
  - Input normalized from `[0, 255]` to `[0, 1]` during forward pass.
- **Agent (`core/agent.py`)**
  - Epsilon-greedy policy with exponential decay (`EPSILON_START` → `EPSILON_END` over `EPSILON_DECAY` frames).
  - Stores transitions in replay buffer; learns once buffer has `BATCH_SIZE` samples.
  - Target network synced every `TARGET_UPDATE_FREQ` frames.
  - Loss: Huber (smooth L1) between current and target Q-values, with gradient clipping.
- **Training Script (`scripts/train.py`)**
  - Applies reward shaping: strong rewards/penalties for scoring, small incentives for movement, small penalty for NOOP.
  - Logs episode rewards, shaped rewards, epsilon, and loss to TensorBoard.
  - Stops after `TOTAL_FRAMES` and saves the policy network weights to `MODEL_SAVE_PATH`.
- **Playback Script (`scripts/play.py`)**
  - Loads saved weights, runs greedy policy (no exploration), and renders gameplay for a few episodes.

### Configuration
`core/config.py` centralizes parameters:
- Env/device: `ENV_NAME`, `DEVICE`, `MODEL_SAVE_PATH`
- Hyperparameters: `BATCH_SIZE`, `GAMMA`, `EPSILON_*`, `TARGET_UPDATE_FREQ`, `LEARNING_RATE`, `BUFFER_SIZE`
- Training schedule: `TOTAL_FRAMES`, `LOG_FREQ`

### Data & Control Flow (Training)
1. Reset env → receive stacked state `(4, 84, 84)`.
2. Agent selects action via epsilon-greedy on policy net.
3. Env step → raw reward → reward shaping → store transition in replay buffer.
4. Once buffer is warm: sample batch → compute target Q with target net → update policy net via Huber loss → clip grads.
5. Every `TARGET_UPDATE_FREQ` frames: copy policy weights to target net.
6. Log metrics to TensorBoard; repeat until frame budget met → save checkpoint.

### Operational Notes
- Rendering is set to `human`; switch to headless modes for remote training if needed.
- Checkpoints land in `models/`; TensorBoard logs in `runs/`.
- The system assumes a GPU if available (`torch.cuda.is_available()`), otherwise falls back to CPU.

