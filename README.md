# DQN Plays Pong 🏓

A highly optimized Deep Q-Network (DQN) implementation capable of achieving a perfect score in Atari Pong. 

This project demonstrates a production-grade Reinforcement Learning pipeline using **PyTorch** and **Gymnasium**, featuring custom wrappers, Dueling DQN architecture, and critical stability optimizations that solve the "stuck at -21" convergence problem common in naive implementations.

---

## 🚀 Performance & Results

### The "Stuck at -21" Challenge
Initial training runs were failing to learn, with the agent stuck at the minimum possible score of **-21** (losing every round). The agent exhibited "jittery" behavior, failing to track ball velocity effectively.

### The Solution
Through performance profiling and debugging, two critical issues were identified and fixed:
1.  **Temporal Resolution (Frame Skipping):** The raw 60Hz feed provided insufficient motion delta between frames. I implemented a `MaxAndSkip` wrapper to repeat actions for 4 frames, effectively quadrupling the agent's temporal field of view and allowing the Convolutional layers to detect velocity.
2.  **Reward Manifold:** Naive "incentive" rewards (giving small points for paddle movement) caused a reward loop where the agent maximized points by vibrating in place rather than hitting the ball. This was replaced with **Reward Clipping** (`[-1, 1]`), forcing the agent to prioritize game outcomes.

### Final Result
With these optimizations, the agent converges to a mean reward of **+20.0 to +21.0** (Perfect Play) within ~800 episodes.

| Metric          | Baseline (Naive) | Optimized (Current) |
| :-------------- | :--------------- | :------------------ |
| **Frame Rate**  | 60Hz (Raw)       | 15Hz (Skipped)      |
| **Observation** | Static Noise     | Clear Velocity      |
| **Mean Reward** | -21.0 (Random)   | **+20.5 (Solved)**  |

---

## 🧠 Key Features & Architecture

### 1. Model Architecture (Dueling DQN)
Instead of a simple feed-forward network, this project uses a **Dueling DQN** architecture. This splits the network into two streams after the convolutional layers:
* **Value Stream $V(s)$:** Estimates the value of the current state.
* **Advantage Stream $A(s, a)$:** Estimates the benefit of taking a specific action.

This stabilizes training by allowing the agent to learn which states are valuable without having to learn the effect of every action for every state.

### 2. Preprocessing Pipeline
Raw Atari frames (210x160 RGB) are too complex for efficient training. The custom pipeline includes:
* **`MaxAndSkipEnv`**: Repeats actions for $k=4$ frames and takes the max pixel value (de-flickering).
* **`GrayScaleObservation`**: Reduces dimensionality (RGB $\to$ Grayscale).
* **`ResizeObservation`**: Downsamples to an efficient $84 \times 84$ matrix.
* **`FrameStack`**: Stacks the last 4 frames into a $(4, 84, 84)$ tensor to capture **motion and velocity**.

### 3. Training Stability
* **Experience Replay Buffer**: Breaks correlation between consecutive samples to prevent overfitting.
* **Target Network**: A frozen copy of the network is used for Q-value targets, updated every 10k steps to prevent oscillation.
* **Huber Loss**: Used instead of MSE to be robust against outliers (exploding gradients).

---

## 🛠 Project Structure

```bash
dqn-plays-pong/
├── core/
│   ├── agent.py          # Epsilon-greedy logic & learning step
│   ├── model.py          # Dueling DQN with Residual Blocks
│   ├── wrappers.py       # Custom Gym wrappers (FrameSkip, etc.)
│   ├── replay_buffer.py  # Circular buffer for experience replay
│   └── config.py         # Hyperparameters (Gamma, LR, Batch Size)
├── scripts/
│   ├── train.py          # Main training loop with TensorBoard logging
│   └── play.py           # Inference script to watch the agent play
├── models/               # Saved checkpoints (.pth)
├── runs/                 # TensorBoard logs
└── README.md
```

## 💻 Installation & Usage

**Prerequisites**
- Python 3.10+
- Make (optional)

**Method 1: Using `uv` (Recommended)**
This project uses `uv` for ultra-fast dependency management.
```bash
# 1. Sync dependencies
uv sync

# 2. Train the agent
uv run python scripts/train.py

# 3. Watch the trained agent play
uv run python scripts/play.py
```

**Method 2: Standard Pip**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # or install manually per pyproject.toml

python scripts/train.py
```

**Monitoring Training**
Launch TensorBoard to visualize the Loss and Average Reward in real-time:
```bash
tensorboard --logdir runs/
```

## ⚙️ Configuration
Hyperparameters are centralized in `core/config.py`. Key defaults:
```python
BATCH_SIZE = 32          # Samples per training step
GAMMA = 0.99             # Discount factor for future rewards
EPSILON_DECAY = 100_000  # Frames to decay exploration
TARGET_UPDATE = 10_000   # Frequency of target net updates
```