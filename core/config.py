import torch

# --- Environment ---
ENV_NAME: str = "ALE/Pong-v5"
MODEL_SAVE_PATH: str = "models/dqn_pong_model.pth"
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Hyperparameters ---
BATCH_SIZE: int = 32
GAMMA: float = 995e-3
EPSILON_START: float = 1.0
EPSILON_END: float = 2e-2
EPSILON_DECAY: int = 100_000
TARGET_UPDATE_FREQ: int = 10_000
LEARNING_RATE: float = 1e-4
BUFFER_SIZE: int = 100_000

# --- Training settings ---
TOTAL_FRAMES: int = 1000000
LOG_FREQ: int = 20  # log every 20 episodes
