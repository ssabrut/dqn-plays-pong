import random
import torch
import numpy as np
from torch import optim
from torch.nn import functional as F

from core.model import DQN
from core.replay_buffer import ReplayBuffer
from core.config import DEVICE

class DQNAgent:
    def __init__(self, env, config) -> None:
        self.env = env
        self.config = config
        self.action_space_size = env.action_space.n
        self.obs_space_shape = env.observation_space.shape
        self.buffer = ReplayBuffer(config.BUFFER_SIZE)
        self.epsilon = config.EPSILON_START

        self.policy_net = DQN(self.obs_space_shape, self.action_space_size).to(DEVICE)
        self.target_net = DQN(self.obs_space_shape, self.action_space_size).to(DEVICE)
        self.update_target_network()
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE, amsgrad=True)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def decay_epsilon(self, frame_idx):
        c = self.config
        self.epsilon = c.EPSILON_END + (c.EPSILON_START - c.EPSILON_END) * np.exp(-1. * frame_idx / c.EPSILON_DECAY)

    def learn(self):
        if len(self.buffer) < self.config.BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.config.BATCH_SIZE)
        states_t = torch.from_numpy(states).to(DEVICE)
        actions_t = torch.from_numpy(actions).long().unsqueeze(-1).to(DEVICE)
        rewards_t = torch.from_numpy(rewards).to(DEVICE)
        next_states_t = torch.from_numpy(next_states).to(DEVICE)
        dones_t = torch.from_numpy(dones).to(DEVICE)

        current_q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(-1)
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(1)[0]
            next_q_values[dones_t] = 0.0
            target_q_values = rewards_t + self.config.GAMMA * next_q_values

        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()