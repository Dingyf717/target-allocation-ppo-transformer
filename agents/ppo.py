# agents/ppo.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from configs.config import cfg
from networks.transformer_net import TransformerActorCritic


class PPOAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化网络
        self.policy = TransformerActorCritic().to(self.device)
        self.optimizer = optim.Adam([
            {'params': self.policy.parameters(), 'lr': cfg.LR_ACTOR}
        ])

        # 旧策略网络
        self.policy_old = TransformerActorCritic().to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.buffer = {
            'states': [],
            'actions': [],
            'logprobs': [],
            'rewards': [],
            'is_terminals': [],
            'values': []
        }

        self.mse_loss = nn.MSELoss()

    def select_action(self, state):
        # 转换 state 为 tensor (5, 14) -> (1, 5, 14)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value, _ = self.policy_old.get_action(state)

        # 存入 buffer (存 CPU tensor 以节省显存)
        self.buffer['states'].append(state.cpu())
        self.buffer['actions'].append(action.cpu())
        self.buffer['logprobs'].append(log_prob.cpu())
        self.buffer['values'].append(value.cpu())

        return action.item()

    def store_transition(self, reward, done):
        self.buffer['rewards'].append(reward)
        self.buffer['is_terminals'].append(done)

    def update(self):
        # 1. 计算 Monte Carlo Returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer['rewards']), reversed(self.buffer['is_terminals'])):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (cfg.GAMMA * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # 堆叠 Buffer 数据
        old_states = torch.cat(self.buffer['states'], dim=0).to(self.device)
        old_actions = torch.cat(self.buffer['actions'], dim=0).to(self.device)
        old_logprobs = torch.cat(self.buffer['logprobs'], dim=0).to(self.device)

        # 2. PPO 更新
        for _ in range(cfg.K_EPOCHS):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - old_logprobs)
            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - cfg.EPS_CLIP, 1 + cfg.EPS_CLIP) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()

            # 【关键】梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)

            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.clear_buffer()

    def clear_buffer(self):
        self.buffer = {k: [] for k in self.buffer}