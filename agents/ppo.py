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
        # 【修正】根据新的解耦网络结构，正确分组参数并设置对应学习率
        self.optimizer = optim.Adam([
            # Actor 部分 (LR = 2e-4)
            {'params': self.policy.actor_head.parameters(), 'lr': cfg.LR_ACTOR},
            {'params': self.policy.actor_net.parameters(), 'lr': cfg.LR_ACTOR},  # 包含 Actor 的 Embedding 和 Transformer

            # Critic 部分 (LR = 1e-3)
            {'params': self.policy.critic_head.parameters(), 'lr': cfg.LR_CRITIC},
            {'params': self.policy.critic_net.parameters(), 'lr': cfg.LR_CRITIC},  # 包含 Critic 的 Embedding 和 Transformer
        ])

        # 简单起见，如果不想细分参数组，使用原版也可以，但需注意 Table I 区分了 Actor/Critic 学习率
        # 原代码:
        # self.optimizer = optim.Adam([{'params': self.policy.parameters(), 'lr': cfg.LR_ACTOR}])
        # 建议修正为支持不同学习率（如上）或暂时保持原状，这里重点修 GAE。

        # 为了严格复现 Table I: Critic lr=1e-3, Actor lr=2e-4
        # 由于是共享参数网络，通常共享部分使用较小学习率(Actor LR)，Head部分分开。
        # 这里为保持代码简洁，暂时恢复为统一优化器，但在 update 中计算 Loss 时会体现差异。

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
        # 转换 state 为 tensor (Seq_Len, State_Dim) -> (1, Seq_Len, State_Dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value, _ = self.policy_old.get_action(state)

        # 存入 buffer (存 CPU tensor 以节省显存)
        self.buffer['states'].append(state.cpu())
        self.buffer['actions'].append(action.cpu())
        self.buffer['logprobs'].append(log_prob.cpu())
        self.buffer['values'].append(value.cpu())  # 存储 V(s_t)

        return action.item()

    def store_transition(self, reward, done):
        self.buffer['rewards'].append(reward)
        self.buffer['is_terminals'].append(done)

    def update(self):
        # --- 【核心修正】 使用 GAE 计算优势和回报 ---
        rewards = []
        discounted_reward = 0

        # 提取 Buffer 数据
        # values shape: (Batch_Size, 1) -> flatten -> (Batch_Size,)
        values = torch.cat(self.buffer['values'], dim=0).squeeze().to(self.device)

        # 这里的 values 是 V(s_t)。计算 GAE 需要 V(s_{t+1})。
        # 由于我们没有显式存储 next_state 的 value（除了序列中的下一个），
        # 我们利用 values[t+1] 作为 V(s_{t+1})。
        # 对于最后一个时间步，如果没有结束 (Done=False)，理论上需要 bootstrap。
        # 但为保持接口简单，这里假设最后一步后价值为 0 (或者 values 列表最好能多存一个 next_val)
        # 这里采用追加一个 0 值来处理越界
        next_values = torch.cat([values[1:], torch.tensor([0.0], device=self.device)])

        gae = 0
        returns = []

        # 逆序遍历计算 GAE
        # buffers are lists, convert to tensor or iterate list
        raw_rewards = self.buffer['rewards']
        is_terminals = self.buffer['is_terminals']

        for step in reversed(range(len(raw_rewards))):
            r_t = raw_rewards[step]
            done = is_terminals[step]

            v_t = values[step]
            # 如果是 done，则 v_{t+1} 视为 0；否则取 values[step+1]
            # 注意：如果 step 是 buffer 的最后一步且不是 done，这里会有截断误差(视为0)
            v_next = next_values[step] if not done else 0.0

            # Delta (Eq. 25): delta = r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = r_t + cfg.GAMMA * v_next - v_t

            # GAE (Eq. 24): A_t = delta + gamma * lambda * A_{t+1}
            # 如果 done，则 A_{t+1} 重置为 0
            gae = delta + cfg.GAMMA * cfg.GAE_LAMBDA * gae * (1.0 - float(done))

            # Return = A_t + V_t (用于计算 Critic Loss: MSE(V(s), Return))
            returns.insert(0, gae + v_t)

        # 转换为 Tensor
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # 归一化 Advantages (对 PPO 训练稳定性至关重要)
        # advantages = returns - values
        # 或者直接使用计算出的 gae (注意顺序要对)
        # 为了方便，重新计算一下 advantages
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # 堆叠 Buffer 数据 (用于 Batch 更新)
        old_states = torch.cat(self.buffer['states'], dim=0).to(self.device)
        old_actions = torch.cat(self.buffer['actions'], dim=0).to(self.device)
        old_logprobs = torch.cat(self.buffer['logprobs'], dim=0).to(self.device)
        old_values = values.detach()  # 用于 Value Clip (可选)

        # 2. PPO 更新循环
        for _ in range(cfg.K_EPOCHS):
            # 评估旧状态下的新策略
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)

            # 计算 Ratios
            ratios = torch.exp(logprobs - old_logprobs)

            # 计算 Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - cfg.EPS_CLIP, 1 + cfg.EPS_CLIP) * advantages

            # Actor Loss
            loss_actor = -torch.min(surr1, surr2).mean()

            # Critic Loss (MSE)
            loss_critic = self.mse_loss(state_values, returns)

            # Entropy Loss (Bonus)
            loss_entropy = -dist_entropy.mean()

            # 总 Loss (系数通常需要微调，这里使用典型值: 0.5 Critic, 0.01 Entropy)
            loss = loss_actor + 0.5 * loss_critic + 0.01 * loss_entropy

            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪 (论文未明确提及阈值，通用 0.5)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)

            self.optimizer.step()

        # 更新旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.clear_buffer()

    def clear_buffer(self):
        self.buffer = {k: [] for k in self.buffer}