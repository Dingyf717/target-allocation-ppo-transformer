# networks/transformer_net.py
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from configs.config import cfg

def init_layer(layer, std=np.sqrt(2), bias_const=0.0):
    """ PPO 专用的正交初始化 """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class TransformerActorCritic(nn.Module):
    def __init__(self):
        super(TransformerActorCritic, self).__init__()

        # 1. 特征嵌入层 (Feature Embedding)
        # 输入: (Batch, Seq_Len, 14) -> 输出: (Batch, Seq_Len, 128)
        self.embedding = nn.Sequential(
            init_layer(nn.Linear(cfg.STATE_DIM, cfg.EMBED_DIM)),
            nn.ReLU() # 【修正】Tanh -> ReLU (对应论文 Fig. 2)
        )

        # 2. Transformer Encoder (核心特征提取)
        # batch_first=True 意味着输入/输出格式为 (Batch, Seq_Len, Embed_Dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.EMBED_DIM,
            nhead=cfg.NUM_HEADS,
            dim_feedforward=256,
            dropout=0.0,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.NUM_LAYERS
        )

        # 计算展平后的特征维度
        # 维度 = SEQ_LEN * EMBED_DIM
        self.flat_dim = cfg.SEQ_LEN * cfg.EMBED_DIM

        # 3. Actor Head (策略头)
        self.actor_head = nn.Sequential(
            init_layer(nn.Linear(self.flat_dim, 64)),
            nn.ReLU(), # 【修正】Tanh -> ReLU
            init_layer(nn.Linear(64, cfg.ACTION_DIM), std=0.01)
        )

        # 4. Critic Head (价值头)
        self.critic_head = nn.Sequential(
            init_layer(nn.Linear(self.flat_dim, 64)),
            nn.ReLU(), # 【修正】Tanh -> ReLU
            init_layer(nn.Linear(64, 1), std=1.0)
        )

    def forward(self, state):
        # state shape: (Batch, Seq_Len, State_Dim)

        # 容错处理：如果输入没有 Batch 维度，则增加 Batch 维度
        if state.dim() == 2:
            state = state.unsqueeze(0)

        # 1. Embedding
        x = self.embedding(state)

        # 2. Transformer
        x = self.transformer(x)

        # 3. Flatten (展平)
        # 将 (Batch, Seq, Feat) 展平为 (Batch, Seq * Feat)
        x_flat = x.reshape(x.size(0), -1)

        action_logits = self.actor_head(x_flat)
        state_value = self.critic_head(x_flat)

        return action_logits, state_value

    def get_action(self, state):
        """ 采样动作 (用于 rollout) """
        logits, value = self(state)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, value, dist.entropy()

    def evaluate(self, state, action):
        """ 评估动作 (用于 update) """
        logits, value = self(state)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, value, dist_entropy