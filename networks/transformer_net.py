# networks/transformer_net.py
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from configs.config import cfg


def init_layer(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class TransformerBlock(nn.Module):
    """
    Transformer 编码模块
    结构: Linear Embedding -> Transformer Encoder
    """

    def __init__(self):
        super(TransformerBlock, self).__init__()
        # 1. Embedding 层: 将 State (14维) 映射到 Embed Dim (128维)
        self.embedding = nn.Sequential(
            init_layer(nn.Linear(cfg.STATE_DIM, cfg.EMBED_DIM)),
            nn.ReLU()
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, cfg.SEQ_LEN, cfg.EMBED_DIM))

        # 2. Transformer Encoder
        # Table I: Heads=8, Channels=128
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.EMBED_DIM,
            nhead=cfg.NUM_HEADS,
            dim_feedforward=256,  # 内部 FFN 维度，通常 2-4 倍 Embed Dim
            dropout=0.0,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.NUM_LAYERS)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, State_Dim)
        x = self.embedding(x)

        # [新增] 加上位置编码
        # 广播机制会自动处理 Batch 维度
        x = x + self.pos_embedding[:, :x.size(1), :]

        # x shape: (Batch, Seq_Len, Embed_Dim)
        x = self.transformer(x)
        return x


class TransformerActorCritic(nn.Module):
    def __init__(self):
        super(TransformerActorCritic, self).__init__()

        self.flat_dim = cfg.SEQ_LEN * cfg.EMBED_DIM

        # --- Actor Network ---
        self.actor_net = TransformerBlock()
        # Head 结构依据 Fig 2: FC -> ReLU -> FC -> Softmax (在 get_action 中做)
        self.actor_head = nn.Sequential(
            init_layer(nn.Linear(self.flat_dim, 64)),
            nn.ReLU(),
            init_layer(nn.Linear(64, cfg.ACTION_DIM), std=0.01)
        )

        # --- Critic Network ---
        self.critic_net = TransformerBlock()
        # Head 结构依据 Fig 2: FC -> ReLU -> FC -> Output
        self.critic_head = nn.Sequential(
            init_layer(nn.Linear(self.flat_dim, 64)),
            nn.ReLU(),
            init_layer(nn.Linear(64, 1), std=1.0)
        )

    def forward(self, state):
        raise NotImplementedError("Please use get_action or evaluate.")

    def get_action(self, state):
        """用于采样动作 (Inference)"""
        if state.dim() == 2: state = state.unsqueeze(0)

        # 1. Actor forward
        x = self.actor_net(state)
        x = x.reshape(x.size(0), -1)  # Flatten: (Batch, Seq*Embed)
        logits = self.actor_head(x)

        # 2. Critic forward (获取当前状态价值)
        v_x = self.critic_net(state)
        v_x = v_x.reshape(v_x.size(0), -1)
        value = self.critic_head(v_x)

        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()

        return action, dist.log_prob(action), value, dist.entropy()

    def evaluate(self, state, action):
        """用于 PPO 更新 (Training)"""
        # 1. Actor forward
        x = self.actor_net(state)
        x = x.reshape(x.size(0), -1)
        logits = self.actor_head(x)

        # 2. Critic forward
        v_x = self.critic_net(state)
        v_x = v_x.reshape(v_x.size(0), -1)
        value = self.critic_head(v_x)

        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, value, dist_entropy