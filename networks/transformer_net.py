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
    """ 独立的 Transformer 模块 (Embedding + Encoder) """

    def __init__(self):
        super(TransformerBlock, self).__init__()
        self.embedding = nn.Sequential(
            init_layer(nn.Linear(cfg.STATE_DIM, cfg.EMBED_DIM)),
            nn.ReLU()
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.EMBED_DIM,
            nhead=cfg.NUM_HEADS,
            dim_feedforward=256,
            dropout=0.0,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.NUM_LAYERS)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return x


class TransformerActorCritic(nn.Module):
    def __init__(self):
        super(TransformerActorCritic, self).__init__()

        self.flat_dim = cfg.SEQ_LEN * cfg.EMBED_DIM

        # --- 1. Actor Network (独立) ---
        self.actor_net = TransformerBlock()
        self.actor_head = nn.Sequential(
            init_layer(nn.Linear(self.flat_dim, 64)),
            nn.ReLU(),
            init_layer(nn.Linear(64, cfg.ACTION_DIM), std=0.01)
        )

        # --- 2. Critic Network (独立) ---
        self.critic_net = TransformerBlock()
        self.critic_head = nn.Sequential(
            init_layer(nn.Linear(self.flat_dim, 64)),
            nn.ReLU(),
            init_layer(nn.Linear(64, 1), std=1.0)
        )

    def forward(self, state):
        if state.dim() == 2: state = state.unsqueeze(0)

        # Actor Path
        x_actor = self.actor_net(state)
        x_actor = x_actor.reshape(x_actor.size(0), -1)
        action_logits = self.actor_head(x_actor)

        # Critic Path
        x_critic = self.critic_net(state)
        x_critic = x_critic.reshape(x_critic.size(0), -1)
        state_value = self.critic_head(x_critic)

        return action_logits, state_value

    def get_action(self, state):
        logits, value = self(state)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), value, dist.entropy()

    def evaluate(self, state, action):
        logits, value = self(state)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        return dist.log_prob(action), value, dist.entropy()