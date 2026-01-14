# configs/config.py
import numpy as np


class Config:
    # ================= 1. 物理模型参数 (保持原样) =================
    PARAM_ZETA_D = 150.0
    PARAM_K = 1.2
    PARAM_C1 = 0.75
    PARAM_C2 = 0.25
    PARAM_C3 = 0.75
    PARAM_C4 = 0.25

    # ================= 2. 天气影响参数 (保持原样) =================
    WEATHER_LEVEL_RANGE = (1, 6)
    # 这些矩阵保留着，但在 Easy Mode 下暂时不会起作用，因为下面我们会把因子设为 1.0
    WEATHER_FACTOR_TYPE1 = np.array([
        [0.85, 0.85, 0.85, 0.85, 0.75],
        [0.85, 0.85, 0.85, 0.85, 0.75],
        [0.85, 0.85, 0.85, 0.85, 0.75],
        [0.85, 0.85, 0.85, 0.85, 0.75],
        [0.75, 0.75, 0.75, 0.75, 0.75]
    ])
    WEATHER_FACTOR_TYPE2 = np.array([
        [0.9, 0.9, 0.9, 0.8, 0.8],
        [0.9, 0.9, 0.9, 0.8, 0.8],
        [0.9, 0.9, 0.9, 0.8, 0.8],
        [0.9, 0.9, 0.9, 0.8, 0.8],
        [0.8, 0.8, 0.8, 0.8, 0.8]
    ])

    # ================= 3. 仿真环境参数 (【重点修改区域】) =================
    MAP_WIDTH = 180.0
    MAP_HEIGHT = 160.0

    # 【修改点 1：缩短距离】
    # 原文是 (0, 30)，距离目标约150km，命中率极低。
    # 改为 (60, 90)，距离目标约90km，基础命中率会大幅提升。
    UAV_GEN_X_RANGE = (60, 90)
    TARGET_GEN_X_RANGE = (160, 180)

    NUM_UAVS = 30
    NUM_TARGETS = 10

    # 【修改点 2：减少干扰】
    # 原文是 2 个禁飞区 + 2 个拦截器，且拦截半径大。
    # 改为 1 个禁飞区 + 1 个拦截器，且拦截半径减小。
    NUM_NFZ = 1
    NUM_INTERCEPTORS = 1
    INTERCEPT_RAD = 2.0  # 原 3.0，减小半径更容易绕过

    # 成本与权重
    COST_WEIGHT_OMEGA = 0.0

    # 【修改点 3：移除天气惩罚】
    # 暂时设为 1.0，排除干扰，确保飞机是满状态出击
    WEATHER_SPEED_FACTOR = 1.0
    WEATHER_LOAD_FACTOR = 1.0

    # ================= 4. PPO & 网络参数 (保持原样) =================
    STATE_DIM = 14
    SEQ_LEN = 5
    ACTION_DIM = 2

    # Transformer 参数
    EMBED_DIM = 128
    NUM_HEADS = 8
    NUM_LAYERS = 2

    # 训练超参数
    LR_ACTOR = 2e-4
    LR_CRITIC = 1e-3
    GAMMA = 0.998
    GAE_LAMBDA = 0.95

    K_EPOCHS = 5
    EPS_CLIP = 0.2
    BATCH_SIZE = 64

    GRAD_NORM_CLIP = 1.0

    MAX_EPISODES = 2000
    RESET_EPISODES = 200

    SEED = 42


cfg = Config()