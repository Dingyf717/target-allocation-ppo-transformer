# configs/config.py

class Config:
    # ================= 仿真环境参数 (Table II) =================
    # 地图规模 (单位: km) -> 对应 Group 1
    MAP_WIDTH = 180.0
    MAP_HEIGHT = 160.0

    # UAV 生成在左侧 0-30km 区域
    UAV_GEN_X_RANGE = (0, 30)
    # Target 生成在右侧 (地图宽度的最后20km)
    TARGET_GEN_X_RANGE = (160, 180)

    # 实体数量
    NUM_UAVS = 20
    NUM_TARGETS = 10
    NUM_NFZ = 2
    NUM_INTERCEPTORS = 2

    # UAV 属性
    UAV_SPEED_RANGE = (0.3, 0.5)  # km/s
    UAV_LOAD_CAPACITY = 10.0  # kg

    # 无人机成本参数 (对应 Eq. 12)
    UAV_COST = 1.0
    COST_WEIGHT_OMEGA = 0.5

    # 天气影响系数
    WEATHER_SPEED_FACTOR = 0.85
    WEATHER_LOAD_FACTOR = 0.90

    # ================= 物理模型参数 =================
    PARAM_C1 = 0.75
    PARAM_K = 5.0
    INTERCEPT_RAD = 3.0
    PARAM_ZETA_D = 25.0

    # ================= PPO & 网络参数 (Table I) =================
    STATE_DIM = 14
    SEQ_LEN = 5  # 如果追求严格复现论文的 scale-independent，建议改为 5
    ACTION_DIM = 2

    EMBED_DIM = 128
    NUM_HEADS = 8
    NUM_LAYERS = 2

    LR_ACTOR = 2e-4
    LR_CRITIC = 1e-3
    GAMMA = 0.998  # 折扣因子

    # 【新增】GAE 参数 (Table I)
    GAE_LAMBDA = 0.95

    K_EPOCHS = 5
    EPS_CLIP = 0.2

    BATCH_SIZE = 64
    MAX_EPISODES = 6000

    # 随机种子
    SEED = 42


cfg = Config()