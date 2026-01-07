# configs/config.py

class Config:
    # ================= 仿真环境参数 (Table II) =================
    # 地图规模 (单位: km) -> 对应 Group 1
    MAP_WIDTH = 180.0
    MAP_HEIGHT = 160.0

    # 【新增】生成区域范围 (红蓝对抗布局)
    # UAV 生成在左侧 0-30km 区域
    UAV_GEN_X_RANGE = (0, 30)
    # Target 生成在右侧 (地图宽度的最后20km)
    TARGET_GEN_X_RANGE = (160, 180)

    # 实体数量
    NUM_UAVS = 20     # 论文 Group 1 Ratio=2: 20架无人机 vs 10个目标
    NUM_TARGETS = 10  # 论文 Small Scale: 10
    NUM_NFZ = 2       # 禁飞区数量
    NUM_INTERCEPTORS = 2  # 拦截系统数量

    # UAV 属性
    UAV_SPEED_RANGE = (0.3, 0.5)  # km/s
    UAV_LOAD_CAPACITY = 10.0      # kg

    # 【新增】无人机成本参数 (对应 Eq. 12)
    # 引入成本是为了防止智能体为了微小的命中率提升而无限堆叠无人机
    UAV_COST = 1.0          # 单架无人机的基础派遣成本
    COST_WEIGHT_OMEGA = 0.5 # 成本在总奖励函数中的权重 (调节收益与成本的平衡)

    # 天气影响系数 (Table III & IV)
    WEATHER_SPEED_FACTOR = 0.85
    WEATHER_LOAD_FACTOR = 0.90

    # ================= 物理模型参数 (公式常数) =================
    PARAM_C1 = 0.75  # 毁伤概率距离系数 (公式 4)
    PARAM_K = 5.0    # 角度评分平滑系数 (公式 1)
    INTERCEPT_RAD = 3.0   # 拦截半径 (km)
    PARAM_ZETA_D = 25.0   # 距离评估的归一化参数

    # ================= PPO & 网络参数 (Table I) =================
    STATE_DIM = 14   # 输入特征维度 (必须包含成本比例等统计特征)
    SEQ_LEN = 20     # 【关键】增加时间窗口至 20，让 Critic 能看到更早的分配历史，避免过度饱和
    ACTION_DIM = 2   # 输出: 0(Skip), 1(Assign)

    EMBED_DIM = 128  # Transformer 隐藏层维度
    NUM_HEADS = 8    # Attention 头数
    NUM_LAYERS = 2   # Encoder 层数

    LR_ACTOR = 2e-4  # 学习率
    LR_CRITIC = 1e-3
    GAMMA = 0.998    # 折扣因子
    K_EPOCHS = 5     # PPO 更新轮次
    EPS_CLIP = 0.2   # PPO Clip 阈值

    BATCH_SIZE = 64
    MAX_EPISODES = 6000 # 增加训练轮数以保证在随机环境下收敛

    # 随机种子
    SEED = 42

# 实例化供其他模块调用
cfg = Config()