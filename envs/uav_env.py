# envs/uav_env.py
import gym
import numpy as np
from gym import spaces
from configs.config import cfg
from envs.entities import UAV, Target, NoFlyZone, Interceptor
from envs.mechanics import get_state_vector, get_distance, calc_damage_prob
from collections import deque


class UAVEnv(gym.Env):
    def __init__(self):
        super(UAVEnv, self).__init__()

        # 定义动作空间: 0 (Skip/不分配), 1 (Assign/分配)
        self.action_space = spaces.Discrete(cfg.ACTION_DIM)

        # 定义状态空间 (Seq_Len, State_Dim)
        # 对应 Config 中 SEQ_LEN=20
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(cfg.SEQ_LEN, cfg.STATE_DIM), dtype=np.float32
        )

        # 内部实体列表
        self.uavs = []
        self.targets = []
        self.nfz_list = []
        self.interceptors = []

        # 决策指针 (Pointer)
        self.uav_idx = 0  # 当前轮到哪架 UAV
        self.target_idx = 0  # 当前轮到哪个 Target

        # 历史状态队列 (Experience Buffer)
        self.state_buffer = deque(maxlen=cfg.SEQ_LEN)

    def reset(self):
        """ 重置环境，生成红蓝对抗布局 """
        # 1. 清空实体
        self.uavs = []
        self.targets = []
        self.nfz_list = []
        self.interceptors = []

        # 【核心修正 1】生成 UAVs (红蓝对抗布局: 左侧)
        # 使用 Config 中定义的范围 (0, 30)
        for i in range(cfg.NUM_UAVS):
            pos_x = np.random.uniform(cfg.UAV_GEN_X_RANGE[0], cfg.UAV_GEN_X_RANGE[1])
            pos_y = np.random.uniform(0, cfg.MAP_HEIGHT)
            pos = np.array([pos_x, pos_y])

            # 随机速度
            vel = (np.random.rand(2) - 0.5) * cfg.UAV_SPEED_RANGE[1]
            self.uavs.append(UAV(
                id=i, pos=pos, velocity=vel,
                max_speed=cfg.UAV_SPEED_RANGE[1], load=cfg.UAV_LOAD_CAPACITY
            ))

        # 【核心修正 2】生成 Targets (红蓝对抗布局: 右侧)
        # 使用 Config 中定义的范围 (160, 180)
        for i in range(cfg.NUM_TARGETS):
            pos_x = np.random.uniform(cfg.TARGET_GEN_X_RANGE[0], cfg.TARGET_GEN_X_RANGE[1])
            pos_y = np.random.uniform(0, cfg.MAP_HEIGHT)
            pos = np.array([pos_x, pos_y])

            val = np.random.uniform(1.0, 10.0)
            self.targets.append(Target(id=i, pos=pos, value=val))

        # 生成禁飞区 (NFZ)
        for i in range(cfg.NUM_NFZ):
            pos = np.random.rand(2) * [cfg.MAP_WIDTH, cfg.MAP_HEIGHT]
            self.nfz_list.append(NoFlyZone(id=i, pos=pos, radius=cfg.MAP_WIDTH * 0.05))

        # 生成拦截者
        for i in range(cfg.NUM_INTERCEPTORS):
            pos = np.random.rand(2) * [cfg.MAP_WIDTH, cfg.MAP_HEIGHT]
            self.interceptors.append(Interceptor(id=i, pos=pos, radius=cfg.INTERCEPT_RAD))

        # 【关键保留】打乱目标列表顺序 (Target Shuffling)
        # 防止智能体只学到"打前几个目标"的短视策略
        np.random.shuffle(self.targets)

        # 2. 重置指针
        self.uav_idx = 0
        self.target_idx = 0

        # 3. 初始化 State Buffer
        self.state_buffer.clear()
        for _ in range(cfg.SEQ_LEN):
            self.state_buffer.append(np.zeros(cfg.STATE_DIM, dtype=np.float32))

        return self._get_obs()

    # def _get_obs(self):
    #     """ 获取时间窗口状态堆叠 """
    #     if self.uav_idx >= len(self.uavs):
    #         current_feat = np.zeros(cfg.STATE_DIM, dtype=np.float32)
    #     else:
    #         curr_uav = self.uavs[self.uav_idx]
    #         curr_target = self.targets[self.target_idx]
    #         # 调用 mechanics 获取 14维特征
    #         current_feat = get_state_vector(curr_uav, curr_target, self.nfz_list, self.interceptors)
    #
    #         # 特征归一化: 将 Target Value (1~10) 归一化到 [0.1, 1.0]
    #         # 假设 mechanics.py 返回的第 8 个特征 (index 7) 是 Value
    #         current_feat[7] /= 10.0
    #
    #     self.state_buffer.append(current_feat)
    #     return np.array(self.state_buffer, dtype=np.float32)

    def _get_obs(self):
        """ 获取时间窗口状态堆叠 """
        if self.uav_idx >= len(self.uavs):
            current_feat = np.zeros(cfg.STATE_DIM, dtype=np.float32)
        else:
            curr_uav = self.uavs[self.uav_idx]
            curr_target = self.targets[self.target_idx]

            # --- 【新增】计算全局统计特征 ---
            # 1. 计算已消耗的成本比例 (chi_c)
            # 已经分配出去的 UAV 数量 / 总 UAV 数量 (假设 Cost=1)
            # self.uav_idx 表示当前正在决策第几架飞机，前面的 (0 到 uav_idx-1) 既然已经跳过了，
            # 说明我们要统计的是“已经 Assign 的”。
            # 更准确的计算：遍历所有 UAV 检查 available 状态
            assigned_count = sum([1 for u in self.uavs if not u.available])
            total_cost_ratio = assigned_count / cfg.NUM_UAVS

            # 2. 计算已覆盖的价值比例 (chi_v)
            covered_value = 0.0
            total_value = sum([t.value for t in self.targets])
            for t in self.targets:
                if len(t.locked_by_uavs) > 0:
                    covered_value += t.value
            total_value_ratio = covered_value / (total_value + 1e-6)

            global_stats = {
                'cost_ratio': total_cost_ratio,
                'value_ratio': total_value_ratio
            }

            # 传入 mechanics
            current_feat = get_state_vector(
                curr_uav, curr_target, self.nfz_list, self.interceptors,
                global_stats=global_stats
            )

            # 注意: target.value 在 mechanics 里已经除以 10.0 了，这里不需要再除

        self.state_buffer.append(current_feat)
        return np.array(self.state_buffer, dtype=np.float32)

    def _calc_J_X(self):
        """
        【核心修正 3】计算目标函数 J(X) = 收益 - 成本
        引入成本是为了防止 '过度饱和' (9架飞机打1个目标)
        """
        total_revenue = 0.0
        total_cost = 0.0

        for tgt in self.targets:
            # 1. 计算单目标的毁伤概率 (收益部分)
            not_hit_prob = 1.0
            for uav_id in tgt.locked_by_uavs:
                u_obj = self.uavs[uav_id]
                p_single = calc_damage_prob(u_obj.pos, tgt.pos)
                not_hit_prob *= (1.0 - p_single)

                # 2. 累加成本 (Cost)
                # 只要这架 UAV 被分配了任务，就计入成本
                total_cost += cfg.UAV_COST

            joint_prob = 1.0 - not_hit_prob
            total_revenue += joint_prob * tgt.value

        # 3. 最终目标函数 (Eq. 12)
        # J = G - omega * C
        J_val = total_revenue - (cfg.COST_WEIGHT_OMEGA * total_cost)

        return J_val

    def _calculate_paper_reward(self):
        """
        复现论文 Eq. (19) 的奖励逻辑
        包含：全覆盖翻倍奖励 + 覆盖率惩罚
        """
        J_X = self._calc_J_X()

        # 计算 N0 (覆盖目标数)
        N0 = 0
        for tgt in self.targets:
            if len(tgt.locked_by_uavs) > 0:
                N0 += 1

        M = cfg.NUM_TARGETS

        if N0 == M:
            return 2.0 * J_X  # 全覆盖翻倍
        else:
            return J_X * (float(N0) / float(M))  # 覆盖率折损

    def step(self, action):
        curr_uav = self.uavs[self.uav_idx]
        curr_target = self.targets[self.target_idx]

        done = False

        # 1. 动作前奖励
        prev_r = self._calculate_paper_reward()

        # 2. 执行动作
        if action == 1:  # Assign
            curr_uav.assigned_target_id = curr_target.id
            curr_uav.available = False
            curr_target.locked_by_uavs.append(curr_uav.id)
            self.uav_idx += 1
            self.target_idx = 0
        else:  # Skip
            self.target_idx += 1
            if self.target_idx >= len(self.targets):
                self.uav_idx += 1
                self.target_idx = 0

        # 3. 动作后奖励
        new_r = self._calculate_paper_reward()

        # 4. 计算增量奖励
        if action == 1:
            # 使用较小的缩放因子，因为 J(X) 现在包含了成本扣除，数值会变小
            reward = (new_r - prev_r) / 5.0
        else:
            reward = -0.005  # 微小的时间惩罚

        # 5. 检查结束
        if self.uav_idx >= len(self.uavs):
            done = True

        return self._get_obs(), reward, done, {}