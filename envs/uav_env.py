# envs/uav_env.py
import gym
import numpy as np
import random
from collections import deque
from gym import spaces

from configs.config import cfg
from envs.entities import UAV, Target, NoFlyZone, Interceptor
# 注意：确保 mechanics.py 已更新并包含 calc_advantage
from envs.mechanics import get_state_vector, calc_advantage


class UAVEnv(gym.Env):
    def __init__(self):
        super(UAVEnv, self).__init__()

        # 动作空间: 0 (Skip/不分配), 1 (Assign/分配)
        self.action_space = spaces.Discrete(cfg.ACTION_DIM)

        # 状态空间 (Seq_Len, State_Dim)
        # 对应论文 "5-step temporal window"
        # 建议在 Config 中将 SEQ_LEN 设为 5 以严格匹配论文
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(cfg.SEQ_LEN, cfg.STATE_DIM), dtype=np.float32
        )

        # 内部实体列表
        self.uavs = []
        self.targets = []
        self.nfz_list = []
        self.interceptors = []

        # 决策指针
        self.uav_idx = 0  # 当前决策的 UAV 索引
        self.target_idx = 0  # 当前决策的 Target 索引

        # 历史状态队列 (Experience Buffer)
        self.state_buffer = deque(maxlen=cfg.SEQ_LEN)

        # 辅助统计量
        self.total_swarm_cost = 0.0

    def reset(self, full_reset=True):
        """
        重置环境。
        :param full_reset:
            True  -> 完全重生成场景（位置、属性），对应论文中每 200 episodes 的重置。
            False -> 仅重置分配状态和指针，保持场景布局不变，用于同场景下的反复试错学习。
        """
        if full_reset:
            self._generate_scene()
        else:
            self._reset_state_only()

        # 初始化指针
        self.uav_idx = 0
        self.target_idx = 0

        # 初始化 State Buffer (填充零向量)
        self.state_buffer.clear()
        for _ in range(cfg.SEQ_LEN):
            self.state_buffer.append(np.zeros(cfg.STATE_DIM, dtype=np.float32))

        return self._get_obs()

    def _generate_scene(self):
        """生成全新的对抗场景 (Table II & Section IV-A)"""
        self.uavs = []
        self.targets = []
        self.nfz_list = []
        self.interceptors = []
        self.total_swarm_cost = 0.0

        # --- 1. 生成异构 UAVs (Section IV-A-3) ---
        # 比例 3:1
        num_type2 = cfg.NUM_UAVS // 4
        num_type1 = cfg.NUM_UAVS - num_type2
        uav_types = [1] * num_type1 + [2] * num_type2
        random.shuffle(uav_types)

        for i, u_type in enumerate(uav_types):
            # 位置: 左侧区域 U(0, 30) [Table II]
            pos_x = np.random.uniform(cfg.UAV_GEN_X_RANGE[0], cfg.UAV_GEN_X_RANGE[1])
            pos_y = np.random.uniform(0, cfg.MAP_HEIGHT)
            pos = np.array([pos_x, pos_y])

            # 属性初始化 [Table II & Section IV-A-3]
            if u_type == 1:
                # Type 1: v ~ [350, 500] m/s -> [0.35, 0.5] km/s (需统一单位，假设地图单位km)
                speed_mag = np.random.uniform(0.35, 0.50)
                cost = 1.0
                load = 0.95
            else:
                # Type 2: v ~ [750, 900] m/s -> [0.75, 0.9] km/s
                speed_mag = np.random.uniform(0.75, 0.90)
                cost = 1.25
                load = 1.0

            # 随机初始航向 (-15 ~ 15 deg)
            angle = np.deg2rad(np.random.uniform(-15, 15))
            vel = np.array([np.cos(angle), np.sin(angle)]) * speed_mag

            uav = UAV(id=i, pos=pos, velocity=vel, max_speed=speed_mag, load=load)
            uav.cost = cost  # 动态绑定 cost 属性
            self.uavs.append(uav)
            self.total_swarm_cost += cost

        # --- 2. 生成分级 Targets (Section IV-A-2) ---
        M = cfg.NUM_TARGETS
        n1 = M // 2  # Level 1 (Val=4): M/2 个
        n4 = 1  # Level 4 (Val=16): 1 个
        n_remain = M - n1 - n4
        # Level 2 (Val=6) & Level 3 (Val=8): 剩余随机
        n2 = np.random.randint(1, n_remain + 1) if n_remain >= 1 else 0
        n3 = n_remain - n2

        target_vals = [4.0] * n1 + [6.0] * n2 + [8.0] * n3 + [16.0] * n4
        random.shuffle(target_vals)

        for i in range(cfg.NUM_TARGETS):
            # 位置: 右侧区域 U(160, 180) [Table II]
            pos_x = np.random.uniform(cfg.TARGET_GEN_X_RANGE[0], cfg.TARGET_GEN_X_RANGE[1])
            pos_y = np.random.uniform(0, cfg.MAP_HEIGHT)
            pos = np.array([pos_x, pos_y])

            # 目标速度: v ~ [10, 15] m/s -> [0.01, 0.015] km/s (假设低速移动)
            # 论文提到的目标也有速度，影响角度计算
            vel = (np.random.rand(2) - 0.5) * 0.03  # 简单随机速度

            tgt = Target(id=i, pos=pos, value=target_vals[i])
            tgt.velocity = vel  # 动态绑定 velocity
            self.targets.append(tgt)

        # --- 3. 生成环境障碍 (Table II) ---
        # 禁飞区 (NFZ)
        for i in range(cfg.NUM_NFZ):
            radius = np.random.uniform(5, 10)
            # 位置随机 (参考 Table II 的范围，这里简化为全图随机分布)
            pos = np.random.rand(2) * [cfg.MAP_WIDTH, cfg.MAP_HEIGHT]
            self.nfz_list.append(NoFlyZone(id=i, pos=pos, radius=radius))

        # 拦截者 (Interceptor)
        for i in range(cfg.NUM_INTERCEPTORS):
            radius = cfg.INTERCEPT_RAD  # 3.0 km
            pos = np.random.rand(2) * [cfg.MAP_WIDTH, cfg.MAP_HEIGHT]
            self.interceptors.append(Interceptor(id=i, pos=pos, radius=radius))

        # 初始打乱目标顺序，模拟随机出现的任务序列
        # 注意：full_reset 时打乱列表顺序
        np.random.shuffle(self.targets)

    def _reset_state_only(self):
        """仅重置状态，不改变布局"""
        for u in self.uavs:
            u.available = True
            u.assigned_target_id = -1

        for t in self.targets:
            t.locked_by_uavs = []

        # 注意：这里不再次 shuffle targets，以保证在同一场景下学习稳定性
        # 如果需要每次都随机顺序，也可以在这里 shuffle

    def _get_obs(self):
        """
        获取当前状态向量，包含论文 Eq. (15) 所需的所有上下文信息
        """
        if self.uav_idx >= len(self.uavs):
            return np.zeros(cfg.STATE_DIM, dtype=np.float32)

        curr_uav = self.uavs[self.uav_idx]
        curr_target = self.targets[self.target_idx]

        # --- 1. 计算全局统计特征 (Global Stats) ---
        # chi_c: 全局已消耗成本比例
        current_assigned_cost = sum([getattr(u, 'cost', 1.0) for u in self.uavs if not u.available])
        chi_c = current_assigned_cost / (self.total_swarm_cost + 1e-6)

        # chi_v: 全局已覆盖价值比例
        total_val = sum([t.value for t in self.targets])
        covered_val = sum([t.value for t in self.targets if len(t.locked_by_uavs) > 0])
        chi_v = covered_val / (total_val + 1e-6)

        # chi_{m,c}^w: 当前目标的投入成本比例
        tgt_assigned_cost = 0.0
        for uid in curr_target.locked_by_uavs:
            # 通过 ID 查找 UAV 对象 (假设 id 对应 index)
            # 为了安全起见使用查找
            u_obj = next((u for u in self.uavs if u.id == uid), None)
            if u_obj: tgt_assigned_cost += getattr(u_obj, 'cost', 1.0)
        chi_mc = tgt_assigned_cost / (self.total_swarm_cost + 1e-6)

        global_stats = {
            'cost_ratio': chi_c,
            'value_ratio': chi_v,
            'target_cost_ratio': chi_mc
        }

        # --- 2. 计算当前目标的联合优势度 (\bar{p}_m) 和 收益 (\bar{G}_m) ---
        # 公式: P_joint = 1 - Prod(1 - p_{k,m})
        # 需要遍历所有已经分配给该目标的 UAV

        not_hit_prob = 1.0
        not_hit_prob_pure = 1.0  # 【新增】不含环境干扰的未命中率

        for uid in curr_target.locked_by_uavs:
            u_obj = next((u for u in self.uavs if u.id == uid), None)
            if u_obj:
                # 获取 p_final 和 p_damage_only
                p_adv, p_pure = calc_advantage(u_obj, curr_target, self.nfz_list, self.interceptors)
                not_hit_prob *= (1.0 - p_adv)
                not_hit_prob_pure *= (1.0 - p_pure)  # 【新增】

        prev_joint_p = 1.0 - not_hit_prob
        prev_joint_p_pure = 1.0 - not_hit_prob_pure  # 【新增】
        prev_revenue = prev_joint_p * curr_target.value

        # --- 3. 生成状态向量 ---
        current_feat = get_state_vector(
            curr_uav, curr_target, self.nfz_list, self.interceptors,
            global_stats=global_stats,
            prev_joint_p=prev_joint_p,
            prev_revenue=prev_revenue,
            prev_joint_p_damage_only=prev_joint_p_pure  # 【传入新增参数】
        )

        self.state_buffer.append(current_feat)
        return np.array(self.state_buffer, dtype=np.float32)

    def _calc_J_X(self):
        """
        计算目标函数 J(X) (Eq. 12)
        J(X) = Sum(G_m) - omega * Sum(Cost)
        """
        total_revenue = 0.0
        total_cost = 0.0

        for tgt in self.targets:
            # 计算联合毁伤概率
            not_hit_prob = 1.0
            for uid in tgt.locked_by_uavs:
                u_obj = next((u for u in self.uavs if u.id == uid), None)
                if u_obj:
                    # 必须使用综合优势度 (Advantage) 而非简单的 Distance Prob
                    p_adv, _ = calc_advantage(u_obj, tgt, self.nfz_list, self.interceptors)
                    not_hit_prob *= (1.0 - p_adv)

                    # 累加成本
                    total_cost += getattr(u_obj, 'cost', 1.0)

            joint_p = 1.0 - not_hit_prob
            # G_m = p_m * xi_m
            total_revenue += joint_p * tgt.value

        # J = G - omega * C
        J_val = total_revenue - (cfg.COST_WEIGHT_OMEGA * total_cost)
        return J_val

    def _calculate_paper_reward(self):
        """
        计算论文奖励 r(X) (Eq. 19)
        包含全覆盖奖励翻倍机制
        """
        J_X = self._calc_J_X()

        # 计算 N0 (覆盖目标数)
        N0 = 0
        for tgt in self.targets:
            if len(tgt.locked_by_uavs) > 0:
                N0 += 1

        M = len(self.targets)

        raw_reward = 0.0
        # Eq. 19
        if N0 == M:
            raw_reward = 2.0 * J_X  # 全覆盖翻倍
        else:
            raw_reward = J_X * (float(N0) / float(M))  # 覆盖率折损

        # 【关键修复】 Reward Scaling
        # 将 Reward 除以一个常数（例如 10.0 或 20.0），使其数值保持在个位数
        # 这对 PPO 的收敛至关重要
        return raw_reward / 20.0

    def step(self, action):
        curr_uav = self.uavs[self.uav_idx]
        curr_target = self.targets[self.target_idx]
        done = False

        # 1. 记录动作前的 r(X)
        prev_r = self._calculate_paper_reward()

        # 2. 执行动作
        if action == 1:  # Assign
            curr_uav.assigned_target_id = curr_target.id
            curr_uav.available = False
            curr_target.locked_by_uavs.append(curr_uav.id)

            # 逻辑: 一旦分配，该 UAV 任务结束，处理下一架 UAV
            self.uav_idx += 1
            self.target_idx = 0
        else:  # Skip
            # 逻辑: 查看当前 UAV 的下一个 Target
            self.target_idx += 1

            # 如果当前 UAV 遍历完所有 Target 都未分配 (或主动 Skip 所有)
            if self.target_idx >= len(self.targets):
                # 轮到下一架 UAV
                self.uav_idx += 1
                self.target_idx = 0

        # 3. 记录动作后的 r(X')
        new_r = self._calculate_paper_reward()

        # 4. 计算即时奖励 R(s, a) (Eq. 17)
        # 如果 action=1, R = r(X') - r(X)
        # 如果 action=0, R = 0 (因为状态 X 未变)
        if action == 1:
            reward = new_r - prev_r
        else:
            reward = 0.0

        # 5. 检查结束条件
        if self.uav_idx >= len(self.uavs):
            done = True

        # 6. 获取新状态
        # 注意: 如果 done=True, _get_obs 会返回零向量，这符合常规处理
        obs = self._get_obs()

        return obs, reward, done, {}