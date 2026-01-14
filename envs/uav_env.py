# envs/uav_env.py
import gym
import numpy as np
import random
from collections import deque
from gym import spaces

from configs.config import cfg
from envs.entities import UAV, Target, NoFlyZone, Interceptor
# 确保 mechanics.py 已经是第一阶段修改后的版本
from envs.mechanics import get_state_vector, calc_advantage

class UAVEnv(gym.Env):
    def __init__(self):
        super(UAVEnv, self).__init__()

        # 动作空间: 0 (Skip/不分配), 1 (Assign/分配)
        self.action_space = spaces.Discrete(cfg.ACTION_DIM)

        # 状态空间 (Seq_Len, State_Dim)
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

        # --- 【Modification 2 (Partial)】: 暂时使用常数模拟天气影响 ---
        # 实际论文中应查表 (Table III/IV)，此处先用 Config 中的默认常数顶替
        # 确保 mechanics 计算时使用的是折损后的物理属性
        weather_speed_factor = cfg.WEATHER_SPEED_FACTOR  # e.g. 0.85
        weather_load_factor = cfg.WEATHER_LOAD_FACTOR    # e.g. 0.90

        # --- 1. 生成异构 UAVs (Section IV-A-3) ---
        # 比例 3:1
        num_type2 = cfg.NUM_UAVS // 4
        num_type1 = cfg.NUM_UAVS - num_type2
        uav_types = [1] * num_type1 + [2] * num_type2
        random.shuffle(uav_types)

        for i, u_type in enumerate(uav_types):
            # 位置: 左侧区域 U(0, 30)
            pos_x = np.random.uniform(cfg.UAV_GEN_X_RANGE[0], cfg.UAV_GEN_X_RANGE[1])
            pos_y = np.random.uniform(0, cfg.MAP_HEIGHT)
            pos = np.array([pos_x, pos_y])

            # 属性初始化 [Table II & Section IV-A-3]
            if u_type == 1:
                # Type 1: v ~ [350, 500] m/s -> [0.35, 0.5] km/s
                base_speed = np.random.uniform(0.35, 0.50)
                cost = 1.0
                base_load = 0.95
            else:
                # Type 2: v ~ [750, 900] m/s -> [0.75, 0.9] km/s
                base_speed = np.random.uniform(0.75, 0.90)
                cost = 1.25
                base_load = 1.0

            # 【关键修改】应用天气折损
            # 这确保了 mechanics.py 中计算 p_damage 和 E_speed 时使用的是真实作战能力
            real_speed = base_speed * weather_speed_factor
            real_load = base_load * weather_load_factor

            # 随机初始航向 (-15 ~ 15 deg)
            angle = np.deg2rad(np.random.uniform(-15, 15))
            vel = np.array([np.cos(angle), np.sin(angle)]) * real_speed

            # 初始化 UAV 对象
            uav = UAV(id=i, pos=pos, velocity=vel, max_speed=real_speed, load=real_load)
            uav.cost = cost
            uav.uav_type = u_type # 记录类型以便后续扩展
            self.uavs.append(uav)
            self.total_swarm_cost += cost

        # --- 2. 生成分级 Targets (Section IV-A-2) ---
        M = cfg.NUM_TARGETS
        n1 = M // 2  # Level 1 (Val=4)
        n4 = 1       # Level 4 (Val=16)
        n_remain = M - n1 - n4
        n2 = np.random.randint(1, n_remain + 1) if n_remain >= 1 else 0
        n3 = n_remain - n2

        target_vals = [4.0] * n1 + [6.0] * n2 + [8.0] * n3 + [16.0] * n4
        random.shuffle(target_vals)

        for i in range(cfg.NUM_TARGETS):
            # 位置: 右侧区域 U(160, 180)
            pos_x = np.random.uniform(cfg.TARGET_GEN_X_RANGE[0], cfg.TARGET_GEN_X_RANGE[1])
            pos_y = np.random.uniform(0, cfg.MAP_HEIGHT)
            pos = np.array([pos_x, pos_y])

            # 目标速度: v ~ [10, 15] m/s -> [0.01, 0.015] km/s
            vel = (np.random.rand(2) - 0.5) * 0.03  # 简单随机速度

            tgt = Target(id=i, pos=pos, value=target_vals[i])
            tgt.velocity = vel
            self.targets.append(tgt)

        # --- 3. 生成环境障碍 (Table II) ---
        self.nfz_list = []  # 确保清空
        # 禁飞区 (NFZ)
        for i in range(cfg.NUM_NFZ):
            radius = np.random.uniform(5, 10)
            # 修改位置生成逻辑
            pos_x = np.random.uniform(120, 140)
            pos_y = np.random.uniform(0, cfg.MAP_HEIGHT)
            pos = np.array([pos_x, pos_y])
            self.nfz_list.append(NoFlyZone(id=i, pos=pos, radius=radius))

        # 拦截者 (Interceptor)
        self.interceptors = []  # 确保清空
        for i in range(cfg.NUM_INTERCEPTORS):
            radius = cfg.INTERCEPT_RAD  # 3.0 km
            # 修改位置生成逻辑
            pos_x = np.random.uniform(140, 160)
            pos_y = np.random.uniform(0, cfg.MAP_HEIGHT)
            pos = np.array([pos_x, pos_y])
            # 速度生成逻辑保持你之前的修复 (0.30 - 0.32)
            inter_speed = np.random.uniform(0.30, 0.32)
            angle = np.random.uniform(0, 2 * np.pi)
            vel = np.array([np.cos(angle), np.sin(angle)]) * inter_speed

            inter = Interceptor(id=i, pos=pos, radius=radius)
            inter.velocity = vel  # 动态属性
            self.interceptors.append(inter)

        # 初始打乱目标顺序
        np.random.shuffle(self.targets)

    def _reset_state_only(self):
        """仅重置状态，不改变布局"""
        for u in self.uavs:
            u.available = True
            u.assigned_target_id = -1

        for t in self.targets:
            t.locked_by_uavs = []

    def _get_obs(self):
        """
        获取当前状态向量，包含论文 Eq. (15) 所需的所有上下文信息
        """
        if self.uav_idx >= len(self.uavs):
            return np.zeros(cfg.STATE_DIM, dtype=np.float32)

        curr_uav = self.uavs[self.uav_idx]
        curr_target = self.targets[self.target_idx]

        # --- 1. 计算全局统计特征 (Global Stats) ---
        current_assigned_cost = sum([getattr(u, 'cost', 1.0) for u in self.uavs if not u.available])
        chi_c = current_assigned_cost / (self.total_swarm_cost + 1e-6)

        total_val = sum([t.value for t in self.targets])
        covered_val = sum([t.value for t in self.targets if len(t.locked_by_uavs) > 0])
        chi_v = covered_val / (total_val + 1e-6)

        tgt_assigned_cost = 0.0
        for uid in curr_target.locked_by_uavs:
            u_obj = next((u for u in self.uavs if u.id == uid), None)
            if u_obj: tgt_assigned_cost += getattr(u_obj, 'cost', 1.0)
        chi_mc = tgt_assigned_cost / (self.total_swarm_cost + 1e-6)

        global_stats = {
            'cost_ratio': chi_c,
            'value_ratio': chi_v,
            'target_cost_ratio': chi_mc
        }

        # --- 2. 计算当前目标的联合优势度 (\bar{p}_m) 和 收益 (\bar{G}_m) ---
        not_hit_prob = 1.0
        not_hit_prob_pure = 1.0  # 不含环境干扰的未命中率

        for uid in curr_target.locked_by_uavs:
            u_obj = next((u for u in self.uavs if u.id == uid), None)
            if u_obj:
                # 获取 p_final 和 p_damage_only (Phase 1 mechanics)
                p_adv, p_pure = calc_advantage(u_obj, curr_target, self.nfz_list, self.interceptors)
                not_hit_prob *= (1.0 - p_adv)
                not_hit_prob_pure *= (1.0 - p_pure)

        prev_joint_p = 1.0 - not_hit_prob
        prev_joint_p_pure = 1.0 - not_hit_prob_pure
        prev_revenue = prev_joint_p * curr_target.value

        # --- 3. 生成状态向量 ---
        current_feat = get_state_vector(
            curr_uav, curr_target, self.nfz_list, self.interceptors,
            global_stats=global_stats,
            prev_joint_p=prev_joint_p,
            prev_revenue=prev_revenue,
            prev_joint_p_damage_only=prev_joint_p_pure
        )

        # --- 【修复】归一化 ---
        norm_feat = np.copy(raw_feat)
        # 坐标除以地图大小，价值除以最大价值
        norm_feat[3] /= 180.0  # x
        norm_feat[4] /= 160.0  # y
        norm_feat[7] /= 180.0  # target_x
        norm_feat[8] /= 160.0  # target_y
        norm_feat[9] /= 16.0  # value
        # 距离除以对角线长度 (约240)
        norm_feat[10] /= 240.0

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
                    # 使用 Phase 1 修正后的 calc_advantage
                    p_adv, _ = calc_advantage(u_obj, tgt, self.nfz_list, self.interceptors)
                    not_hit_prob *= (1.0 - p_adv)
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

        # --- 【修复】奖励缩放 ---
        # 估算最大可能得分约为 2 * 总价值 (10个目标 * 平均价值6 * 2倍 = 120左右)
        # 将其缩放到 0~10 的范围
        normalized_reward = raw_reward / 10.0

        return normalized_reward

    def step(self, action):
        curr_uav = self.uavs[self.uav_idx]
        curr_target = self.targets[self.target_idx]
        done = False

        # 1. 记录动作前的 r(X) (即论文中的 r(X))
        prev_r = self._calculate_paper_reward()

        reward = 0.0

        # 2. 执行动作逻辑
        if action == 1:  # 尝试分配 (Assign)
            # --- 预执行 (Tentative Execution) ---
            curr_uav.assigned_target_id = curr_target.id
            curr_uav.available = False
            curr_target.locked_by_uavs.append(curr_uav.id)

            # 计算潜在的新奖励 r(X')
            new_r = self._calculate_paper_reward()

            # --- 【Critical Fix: Eq. 21 条件判断】 ---
            # 只有当新方案的评分非降 (r(X') >= r(X)) 时，才确认分配
            if new_r >= prev_r:
                # [Case A: 接受分配]
                # 状态更新：X -> X' (已在预执行中完成)
                # 即时奖励 R(s,a) = r(X') - r(X) [Eq. 17]
                reward = new_r - prev_r

                # 逻辑流转：该 UAV 已分配任务，跳出循环，处理下一架 UAV
                self.uav_idx += 1
                self.target_idx = 0
            else:
                # [Case B: 拒绝分配 (Rollback)]
                # 状态保持：X 不变 (回滚预执行的操作)
                curr_uav.assigned_target_id = -1
                curr_uav.available = True
                curr_target.locked_by_uavs.pop()  # 移除刚才添加的 UAV ID

                # 即时奖励：由于状态未变，X' = X，故 r(X') - r(X) = 0
                reward = 0.0

                # 逻辑流转：虽然 Agent 选择了 Assign，但被规则拒绝。
                # 按照序贯决策逻辑，这等同于被迫 "Skip"，继续检查当前 UAV 的下一个目标
                self.target_idx += 1
                if self.target_idx >= len(self.targets):
                    # 如果遍历完所有目标都未分配/被拒绝，轮到下一架 UAV
                    self.uav_idx += 1
                    self.target_idx = 0

        else:  # action == 0 (Skip)
            # 保持状态不变，检查下一个目标
            reward = 0.0
            self.target_idx += 1

            # 边界检查：当前 UAV 遍历完所有 Target
            if self.target_idx >= len(self.targets):
                self.uav_idx += 1
                self.target_idx = 0

        # 3. 检查 Episode 是否结束
        if self.uav_idx >= len(self.uavs):
            done = True

        # 4. 【Reward Correction】: 引入 Goal Reward (Eq. 20)
        # 论文 Eq. 20 显示总目标是最大化累积奖励 + 最终目标奖励
        # 但在 PPO 实现中，通常将最终局面的评分加在最后一步
        if done:
            final_r = self._calculate_paper_reward()
            reward += final_r  # R_goal = r(X_final)

        # 5. 获取新状态
        # 如果 done=True，_get_obs 通常返回全零或最后状态，根据你的实现逻辑
        obs = self._get_obs()

        # 统计当前所有“已锁定”关系的平均物理概率，用于诊断环境难度
        total_pen_prob = 0.0
        total_dmg_prob = 0.0
        count = 0

        # 遍历所有目标，检查谁锁定了它
        for tgt in self.targets:
            for uid in tgt.locked_by_uavs:
                u_obj = next((u for u in self.uavs if u.id == uid), None)
                if u_obj:
                    # 调用 mechanics 计算具体的概率分量
                    # 注意：需要确保已 import mechanics
                    # p_final = p_damage * p_pen
                    p_final, p_damage_only = calc_advantage(u_obj, tgt, self.nfz_list, self.interceptors)

                    # 反推突防概率 (加 1e-6 防止除以 0)
                    p_pen = p_final / (p_damage_only + 1e-9)
                    # 也可以重新调用 calc_penetration_prob，但这样反推更省计算量
                    # 不过要注意 p_pen 可能会因为 p_damage=0 而计算不准，
                    # 更稳妥的方式是直接调用 mechanics.calc_penetration_prob(u_obj, tgt, ...)
                    # 鉴于 import 问题，这里直接用 p_final 和 p_damage_only 统计即可

                    total_dmg_prob += p_damage_only
                    # 如果 p_damage_only 太小，说明根本没法攻击，p_final 也是 0，p_pen 意义不大
                    # 我们这里简单统计 p_final (联合概率) 和 p_damage (基础概率)
                    # 通过比较这两个值，就能知道损耗了多少

                    # 修正：为了准确记录“突防概率”，我们还是计算一下比值，或者仅记录 p_final 和 p_dmg
                    # 如果 p_dmg 是 0.8，p_final 是 0.4，说明突防率是 0.5

                    count += 1

        avg_p_dmg = total_dmg_prob / count if count > 0 else 0.0
        # 这里记录“联合概率(Final)”，它等于 Dmg * Pen。
        # 在 CSV 分析时，用 Avg_Final / Avg_Dmg 就能算出突防率
        total_final_prob = sum([calc_advantage(
            next(u for u in self.uavs if u.id == uid), t, self.nfz_list, self.interceptors)[0]
                                for t in self.targets for uid in t.locked_by_uavs
                                ])
        avg_p_final = total_final_prob / count if count > 0 else 0.0



        # ================== 【新增修改开始】 ==================
        # 在 info 中返回真实的目标函数值 J(X)，用于画出论文 Fig 3 的曲线
        info = {}
        if done:
            # 调用现有的 _calc_J_X() 方法获取纯粹的 J(X)
            true_objective_value = self._calc_J_X()
            info['final_J'] = true_objective_value
        # ================== 【新增修改结束】 ==================





        # 在 return 之前收集信息
        info = {
            "J_val": self._calc_J_X(),  # 当前方案的 J(X)
            "num_assigned": sum([1 for t in self.targets if len(t.locked_by_uavs) > 0]),  # 覆盖目标数
            "is_valid_action": (reward != 0) if action == 1 else None,  # 是否是有效分配
            # 【新增】返回这两个关键物理指标
            "avg_p_dmg": avg_p_dmg,  # 纯毁伤概率 (如果不考虑障碍物，能打多少分)
            "avg_p_final": avg_p_final  # 最终概率 (考虑障碍物后，实际打多少分)
        }

        return obs, reward, done, info