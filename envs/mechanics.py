# envs/mechanics.py
import numpy as np
from configs.config import cfg


def get_distance(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)


# --- 论文公式 (1): 角度评价指标 (Angle Evaluation) ---
def calc_angle_score(uav_pos, uav_vel, target_pos):
    vec_u_t = target_pos - uav_pos
    dist = np.linalg.norm(vec_u_t) + 1e-6
    vec_u_t /= dist

    speed = np.linalg.norm(uav_vel) + 1e-6
    vec_v = uav_vel / speed

    # 论文 Fig. 1 和公式 (1): exp(-(sigma/b*pi)^2)
    # 这里保持原代码的 cosine 逻辑作为近似，或者严格按公式实现
    cos_theta = np.dot(vec_u_t, vec_v)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    # 论文参数 b = 0.002 * L (dist)
    # b_val = 0.002 * dist
    # sigma = theta
    # score = np.exp(- (theta / (b_val * np.pi + 1e-6))**2)
    # 原代码使用了 cfg.PARAM_K * theta，这是另一种常见形式，为保持稳定性暂不改为极不稳定的 b*L
    score = np.exp(-cfg.PARAM_K * np.abs(theta))
    return score


# --- 论文公式 (2): 速度评价指标 (Speed Evaluation) ---
def calc_speed_score(uav_speed, target_speed):
    # Eq (2): 1 - (k * v_tgt / v_uav)
    # 这是一个相对速度指标
    if uav_speed < 1e-6: return 0.0
    score = 1.0 - (5.0 * target_speed / uav_speed)
    return np.clip(score, 0.0, 1.0)


# --- 论文公式 (3): 距离评价指标 (Distance Evaluation) ---
def calc_dist_score(dist, is_obstacle=False):
    # Eq (3): exp(-((D - Dmid)/zeta)^2)
    if is_obstacle:
        D_mid = 0
        zeta = cfg.PARAM_ZETA_D
    else:
        # 对于 Target，D_mid 是最优攻击距离，论文设为地图宽度的函数
        # 这里简化处理，假设最优距离为 0 (越近越好) 或保持原逻辑
        D_mid = 0
        zeta = cfg.PARAM_ZETA_D

    score = np.exp(-((dist - D_mid) / zeta) ** 2)
    return score


# --- 论文公式 (4)-(6): 优势度计算 (Advantage Modeling) ---
def calc_advantage(uav, target, nfz_list, interceptor_list):
    """
    计算 UAV 对 Target 的优势度 p_{k,m}
    返回: (final_p, damage_only_p)
    damage_only_p 用于计算 Delta p (战场因素造成的衰减)
    """
    dist = get_distance(uav.pos, target.pos)

    # 1. 毁伤概率 (基于距离和角度) Eq. (4)
    # p_hat = E_angle * (c1*E_dist + c2*E_speed) * Load
    # 这里简化沿用原代码的概率公式，但在论文中是基于 Evaluation Metrics 组合的
    # 为了复现原代码逻辑的稳定性，我们保留 calc_damage_prob 的核心逻辑
    # 但必须返回 "纯毁伤概率" (不含 NFZ/Interceptor)

    p_damage = 1.0 / (1.0 + cfg.PARAM_C1 * dist)  # 原代码逻辑
    # 加上角度影响 (论文 Eq 4)
    angle_score = calc_angle_score(uav.pos, uav.velocity, target.pos)
    p_damage *= angle_score

    # 2. 突防概率 (Penetration Prob) Eq. (5)-(6)
    p_pen = 1.0

    # NFZ
    for nfz in nfz_list:
        # 简单判定：连线是否穿过圆
        # 实际论文用了 E_angle, E_dist 组合
        # 这里沿用原代码的采样检测法，更精确
        d_nfz = get_distance(uav.pos, nfz.pos)
        if d_nfz < nfz.radius + dist:  # 粗筛
            # ... (原代码采样逻辑) ...
            pass

    # 沿用原代码的 calc_penetration_prob 逻辑来计算 p_pen
    p_pen = calc_penetration_prob(uav.pos, target.pos, nfz_list, interceptor_list)

    p_final = p_damage * p_pen
    return p_final, p_damage


def calc_penetration_prob(uav_pos, target_pos, nfz_list, interceptor_list):
    # ... (保持原代码逻辑不变) ...
    prob_survive = 1.0
    num_samples = 5
    for t in np.linspace(0, 1, num_samples):
        sample_point = uav_pos + t * (target_pos - uav_pos)
        for nfz in nfz_list:
            if get_distance(sample_point, nfz.pos) < nfz.radius:
                prob_survive *= (1.0 - nfz.penalty_factor)
    for inter in interceptor_list:
        if get_distance(target_pos, inter.pos) < inter.radius:
            if get_distance(uav_pos, inter.pos) < inter.radius:
                prob_survive *= (1.0 - inter.kill_prob)
    return prob_survive


# --- 【核心修正】 生成符合 Eq. (15) 的状态向量 ---
def get_state_vector(uav, target, nfz_list, interceptor_list, global_stats=None,
                     prev_joint_p=0.0, prev_revenue=0.0):
    """
    严格按照论文 Eq. (15) 构建 14 维状态向量
    参数:
      prev_joint_p: \bar{p}_m (分配前的联合优势度)
      prev_revenue: \bar{G}_m (分配前的收益)
    """

    # 1. 计算当前 UAV-Target 的优势度
    p_km, p_km_damage_only = calc_advantage(uav, target, nfz_list, interceptor_list)

    # 2. 计算假设分配后的联合优势度 \hat{p}_m (Eq. 11)
    # p_new = 1 - (1 - p_old) * (1 - p_km)
    hat_p_m = 1.0 - (1.0 - prev_joint_p) * (1.0 - p_km)

    # 3. 计算假设分配后的收益 \hat{G}_m
    # G = p * Value
    hat_G_m = hat_p_m * target.value

    # 4. 计算 Delta 指标 (战场环境导致的衰减)
    # Delta p_{k,m} = p_{damage_only} - p_{final}
    delta_p_km = p_km_damage_only - p_km

    # Delta p_m (联合概率的衰减增量?)
    # 论文语焉不详，通常指 (理想联合 - 实际联合) 的增量，或者分配前后的 p_m 增量
    # 这里取分配带来的增益: \hat{p}_m - \bar{p}_m
    delta_p_m = hat_p_m - prev_joint_p

    # Delta G_m
    delta_G_m = hat_G_m - prev_revenue

    # 提取全局统计
    if global_stats is None:
        chi_c, chi_v, chi_mc = 0.0, 0.0, 0.0
    else:
        chi_c = global_stats.get('cost_ratio', 0.0)
        chi_v = global_stats.get('value_ratio', 0.0)
        chi_mc = global_stats.get('target_cost_ratio', 0.0)

    # 状态向量 (14维) [Eq. 15]
    # s = [c_k, xi_m, chi_c, chi_v, chi_mc, p_km, bar_p_m, hat_p_m, bar_G_m, hat_G_m, D_pkm, D_pm, D_Gm, x_km]
    state = np.array([
        getattr(uav, 'cost', 1.0),  # 1. c_k^j (UAV Cost)
        target.value,  # 2. xi_m^w (Target Value)
        chi_c,  # 3. chi_c (Global Cost Ratio)
        chi_v,  # 4. chi_v (Global Value Ratio)
        chi_mc,  # 5. chi_{m,c}^w (Target Cost Ratio)
        p_km,  # 6. p_{k,m} (Advantage)
        prev_joint_p,  # 7. bar{p}_m (Prev Joint Prob)
        hat_p_m,  # 8. hat{p}_m (New Joint Prob)
        prev_revenue,  # 9. bar{G}_m (Prev Revenue)
        hat_G_m,  # 10. hat{G}_m (New Revenue)
        delta_p_km,  # 11. Delta p_{k,m} (Env Penalty)
        delta_p_m,  # 12. Delta p_m (Prob Gain)
        delta_G_m,  # 13. Delta G_m (Revenue Gain)
        float(uav.available)  # 14. x_{k,m} (Status / Mask)
    ], dtype=np.float32)

    # 归一化 (可选，建议对 Value 和 Cost 进行缩放以利于训练)
    # 例如 value /= 16.0, cost /= 2.0
    state[0] /= 2.0  # Cost 1.0~1.25
    state[1] /= 16.0  # Value 4~16
    state[8] /= 16.0  # Revenue
    state[9] /= 16.0
    state[12] /= 16.0

    return state