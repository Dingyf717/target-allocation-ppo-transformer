# envs/mechanics.py
import numpy as np
from configs.config import cfg


def get_distance(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)


# --- 论文公式 (1): 角度评价指标 (Angle Evaluation) ---
def calc_angle_score(uav_pos, uav_vel, target_pos):
    """
    计算角度评价指标 E_angle
    Eq. (1): exp(-(sigma / (b * pi))^2)
    其中 sigma 是速度方向与连线的夹角，b = 0.002 * L (L为距离)
    """
    # 1. 计算连线向量和距离
    vec_u_t = target_pos - uav_pos
    dist = np.linalg.norm(vec_u_t)

    # 边界处理：重合时角度得分为 1
    if dist < 1e-6:
        return 1.0

    # 2. 计算夹角 sigma (弧度)
    # 归一化连线向量
    vec_u_t_norm = vec_u_t / dist

    # UAV 速度向量归一化
    speed = np.linalg.norm(uav_vel)
    if speed < 1e-6:
        # 速度为0时，无法定义航向，假设最差情况或保持朝向
        vec_v = np.array([1.0, 0.0])
    else:
        vec_v = uav_vel / speed

    # 计算夹角
    cos_theta = np.dot(vec_u_t_norm, vec_v)
    sigma = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 弧度 [0, pi]

    # 3. 计算动态参数 b
    # b = 0.002 * L
    b_val = 0.002 * dist
    # 防止分母为0 (极近距离时 b_val 极小，指数项会很大，score -> 0，但这不符合直觉)
    # 论文中 b 用于调节 mapping ratio。
    # 修正逻辑：当距离很近时，角度容忍度应该变大还是变小？
    # 通常距离越近，对角度要求越低（容易击中）？或者越高（不仅要近还要准）？
    # 根据公式，L越小，b越小，分母越小，指数绝对值越大，Score衰减越快。
    # 这意味着距离越近，对角度偏差越敏感（High-speed UAV large turning radius）。
    if b_val < 1e-6: b_val = 1e-6

    # 4. 计算指标
    # exponent = (sigma / (b * pi))^2
    exponent = (sigma / (b_val * np.pi)) ** 2
    score = np.exp(-exponent)

    return score


# --- 论文公式 (2): 速度评价指标 (Speed Evaluation) ---
def calc_speed_score(uav_speed, target_speed):
    """
    Eq (2): 1 - (k * v_tgt / v_uav)
    """
    if uav_speed < 1e-6: return 0.0
    # k = 5.0 (cfg.PARAM_K)
    score = 1.0 - (cfg.PARAM_K * target_speed / uav_speed)
    return np.clip(score, 0.0, 1.0)


# --- 论文公式 (3): 距离评价指标 (Distance Evaluation) ---
def calc_dist_score(dist, is_obstacle=False):
    """
    Eq (3): exp(-((D - Dmid)/zeta)^2)
    """
    if is_obstacle:
        # 对于障碍物 (NFZ/Interceptor)，D_mid = 0
        D_mid = 0.0
        zeta = 10
    else:
        # 对于 Target，D_mid 应该是最优攻击距离
        # 论文中 D_mid = [(x_max^T - x_min^U) + (x_min^T - x_max^U)] / 2
        # 这里为了简化复现且保持稳定，暂时沿用 D_mid = 0 (越近越好)
        # 如果需要严格复现，需要传入 map 边界参数
        D_mid = 0.0
        zeta = cfg.PARAM_ZETA_D

    score = np.exp(-((dist - D_mid) / zeta) ** 2)
    return score


# --- 论文公式 (4): 毁伤概率 (Damage Probability) ---
def calc_damage_prob(uav, target):
    """
    Eq. (4): p_hat = E_angle * (c1 * E_dist + c2 * E_speed) * Load
    注意：这里的 Load 应该是经过天气折损后的实际载荷
    """
    dist = get_distance(uav.pos, target.pos)
    uav_speed = np.linalg.norm(uav.velocity)
    target_speed = np.linalg.norm(target.velocity)  # 假设Target有velocity属性

    # 1. 计算三个评价指标
    E_angle = calc_angle_score(uav.pos, uav.velocity, target.pos)
    E_dist = calc_dist_score(dist, is_obstacle=False)
    E_speed = calc_speed_score(uav_speed, target_speed)

    # 2. 组合 (Eq. 4)
    # Load 属性在 uav_env.py 中已经应该是考虑天气后的值 (如果实现了 Mod 2)
    # 这里直接使用 uav.load
    # 确保 load 是归一化或者合理的数值？论文中 load ~ 0.95/1.0
    term_bracket = cfg.PARAM_C1 * E_dist + cfg.PARAM_C2 * E_speed
    p_hat = E_angle * term_bracket * uav.load

    return np.clip(p_hat, 0.0, 1.0)


# --- 论文公式 (5)-(6): 突防概率 (Penetration Probability) ---
def calc_penetration_prob(uav, target, nfz_list, interceptor_list):
    """
    计算针对所有障碍物的联合突防概率
    Eq. (5) for NFZ: p = (1 - E_a)(1 - E_d)
    Eq. (6) for Int: p = (1 - E_a)(c3(1-E_d) + c4 E_v)
    Final P_pen = Prod(p_i)
    """
    p_pen_total = 1.0

    uav_speed = np.linalg.norm(uav.velocity)

    # 1. 遍历禁飞区 (NFZ)
    for nfz in nfz_list:
        # 计算相对于 NFZ 的评价指标
        # 注意：角度指标是 UAV 指向 NFZ 的连线与 UAV 速度的夹角
        E_angle = calc_angle_score(uav.pos, uav.velocity, nfz.pos)
        dist = get_distance(uav.pos, nfz.pos)
        E_dist = calc_dist_score(dist, is_obstacle=True)

        # Eq. (5)
        # 注意：这里 E_angle 越大(越准)，(1-E_a) 越小，突防率越低 -> 撞上了
        # (1-E_d) 越小(离得近，E_d大)，突防率越低
        p_nfz = (1.0 - E_angle) * (1.0 - E_dist)
        p_pen_total *= np.clip(p_nfz, 0.0, 1.0)

    # 2. 遍历拦截者 (Interceptor)
    for inter in interceptor_list:
        E_angle = calc_angle_score(uav.pos, uav.velocity, inter.pos)
        dist = get_distance(uav.pos, inter.pos)
        E_dist = calc_dist_score(dist, is_obstacle=True)

        # 拦截者的速度评价指标 E_speed
        # 论文中 E_speed 是 UAV 与 Interceptor 的速度对比吗？
        # Eq (2) 定义是 v_target，这里应该是 v_interceptor
        inter_speed = getattr(inter, 'velocity', 0.0)  # 假设实体有速度，如果没有默认为0
        if isinstance(inter_speed, np.ndarray):
            inter_speed = np.linalg.norm(inter_speed)

        E_speed = calc_speed_score(uav_speed, inter_speed)

        # Eq. (6)
        term_bracket = cfg.PARAM_C3 * (1.0 - E_dist) + cfg.PARAM_C4 * E_speed
        p_int = (1.0 - E_angle) * term_bracket
        p_pen_total *= np.clip(p_int, 0.0, 1.0)

    return p_pen_total


# --- 综合优势度计算 ---
def calc_advantage(uav, target, nfz_list, interceptor_list):
    """
    计算 UAV 对 Target 的优势度 p_{k,m}
    返回: (p_final, p_damage_only)
    """
    # 1. 纯毁伤概率 (只看 Target)
    p_damage = calc_damage_prob(uav, target)

    # 2. 突防概率 (看环境)
    p_pen = calc_penetration_prob(uav, target, nfz_list, interceptor_list)

    # 3. 最终优势度
    p_final = p_damage * p_pen

    return p_final, p_damage


# --- 状态向量构建 (保持原逻辑，但数值来源变了) ---
def get_state_vector(uav, target, nfz_list, interceptor_list, global_stats=None,
                     prev_joint_p=0.0, prev_revenue=0.0, prev_joint_p_damage_only=0.0):
    """
    严格按照论文 Eq. (15) 构建 14 维状态向量
    """
    # 1. 计算当前 UAV 的 p (实际) 和 p_pure (理想)
    # 这里的 calc_advantage 已经是修改后的版本
    p_km, p_km_damage_only = calc_advantage(uav, target, nfz_list, interceptor_list)

    # 2. 计算分配后的状态
    # 实际联合概率: 1 - (1 - P_old) * (1 - p_km)
    hat_p_m = 1.0 - (1.0 - prev_joint_p) * (1.0 - p_km)

    # 理想联合概率: 1 - (1 - P_old_pure) * (1 - p_km_pure)
    hat_p_m_pure = 1.0 - (1.0 - prev_joint_p_damage_only) * (1.0 - p_km_damage_only)

    hat_G_m = hat_p_m * target.value

    # 3. 计算 Delta 指标
    delta_p_km = p_km_damage_only - p_km
    delta_p_m = hat_p_m_pure - hat_p_m
    delta_G_m = (hat_p_m_pure * target.value) - hat_G_m

    # 提取全局统计
    if global_stats is None:
        chi_c, chi_v, chi_mc = 0.0, 0.0, 0.0
    else:
        chi_c = global_stats.get('cost_ratio', 0.0)
        chi_v = global_stats.get('value_ratio', 0.0)
        chi_mc = global_stats.get('target_cost_ratio', 0.0)

    # 状态向量 (14维) [Eq. 15]
    state = np.array([
        getattr(uav, 'cost', 1.0),
        target.value,
        chi_c,
        chi_v,
        chi_mc,
        p_km,
        prev_joint_p,
        hat_p_m,
        prev_revenue,
        hat_G_m,
        delta_p_km,
        delta_p_m,
        delta_G_m,
        float(uav.available)
    ], dtype=np.float32)

    # 简单的归一化处理 (可选)
    state[0] /= 2.0
    state[1] /= 16.0
    state[8] /= 16.0
    state[9] /= 16.0
    state[12] /= 16.0

    return state