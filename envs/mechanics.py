# envs/mechanics.py
import numpy as np
from configs.config import cfg


def get_distance(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)


# --- 论文公式 (1)-(2): 角度优势 ---
def calc_angle_score(uav_pos, uav_vel, target_pos):
    vec_u_t = target_pos - uav_pos
    dist = np.linalg.norm(vec_u_t) + 1e-6
    vec_u_t /= dist

    speed = np.linalg.norm(uav_vel) + 1e-6
    vec_v = uav_vel / speed

    cos_theta = np.dot(vec_u_t, vec_v)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    score = np.exp(-cfg.PARAM_K * np.abs(theta))
    return score


# --- 论文公式 (4): 毁伤概率 ---
def calc_damage_prob(uav_pos, target_pos):
    d_km = get_distance(uav_pos, target_pos)
    prob = 1.0 / (1.0 + cfg.PARAM_C1 * d_km)
    return prob


# --- 论文公式 (5)-(6): 突防概率 ---
def calc_penetration_prob(uav_pos, target_pos, nfz_list, interceptor_list):
    prob_survive = 1.0

    # 禁飞区近似检测
    num_samples = 5
    for t in np.linspace(0, 1, num_samples):
        sample_point = uav_pos + t * (target_pos - uav_pos)
        for nfz in nfz_list:
            if get_distance(sample_point, nfz.pos) < nfz.radius:
                prob_survive *= (1.0 - nfz.penalty_factor)

    # 拦截区检测
    for inter in interceptor_list:
        if get_distance(target_pos, inter.pos) < inter.radius:
            if get_distance(uav_pos, inter.pos) < inter.radius:
                prob_survive *= (1.0 - inter.kill_prob)

    return prob_survive


# --- 核心: 生成状态向量 (State Generation) ---
# 【修正】新增 global_stats 参数
def get_state_vector(uav, target, nfz_list, interceptor_list, global_stats=None):
    # 基础计算
    dist = get_distance(uav.pos, target.pos)
    angle_score = calc_angle_score(uav.pos, uav.velocity, target.pos)
    damage_prob = calc_damage_prob(uav.pos, target.pos)
    pen_prob = calc_penetration_prob(uav.pos, target.pos, nfz_list, interceptor_list)

    # 提取全局统计量 (如果没有传入，默认为0)
    if global_stats is None:
        cost_ratio = 0.0
        value_ratio = 0.0
    else:
        cost_ratio = global_stats.get('cost_ratio', 0.0)
        value_ratio = global_stats.get('value_ratio', 0.0)

    # 状态拼接 (14维)
    state = np.array([
        uav.pos[0] / cfg.MAP_WIDTH,  # 1. UAV x
        uav.pos[1] / cfg.MAP_HEIGHT,  # 2. UAV y
        target.pos[0] / cfg.MAP_WIDTH,  # 3. Target x
        target.pos[1] / cfg.MAP_HEIGHT,  # 4. Target y

        # 【关键修改】替换掉原本无用的 Vx, Vy，改为全局统计特征
        cost_ratio,  # 5. Global Cost Ratio (已消耗成本 / 总成本)
        value_ratio,  # 6. Global Value Ratio (已覆盖价值 / 总价值)

        uav.load / 10.0,  # 7. Load
        target.value / 10.0,  # 8. Value (这里顺便做归一化)
        dist / cfg.MAP_WIDTH,  # 9. Distance
        angle_score,  # 10. Angle Score
        damage_prob,  # 11. Kill Prob
        pen_prob,  # 12. Penetration Prob
        len(target.locked_by_uavs),  # 13. Current Attackers Count (Target Cost Ratio Proxy)
        float(uav.available)  # 14. Is Available
    ], dtype=np.float32)

    return state