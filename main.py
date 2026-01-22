import numpy as np
import matplotlib.pyplot as plt
from configs.config import cfg
from envs.uav_env import UAVEnv
from envs.mechanics import (
    calc_angle_score,
    calc_speed_score,
    calc_dist_score,
    calc_damage_prob,
    calc_penetration_prob,
    get_distance
)


def analyze_reward_components(num_rounds=50):
    """
    深度诊断：拆解计算每一个物理评分组件的分布
    """
    print(f"================================================================")
    print(f"🔍 奖励函数全链路诊断 (Rounds: {num_rounds})")
    print(f"📦 参数配置: Zeta_D={cfg.PARAM_ZETA_D}, K={cfg.PARAM_K}")
    print(f"================================================================\n")

    env = UAVEnv()

    # === 1. 数据收集容器 ===
    data = {
        # 原始物理量
        "raw_dist": [],  # 距离 (km)
        "raw_angle": [],  # 偏差角 (deg)
        "raw_speed_r": [],  # 速度比 (v_tgt/v_uav)

        # 归一化评分 (0-1)
        "score_dist": [],  # E_dist
        "score_angle": [],  # E_angle
        "score_speed": [],  # E_speed

        # 最终概率
        "p_dmg": [],  # 基础毁伤概率
        "p_pen": [],  # 突防概率
        "p_final": []  # 最终优势度
    }

    for i in range(num_rounds):
        env.reset(full_reset=True)

        # 遍历所有 UAV-Target 配对，收集“所有可能的攻击方案”数据
        for uav in env.uavs:
            uav_speed = np.linalg.norm(uav.velocity)

            for tgt in env.targets:
                # --- A. 距离组件 ---
                dist = get_distance(uav.pos, tgt.pos)
                e_dist = calc_dist_score(dist, is_obstacle=False)

                # --- B. 角度组件 ---
                # 为了计算原始角度，我们需要手动算一下，mechanics里直接出分了
                vec_u_t = tgt.pos - uav.pos
                vec_v = uav.velocity
                cos_theta = np.dot(vec_u_t, vec_v) / (np.linalg.norm(vec_u_t) * np.linalg.norm(vec_v) + 1e-9)
                angle_deg = np.rad2deg(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

                e_angle = calc_angle_score(uav.pos, uav.velocity, tgt.pos)

                # --- C. 速度组件 ---
                tgt_speed = np.linalg.norm(tgt.velocity) if hasattr(tgt, 'velocity') else 0.015
                e_speed = calc_speed_score(uav_speed, tgt_speed)

                # --- D. 综合概率 ---
                p_d = calc_damage_prob(uav, tgt)
                p_p = calc_penetration_prob(uav, tgt, env.nfz_list, env.interceptors)
                p_f = p_d * p_p

                # 存入数据
                data["raw_dist"].append(dist)
                data["score_dist"].append(e_dist)

                data["raw_angle"].append(angle_deg)
                data["score_angle"].append(e_angle)

                data["raw_speed_r"].append(tgt_speed / (uav_speed + 1e-9))
                data["score_speed"].append(e_speed)

                data["p_dmg"].append(p_d)
                data["p_pen"].append(p_p)
                data["p_final"].append(p_f)

    # === 2. 统计分析与诊断 ===

    def print_stat(name, values, threshold_low=0.2):
        avg = np.mean(values)
        med = np.median(values)
        v_min = np.min(values)
        v_max = np.max(values)

        status = "✅ 正常"
        if avg < threshold_low: status = "⚠️ 过低 (瓶颈)"

        print(f"🔹 {name:<15} | 均值: {avg:.3f} | 中位数: {med:.3f} | 范围: [{v_min:.2f}, {v_max:.2f}] -> {status}")

    print("📊 [1. 距离组件诊断] (Distance Evaluation)")
    print(f"   平均距离: {np.mean(data['raw_dist']):.2f} km")
    print_stat("Score_Dist", data["score_dist"], threshold_low=0.3)
    if np.mean(data["score_dist"]) < 0.3:
        print("   >>> 💡 建议: 距离得分太低！检查 config.py 中的 PARAM_ZETA_D 是否太小？")
        print("              或者检查 mechanics.py 中 D_mid 是否设为了 0 而实际距离很远？")

    print("\n📊 [2. 角度组件诊断] (Angle Evaluation)")
    print(f"   平均偏差角: {np.mean(data['raw_angle']):.2f}°")
    print_stat("Score_Angle", data["score_angle"], threshold_low=0.1)
    if np.mean(data["score_angle"]) < 0.1:
        print("   >>> 💡 建议: 角度得分极低！说明 UAV 初始朝向太乱，或者角度惩罚系数(b)太严苛。")

    print("\n📊 [3. 速度组件诊断] (Speed Evaluation)")
    print(f"   平均速度比(T/U): {np.mean(data['raw_speed_r']):.3f}")
    print_stat("Score_Speed", data["score_speed"], threshold_low=0.5)

    print("\n📊 [4. 最终概率诊断]")
    print_stat("P_Damage (基础)", data["p_dmg"], threshold_low=0.4)
    print_stat("P_Penetration", data["p_pen"], threshold_low=0.5)
    print_stat("P_Final (最终)", data["p_final"], threshold_low=0.2)

    avg_final = np.mean(data["p_final"])
    if avg_final < 0.05:
        print("\n❌ [严重警告] 最终平均成功率接近 0！模型几乎学不到东西。")
        print("   最可能的罪魁祸首是上面的 '⚠️ 过低' 项。请优先修复该物理公式。")
    elif avg_final > 0.3:
        print("\n✅ [环境健康] 物理环境难度适中，适合训练。")
    else:
        print("\n⚠️ [环境偏难] 成功率较低，训练可能需要较长时间或 Curriculum Learning。")

    # === 3. (可选) 画图 ===
    # plt.hist(data["score_dist"], bins=20, alpha=0.7, label="Dist Score")
    # plt.hist(data["score_angle"], bins=20, alpha=0.7, label="Angle Score")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    analyze_reward_components()