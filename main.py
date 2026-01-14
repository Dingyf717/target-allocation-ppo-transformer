import numpy as np
import matplotlib.pyplot as plt
from configs.config import cfg
from envs.uav_env import UAVEnv
# 直接导入物理计算核心函数，确保算法一致
from envs.mechanics import calc_advantage


def analyze_environment_difficulty(num_rounds=10):
    """
    模拟 main_train.py 中的环境生成过程，并进行物理属性体检。
    :param num_rounds: 测试生成的环境数量 (模拟多次 reset)
    """
    print(f"================================================================")
    print(f"环境体检工具 | 地图: {cfg.MAP_WIDTH}x{cfg.MAP_HEIGHT} | UAVs: {cfg.NUM_UAVS} | Targets: {cfg.NUM_TARGETS}")
    print(f"检查项: 基础毁伤概率(P_Dmg) & 最终突防成功率(P_Final)")
    print(f"================================================================\n")

    env = UAVEnv()

    # 统计容器
    global_avg_p_dmg = []
    global_avg_p_final = []

    best_match_p_dmg = []
    best_match_p_final = []

    for i in range(num_rounds):
        # 1. 强制完全重置，生成新的随机场景 (UAV/Target/拦截器位置)
        # 这模拟了 main_train.py 中 if i_episode % 200 == 0 的逻辑
        env.reset(full_reset=True)

        # 收集当前场景下所有可能的配对数据
        # Shape: [Num_UAVs, Num_Targets]
        p_dmg_matrix = np.zeros((len(env.uavs), len(env.targets)))
        p_final_matrix = np.zeros((len(env.uavs), len(env.targets)))

        for u_idx, uav in enumerate(env.uavs):
            for t_idx, target in enumerate(env.targets):
                # 调用核心物理引擎计算
                # calc_advantage 返回 (p_final, p_damage_only)
                p_fin, p_dmg = calc_advantage(uav, target, env.nfz_list, env.interceptors)

                p_dmg_matrix[u_idx, t_idx] = p_dmg
                p_final_matrix[u_idx, t_idx] = p_fin

        # --- 指标 1: 环境整体“友好度” (所有可能配对的平均值) ---
        # 如果这个值极低，说明大部分飞机对大部分目标都无效（距离太远或太弱）
        avg_dmg = np.mean(p_dmg_matrix)
        avg_final = np.mean(p_final_matrix)

        global_avg_p_dmg.append(avg_dmg)
        global_avg_p_final.append(avg_final)

        # --- 指标 2: 任务“可解性” (对于每个目标，选最好的飞机去打) ---
        # 这模拟了一个“完美策略”能达到的上限。如果这个值都很低，说明任务在物理上是不可能完成的。
        # 对每一列(Target)取最大值，然后求这些最大值的平均
        best_uav_for_targets_dmg = np.max(p_dmg_matrix, axis=0)  # 每个目标能受到的最大基础伤害
        best_uav_for_targets_final = np.max(p_final_matrix, axis=0)  # 每个目标能获得的最大最终概率

        scene_solvability_dmg = np.mean(best_uav_for_targets_dmg)
        scene_solvability_final = np.mean(best_uav_for_targets_final)

        best_match_p_dmg.append(scene_solvability_dmg)
        best_match_p_final.append(scene_solvability_final)

        print(f"[Round {i + 1:02d}] Scene Analysis:")
        print(f"  > 随机配对均值: P_Dmg={avg_dmg:.3f}, P_Final={avg_final:.3f}")
        print(f"  > 最优匹配均值: P_Dmg={scene_solvability_dmg:.3f}, P_Final={scene_solvability_final:.3f}")

        # 简单的突防率估算 (Penetration Rate)
        pen_rate = scene_solvability_final / (scene_solvability_dmg + 1e-6)
        print(f"  > 估计突防率 (Final/Dmg): {pen_rate:.1%}")

        # 警告逻辑
        if scene_solvability_final < 0.2:
            print(f"  [!!!] 警告: 此场景极难! 即使最优分配，平均成功率也只有 {scene_solvability_final:.1%}")
        print("-" * 60)

    # --- 最终汇总 ---
    print("\n========================= 最终诊断报告 =========================")
    print(f"测试轮数: {num_rounds}")
    print(f"1. 基础物理属性 (P_Damage):")
    print(f"   - 理论上限 (最优匹配): {np.mean(best_match_p_dmg):.3f} (如果此值低，说明UAV载荷不够或距离太远)")
    print(f"   - 环境均值 (随机匹配): {np.mean(global_avg_p_dmg):.3f}")

    print(f"\n2. 最终成功概率 (P_Final, 含拦截/禁飞区):")
    print(f"   - 理论上限 (最优匹配): {np.mean(best_match_p_final):.3f} (这是Agent能学到的天花板)")
    print(f"   - 环境均值 (随机匹配): {np.mean(global_avg_p_final):.3f}")

    avg_pen = np.mean(best_match_p_final) / (np.mean(best_match_p_dmg) + 1e-6)
    print(f"\n3. 综合突防率 (Penetration): {avg_pen:.1%}")

    if avg_pen < 0.5:
        print("\n[诊断建议]: 突防率过低 (<50%)。拦截器或禁飞区可能过于密集/强大。")
        print("           建议: 减少 interceptors 数量，或减小禁飞区半径，或增加 UAV 速度。")
    elif np.mean(best_match_p_dmg) < 0.5:
        print("\n[诊断建议]: 基础毁伤率过低。天气折损可能太高，或 Target 距离 UAV 生成点太远。")
        print("           建议: 检查 Config 中的 WEATHER_FACTOR 或 UAV_GEN_X_RANGE。")
    else:
        print("\n[诊断建议]: 环境参数看起来很健康，理论上有解。如果训练效果差，请检查 Reward 设置或 PPO 超参数。")

    print("================================================================")


if __name__ == "__main__":
    analyze_environment_difficulty(num_rounds=10)