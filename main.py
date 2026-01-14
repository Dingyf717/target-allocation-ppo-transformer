import numpy as np
import pandas as pd
from envs.entities import UAV, Target, NoFlyZone, Interceptor
from envs.mechanics import calc_dist_score, calc_angle_score, calc_damage_prob, calc_penetration_prob, calc_advantage
from configs.config import cfg


def run_diagnostics():
    print("=================================================================")
    print("              UAV Swarm 物理模型诊断工具 (Diagnostics)           ")
    print("=================================================================")
    print(f"当前配置参数检测:")
    print(f"  - PARAM_ZETA_D (Config): {cfg.PARAM_ZETA_D} (注意：障碍物应在 mechanics.py 中硬编码为 25.0)")
    print(f"  - PARAM_K: {cfg.PARAM_K}")
    print("=================================================================\n")

    # ================= [测试场景 1]：最佳攻击距离验证 (D_mid) =================
    print(">>> [Test 1] 攻击距离评分验证 (Target Distance Score)")
    print("    * 论文设定: D_mid ≈ 155km 时分数最高 (接近1.0)")
    print("    * 如果 mechanics.py 未修改 (D_mid=0)，则距离越近分越高")

    # 构建一个静止 UAV 和不同距离的目标
    uav = UAV(id=0, pos=np.array([0, 0]), velocity=np.array([100, 0]), max_speed=100, load=1.0)

    test_dists = [10, 50, 100, 150, 155, 160, 200]
    results_1 = []

    for d in test_dists:
        # 目标在 x 轴正方向 d 处
        tgt = Target(id=0, pos=np.array([float(d), 0]), value=10.0)

        # 只计算距离分数 (假设 mechanics.py 里 calc_dist_score 第二个参数 False 代表目标)
        dist_score = calc_dist_score(d, is_obstacle=False)

        results_1.append({
            "Distance (km)": d,
            "Dist Score (0-1)": f"{dist_score:.4f}",
            "评价": "Excellent" if dist_score > 0.9 else "Low"
        })

    print(pd.DataFrame(results_1))
    print("\n-----------------------------------------------------------------\n")

    # ================= [测试场景 2]：障碍物威胁范围验证 (Zeta) =================
    print(">>> [Test 2] 障碍物威胁范围验证 (Obstacle Threat)")
    print("    * 论文设定: Zeta=25.0。")
    print("    * 预期: 距离 25km 时分数约 0.36; 距离 50km (2*Zeta) 时分数应 < 0.02 (安全)。")
    print("    * 如果未修复 (Zeta=150)，50km 处分数仍高达 0.89 (极度危险)。")

    test_obs_dists = [10, 25, 50, 75, 100, 150]
    results_2 = []

    for d in test_obs_dists:
        # 模拟一个距离 d 的障碍物
        dist_score_obs = calc_dist_score(d, is_obstacle=True)

        status = "BUG (Zeta过大)" if (d == 50 and dist_score_obs > 0.1) else "Normal"
        if d == 25 and 0.3 < dist_score_obs < 0.4: status = "Paper Match (Zeta≈25)"

        results_2.append({
            "Dist to Obs (km)": d,
            "Threat Score (E_d)": f"{dist_score_obs:.4f}",
            "Penetration Limit (1-E_d)": f"{1.0 - dist_score_obs:.4f}",
            "诊断": status
        })

    print(pd.DataFrame(results_2))
    print("\n-----------------------------------------------------------------\n")

    # ================= [测试场景 3]：综合突防概率 (正对撞击 vs 侧向规避) =================
    print(">>> [Test 3] 综合突防概率验证 (Penetration Probability)")
    print("    * 场景: UAV 向右飞 (Vel=[1,0])，正前方 50km 处有一个 NFZ。")

    uav_pos = np.array([0.0, 0.0])
    uav_vel = np.array([300.0, 0.0])  # 向右飞
    uav = UAV(0, uav_pos, uav_vel, 300.0, 1.0)

    target = Target(0, np.array([200.0, 0.0]), 10.0)  # 目标在正前方很远

    # Case A: NFZ 在正前方 (必死)
    nfz_center = NoFlyZone(0, np.array([50.0, 0.0]), radius=10.0)

    # Case B: NFZ 在侧方 50km (安全)
    nfz_side = NoFlyZone(1, np.array([50.0, 50.0]), radius=10.0)

    # 计算 A
    print("3.1 [高危场景] NFZ 在正前方 50km:")
    p_pen_a = calc_penetration_prob(uav, target, [nfz_center], [])
    ang_score_a = calc_angle_score(uav.pos, uav.velocity, nfz_center.pos)
    dist_score_a = calc_dist_score(50.0, is_obstacle=True)

    print(f"   - Angle Score (1.0=正对): {ang_score_a:.4f}")
    print(f"   - Dist Score  (Threat):   {dist_score_a:.4f}")
    print(f"   - Penetration Prob:       {p_pen_a:.4f}  <-- 预期应接近 0.0")

    # 计算 B
    print("\n3.2 [安全场景] NFZ 在侧方 50km (直线距离70km):")
    p_pen_b = calc_penetration_prob(uav, target, [nfz_side], [])
    dist_b = np.linalg.norm(uav.pos - nfz_side.pos)
    ang_score_b = calc_angle_score(uav.pos, uav.velocity, nfz_side.pos)
    dist_score_b = calc_dist_score(dist_b, is_obstacle=True)

    print(f"   - Real Dist:              {dist_b:.1f} km")
    print(f"   - Angle Score (1.0=正对): {ang_score_b:.4f} (侧向应很小)")
    print(f"   - Dist Score:             {dist_score_b:.4f}")
    print(f"   - Penetration Prob:       {p_pen_b:.4f}  <-- 预期应接近 1.0")

    print("\n-----------------------------------------------------------------\n")
    print("诊断结束。请根据上述表格确认 mechanics.py 是否修改正确。")


if __name__ == "__main__":
    run_diagnostics()