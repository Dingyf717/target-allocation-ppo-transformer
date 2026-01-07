# test_visualize.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from configs.config import cfg
from envs.uav_env import UAVEnv
from agents.ppo import PPOAgent

# 设置 Matplotlib 支持中文 (可选，如果乱码可去掉)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def visualize_decision(env, agent, model_path):
    print(f"正在加载模型: {model_path} ...")

    # 加载模型参数
    checkpoint = torch.load(model_path)
    agent.policy.load_state_dict(checkpoint)
    agent.policy.eval()  # 切换到评估模式

    state = env.reset()
    done = False

    # 记录决策结果: [(uav_id, target_id), ...]
    assignments = []

    print("开始推理决策...")
    while not done:
        # 获取当前指针对应的 UAV 和 Target ID
        u_id = env.uavs[env.uav_idx].id
        t_id = env.targets[env.target_idx].id

        # 神经网络决策
        action = agent.select_action(state)

        # 记录分配动作
        if action == 1:
            assignments.append((u_id, t_id))
            print(f"  [决策] UAV-{u_id} 锁定 -> Target-{t_id}")

        state, _, done, _ = env.step(action)

    print(f"决策结束，共生成 {len(assignments)} 个攻击对。")
    plot_results(env, assignments)


def plot_results(env, assignments):
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # 1. 画地图边界
    plt.xlim(0, cfg.MAP_WIDTH)
    plt.ylim(0, cfg.MAP_HEIGHT)
    plt.title(f"UAV Swarm Decision Visualization (Map: {cfg.MAP_WIDTH:.0f}x{cfg.MAP_HEIGHT:.0f})")

    # 2. 画禁飞区 (灰色圆圈)
    for nfz in env.nfz_list:
        circle = plt.Circle(nfz.pos, nfz.radius, color='gray', alpha=0.3, label='NFZ')
        ax.add_patch(circle)
        # 画个边界
        circle_border = plt.Circle(nfz.pos, nfz.radius, color='black', fill=False, linestyle='--')
        ax.add_patch(circle_border)

    # 3. 画拦截者 (红色X)
    for inter in env.interceptors:
        plt.scatter(inter.pos[0], inter.pos[1], c='red', marker='x', s=100, linewidths=2, label='Interceptor')
        # 画拦截范围
        circle = plt.Circle(inter.pos, inter.radius, color='red', alpha=0.1)
        ax.add_patch(circle)

    # 4. 画目标 (根据价值画不同大小的五角星)
    # 颜色映射: 价值越高越红
    for tgt in env.targets:
        size = tgt.value * 30  # 大小随价值变化
        plt.scatter(tgt.pos[0], tgt.pos[1], c='orange', marker='*', s=size, edgecolors='black',
                    label='Target' if tgt.id == 0 else "")
        plt.text(tgt.pos[0], tgt.pos[1] + 2, f"T{tgt.id}\n{tgt.value:.1f}", fontsize=9, ha='center')

    # 5. 画无人机 (蓝色三角)
    for uav in env.uavs:
        plt.scatter(uav.pos[0], uav.pos[1], c='blue', marker='^', s=80, label='UAV' if uav.id == 0 else "")
        plt.text(uav.pos[0], uav.pos[1] - 5, f"U{uav.id}", fontsize=9, ha='center', color='blue')

    # 6. 画连线 (分配关系)
    for (u_id, t_id) in assignments:
        u_pos = env.uavs[u_id].pos
        t_pos = env.targets[t_id].pos

        # 画虚线
        plt.plot([u_pos[0], t_pos[0]], [u_pos[1], t_pos[1]], 'k--', alpha=0.6)

    # 去重图例
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.grid(True, linestyle=':', alpha=0.6)
    save_path = "decision_vis0.png"
    plt.savefig(save_path, dpi=150)
    print(f"可视化结果已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    # 自动寻找最新的模型
    model_dir = "./saved_models"
    # 找到最近创建的文件夹
    all_subdirs = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if
                   os.path.isdir(os.path.join(model_dir, d))]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    model_path = os.path.join(latest_subdir, "best_model.pth")

    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
    else:
        env = UAVEnv()
        agent = PPOAgent()
        visualize_decision(env, agent, model_path)