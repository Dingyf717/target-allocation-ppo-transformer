# test_visualize.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from configs.config import cfg
from envs.uav_env import UAVEnv
from agents.ppo import PPOAgent

# 设置 Matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def visualize_decision(env, agent, model_path):
    print(f"正在加载模型: {model_path} ...")

    # 显式设置 weights_only=True 以消除警告
    # map_location='cpu' 确保兼容性
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)

    agent.policy.load_state_dict(checkpoint)
    agent.policy.eval()

    state = env.reset()
    done = False
    assignments = []

    print("开始推理决策...")
    print(f"{'决策动作':<30} | {'结果':<8} | {'全队总分 J(X)':<15} | {'本步奖励 Reward'}")
    print("-" * 80)

    while not done:
        # 1. 在 Step 之前获取当前正在做决策的 UAV 和 Target ID
        # 因为 Step 之后索引会跳到下一个，所以必须先取
        u_id = env.uavs[env.uav_idx].id
        t_id = env.targets[env.target_idx].id

        # 2. 神经网络决策
        action = agent.select_action(state)

        # 3. 执行环境交互 (获取 info 以读取分数)
        next_state, reward, done, info = env.step(action)

        # 4. 打印决策和分数
        if action == 1:
            # 检查环境是否接受了该分配 (info 中 'is_valid_action' 字段)
            # 如果 main_train.py 的逻辑是 "new_r < prev_r 则拒绝"，这里可以体现出来
            is_valid = info.get('is_valid_action', True)
            status = "✅ 成功" if is_valid else "❌ 拒绝"

            # 只有成功的分配才记录到绘图列表
            if is_valid:
                assignments.append((u_id, t_id))

            # 获取当前总分
            current_j = info.get('J_val', 0.0)

            print(f"UAV-{u_id} 尝试锁定 -> Target-{t_id}    | {status} | {current_j:15.4f} | {reward:+.4f}")

        # 更新状态
        state = next_state

    # 最终结果打印
    final_j = info.get('J_val', 0.0) if 'info' in locals() else 0.0
    print("-" * 80)
    print(f"决策结束。最终全队总分 J(X): {final_j:.4f}")
    print(f"共生成 {len(assignments)} 个有效攻击对。")

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
        # 画边界
        circle_border = plt.Circle(nfz.pos, nfz.radius, color='black', fill=False, linestyle='--')
        ax.add_patch(circle_border)

    # 3. 画拦截者 (红色X) - 如果有的话
    for inter in env.interceptors:
        plt.scatter(inter.pos[0], inter.pos[1], c='red', marker='x', s=100, linewidths=2, label='Interceptor')
        circle = plt.Circle(inter.pos, inter.radius, color='red', alpha=0.1)
        ax.add_patch(circle)

    # 4. 画目标 (根据价值画不同大小的五角星)
    for tgt in env.targets:
        size = tgt.value * 30
        plt.scatter(tgt.pos[0], tgt.pos[1], c='orange', marker='*', s=size, edgecolors='black',
                    label='Target' if tgt.id == 0 else "")
        plt.text(tgt.pos[0], tgt.pos[1] + 2, f"T{tgt.id}\n{tgt.value:.0f}", fontsize=9, ha='center')

    # 5. 画无人机 (蓝色三角)
    for uav in env.uavs:
        plt.scatter(uav.pos[0], uav.pos[1], c='blue', marker='^', s=80, label='UAV' if uav.id == 0 else "")
        plt.text(uav.pos[0], uav.pos[1] - 5, f"U{uav.id}", fontsize=9, ha='center', color='blue')

    # 6. 画连线 (分配关系)
    for (u_id, t_id) in assignments:
        u_pos = env.uavs[u_id].pos
        t_pos = env.targets[t_id].pos
        plt.plot([u_pos[0], t_pos[0]], [u_pos[1], t_pos[1]], 'k--', alpha=0.6)

    # 去重图例
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.grid(True, linestyle=':', alpha=0.6)

    # 保存图片
    save_path = "decision_vis_result.png"
    plt.savefig(save_path, dpi=150)
    print(f"可视化结果已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    # 自动寻找最新的模型
    model_dir = "./saved_models"
    if os.path.exists(model_dir):
        all_subdirs = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if
                       os.path.isdir(os.path.join(model_dir, d))]
        if all_subdirs:
            latest_subdir = max(all_subdirs, key=os.path.getmtime)

            # 这里可以修改为你想要测试的具体模型文件名
            # model_path = os.path.join(latest_subdir, "best_model.pth")
            model_path = os.path.join(latest_subdir, "checkpoint_ep600.pth")  # 示例

            # 如果找不到指定文件，回退去找 best_model
            if not os.path.exists(model_path):
                print(f"提示: {model_path} 不存在，尝试加载 best_model.pth")
                model_path = os.path.join(latest_subdir, "best_model.pth")

            if not os.path.exists(model_path):
                print(f"错误: 找不到模型文件 {model_path}")
            else:
                env = UAVEnv()
                agent = PPOAgent()
                visualize_decision(env, agent, model_path)
        else:
            print(f"错误: {model_dir} 下没有子文件夹")
    else:
        print(f"错误: {model_dir} 文件夹不存在")