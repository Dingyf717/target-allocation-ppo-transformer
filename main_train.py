# main_train.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from configs.config import cfg
from envs.uav_env import UAVEnv
from agents.ppo import PPOAgent


def save_learning_curve(rewards, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title("Learning Curve (Average Reward)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def train():
    print("============================================================================================")
    print(f"开始训练 PPO-Transformer | 地图: {cfg.MAP_WIDTH}x{cfg.MAP_HEIGHT} | UAVs: {cfg.NUM_UAVS}")
    print("============================================================================================")

    # 1. 初始化环境和智能体
    env = UAVEnv()
    agent = PPOAgent()

    # 2. 实验记录设置
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/{time_str}"
    os.makedirs(log_dir, exist_ok=True)
    model_dir = f"./saved_models/{time_str}"
    os.makedirs(model_dir, exist_ok=True)

    # 统计变量
    ep_rewards = []  # 记录每回合总奖励
    avg_rewards = []  # 记录平均奖励（用于画图）
    best_reward = -np.inf  # 记录最高分

    # ================= 3. 主训练循环 (Algorithm 1) =================
    for i_episode in range(1, cfg.MAX_EPISODES + 1):

        # 【核心修正】 Algorithm 1 Line 6-7
        # 每 200 episodes 重置一次环境布局 (E, U, T)
        # 其余时间保持布局不变，只重置决策状态
        if i_episode == 1 or i_episode % 200 == 0:
            # 假设你在 uav_env.reset() 中增加了一个参数 reset_layout
            # 或者 env.reset() 默认只重置状态，env.reset_layout() 重置布局
            # 这里演示最通用的写法：
            # 如果 uav_env.reset() 每次都随机，你需要修改 uav_env.py
            # 建议：修改 uav_env.reset(full_reset=False)
            state = env.reset(full_reset=True)
        else:
            state = env.reset(full_reset=False)
        current_ep_reward = 0
        done = False

        # --- 一个回合 (Episode) ---
        while not done:
            # a. 智能体选择动作
            action = agent.select_action(state)

            # b. 环境执行动作
            next_state, reward, done, _ = env.step(action)

            # c. 存储 Transition
            agent.store_transition(reward, done)

            # d. 更新状态 & 累计奖励
            state = next_state
            current_ep_reward += reward

            # e. 如果 Buffer 满了或者回合结束，触发 PPO 更新
            # (这里简化处理：PPO通常是攒够一定步数更新，或者是回合结束更新)
            # 我们选择：每回合结束后更新，或者 Buffer 超过 2000 步更新
            if len(agent.buffer['states']) >= cfg.BATCH_SIZE * 4:
                agent.update()

        # 回合结束，强制更新剩余数据 (保证 On-Policy)
        agent.update()

        # --- 记录与日志 ---
        ep_rewards.append(current_ep_reward)
        avg_reward = np.mean(ep_rewards[-50:])  # 计算最近50回合的滑动平均
        avg_rewards.append(avg_reward)

        # 打印进度
        if i_episode % 10 == 0:
            print(f"Episode: {i_episode:4d} | Reward: {current_ep_reward:8.2f} | Avg Reward: {avg_reward:8.2f}")

        # 保存最佳模型
        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save(agent.policy.state_dict(), f"{model_dir}/best_model.pth")
            print(f"   >>> 新的最佳模型已保存! Avg Reward: {best_reward:.2f}")

        # 定期保存曲线图
        if i_episode % 100 == 0:
            save_learning_curve(avg_rewards, f"{log_dir}/learning_curve.png")

    print("============================================================================================")
    print("训练结束！")
    print(f"模型已保存至: {model_dir}")
    print(f"日志已保存至: {log_dir}")


if __name__ == "__main__":
    train()