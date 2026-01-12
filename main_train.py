# main_train.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from configs.config import cfg
from envs.uav_env import UAVEnv
from agents.ppo import PPOAgent


def save_learning_curve(rewards, q0s, save_path):
    plt.figure(figsize=(12, 6))

    # 绘制 Reward 曲线 (蓝色)
    plt.plot(rewards, label='Average Reward', color='tab:blue', linewidth=1.5)

    # 绘制 Q0 曲线 (橙色)
    # 只有当 q0s 不为空时才画（兼容性处理）
    if q0s is not None and len(q0s) > 0:
        plt.plot(q0s, label='Episode Q0 (Value Est.)', color='tab:orange', linewidth=1.5, linestyle='--')

    plt.title("Learning Curve: Reward vs Q0")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.legend(loc='lower right')  # 图例放在右下角，避免遮挡
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
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

    # 【新增】Q0 统计变量
    ep_q0s = []  # 记录每回合的 Q0
    avg_q0s = []  # 记录平均 Q0 (用于画图)

    best_reward = -np.inf  # 记录最高分

    # ================= 3. 主训练循环 (Algorithm 1) =================
    for i_episode in range(1, cfg.MAX_EPISODES + 1):

        state = env.reset(full_reset=(i_episode == 1 or i_episode % 200 == 0))  # 结合之前的 Curriculum Learning 修改
        # 每一回合都完全重置环境，生成新的 UAV 和 Target 位置
        # state = env.reset(full_reset=True)
        current_ep_reward = 0
        done = False

        # --- 【核心修改】获取当前回合初始状态的 Value (Q0) ---
        # 此时 state 是初始状态 S_0
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)

        with torch.no_grad():
            # 直接调用 old_policy 获取 value。
            # 根据 transformer_net.py, get_action 返回: action, log_prob, value, entropy
            # 我们只需要第三个返回值 value
            _, _, q0_val, _ = agent.policy_old.get_action(state_tensor)

            # 提取标量值
            current_q0 = q0_val.item()

        # --- 一个回合 (Episode) ---
        while not done:
            # a. select action
            action = agent.select_action(state)
            # b. step
            next_state, reward, done, _ = env.step(action)
            # c. store
            agent.store_transition(reward, done)
            # d. update state
            state = next_state
            current_ep_reward += reward

            # --- 回合结束后的处理 ---

        # 【关键修改】只有当 Buffer 里的数据积累得足够多时，才触发更新
        # 这样保证了 Buffer 里包含的永远是“完整的 Episode”，不会出现截断
        # 建议阈值设为 Batch_Size 的 4~8 倍（例如 256~512 步）
        # 这样如果是短 Episode (30步) 会攒好几个才更新；如果是长 Episode (300步) 每次都会更新
        if len(agent.buffer['states']) >= cfg.BATCH_SIZE * 4:
            agent.update()

        # --- 记录与日志 ---
        ep_rewards.append(current_ep_reward)
        avg_reward = np.mean(ep_rewards[-50:])  # 计算最近50回合的滑动平均
        avg_rewards.append(avg_reward)

        # 2. 【新增】记录 Q0
        ep_q0s.append(current_q0)
        # 为了曲线平滑，我们也取最近 50 个的平均值，或者直接画原始值(论文看似有波动，可能是原始值)
        # 这里建议先做平滑，画出来好看；如果想看原始波动，就直接 avg_q0 = current_q0
        avg_q0 = np.mean(ep_q0s[-50:])
        avg_q0s.append(avg_q0)

        # 3. 打印进度 (增加 Q0 显示)
        if i_episode % 10 == 0:
            print(f"Episode: {i_episode:4d} | Reward: {current_ep_reward:8.2f} | Avg R: {avg_reward:8.2f} | Q0: {avg_q0:6.2f}")

        # 保存最佳模型
        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save(agent.policy.state_dict(), f"{model_dir}/best_model.pth")
            print(f"   >>> 新的最佳模型已保存! Avg Reward: {best_reward:.2f}")

        # 4. 【修改】定期保存曲线图 (传入两个列表)
        if i_episode % 100 == 0:
            save_learning_curve(avg_rewards, avg_q0s, f"{log_dir}/learning_curve.png")

        # # [新增] 在每个 Episode 结束时，更新学习率调度器
        # agent.update_lr()
        #
        # # ... (日志打印代码) ...
        # if i_episode % 10 == 0:
        #     # 你甚至可以在这里把当前 LR 打印出来监控
        #     curr_lr = agent.optimizer.param_groups[0]['lr']
        #     print(f"Ep: {i_episode} | Reward: {current_ep_reward:.2f} | LR: {curr_lr:.2e}")

    print("============================================================================================")
    print("训练结束！")
    print(f"模型已保存至: {model_dir}")
    print(f"日志已保存至: {log_dir}")


if __name__ == "__main__":
    train()