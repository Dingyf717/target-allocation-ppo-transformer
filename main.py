# main.py
import numpy as np
import torch
from configs.config import cfg
from envs.entities import UAV, Target, NoFlyZone, Interceptor
from envs.mechanics import get_state_vector
from agents.ppo import PPOAgent


def test_pipeline():
    print("========== 1. 配置加载测试 ==========")
    print(f"地图大小: {cfg.MAP_WIDTH}x{cfg.MAP_HEIGHT} km")
    print(f"PPO设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"网络输入维度: {cfg.STATE_DIM}, 嵌入维度: {cfg.EMBED_DIM}")
    print("配置加载成功！\n")

    print("========== 2. 实体初始化测试 ==========")
    # 随机生成一个 UAV 和一个 Target
    uav = UAV(
        id=0,
        pos=np.array([5.0, 5.0]),
        velocity=np.array([0.1, 0.0]),
        max_speed=0.5,
        load=10.0
    )
    target = Target(
        id=0,
        pos=np.array([15.0, 15.0]),
        value=10.0
    )
    nfz_list = [NoFlyZone(id=0, pos=np.array([10.0, 10.0]), radius=2.0)]
    interceptor_list = []

    print(f"UAV 位置: {uav.pos}")
    print(f"Target 位置: {target.pos}")
    print("实体初始化成功！\n")

    print("========== 3. 物理引擎计算测试 ==========")
    # 测试 mechanics.py 中的核心函数
    try:
        state_vec = get_state_vector(uav, target, nfz_list, interceptor_list)
        print(f"状态向量内容: {state_vec}")
        print(f"状态向量形状: {state_vec.shape}")

        # 维度检查断言
        assert state_vec.shape == (cfg.STATE_DIM,), \
            f"维度错误！期望 ({cfg.STATE_DIM},), 实际 {state_vec.shape}"

        # 检查数值是否包含 NaN
        if np.isnan(state_vec).any():
            print("警告: 状态向量包含 NaN！请检查除零错误。")
        else:
            print("物理计算数值正常。")

    except Exception as e:
        print(f"物理引擎报错: {e}")
        return
    print("物理引擎测试成功！\n")

    print("========== 4. 智能体与网络测试 ==========")
    try:
        # 实例化 PPO Agent
        agent = PPOAgent()
        print("PPO Agent 初始化完成，网络结构如下:")
        print(agent.policy)  # 打印网络结构看是否符合 Transformer 定义

        # 测试动作选择 (Select Action)
        print("\n正在尝试将状态输入网络...")
        action = agent.select_action(state_vec)

        print(f"网络输出动作: {action} (0=不选, 1=选)")
        print(f"Buffer 当前长度: {len(agent.buffer['states'])}")

        # 检查 Buffer 中的 logprob 是否有梯度 (应该没有，因为是 detach 的，或者是 item)
        # 这里只要确认能存进去就行

    except Exception as e:
        print(f"智能体报错: {e}")
        import traceback
        traceback.print_exc()
        return
    print("智能体前向推理成功！\n")

    print("========== 5. 模拟一次更新 (Update) ==========")
    try:
        # 伪造一些数据填满 Buffer 以触发更新逻辑测试
        # 只需要存入几个数据，确保 update() 函数里的 tensor 拼接和运算不报错
        agent.store_transition(reward=1.0, done=False)  # 对应刚才的那次动作

        # 再多存两个，模拟序列数据
        for _ in range(3):
            agent.select_action(state_vec)
            agent.store_transition(reward=0.5, done=False)

        print("正在尝试执行 agent.update()...")
        agent.update()
        print("PPO 更新过程无报错。")

    except Exception as e:
        print(f"更新逻辑报错: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n🎉🎉🎉 恭喜！所有模块联调测试通过！ 🎉🎉🎉")


if __name__ == "__main__":
    test_pipeline()