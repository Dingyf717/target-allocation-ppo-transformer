# envs/entities.py
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class Entity:
    id: int
    pos: np.ndarray  # [x, y]


@dataclass
class UAV(Entity):
    # 动态属性
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))  # [vx, vy]
    max_speed: float = 0.0
    load: float = 0.0
    uav_type: int = 1  # 1: Fixed-wing, 2: Rotor

    # 【新增】成本属性 (对应论文 ck^j)
    # Type 1 = 1.0, Type 2 = 1.25
    cost: float = 1.0

    # 状态标记
    assigned_target_id: int = -1  # -1 表示未分配
    available: bool = True

    def reset(self, pos, v, speed, load, cost=1.0):
        self.pos = pos
        self.velocity = v
        self.max_speed = speed
        self.load = load
        self.cost = cost  # 【新增】重置成本
        self.assigned_target_id = -1
        self.available = True


@dataclass
class Target(Entity):
    value: float = 1.0
    defense_level: float = 0.0  # 防御能力
    required_load: float = 0.0  # 需要的打击载荷

    # 状态标记
    locked_by_uavs: List[int] = field(default_factory=list)  # 被哪些 UAV 锁定了

    def reset(self):
        self.locked_by_uavs = []


@dataclass
class NoFlyZone(Entity):
    radius: float = 1.0
    penalty_factor: float = 0.5  # 穿过惩罚系数


@dataclass
class Interceptor(Entity):
    radius: float = 2.0
    kill_prob: float = 0.3  # 拦截概率