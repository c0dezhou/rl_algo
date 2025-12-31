from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any, Dict

import torch


@dataclass
class Config:
    # 最简单的配置基类：
    # - 每个算法都有一个 Config dataclass，字段就是超参数（带默认值）
    # - 从 yaml 读到 dict 后：只覆盖同名字段；多出来的字段也允许（临时参数直接挂在对象上）

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        data = data or {}
        if not is_dataclass(cls):
            return cls(**data)

        field_names = {f.name for f in fields(cls)}
        obj = cls(**{k: v for k, v in data.items() if k in field_names})

        # 兼容 yaml 里多写的字段：不报错，直接挂到对象上
        for k, v in data.items():
            if k not in field_names:
                setattr(obj, k, v)
        return obj

    def to_dict(self):
        if is_dataclass(self):
            data = asdict(self)
        else:
            data = dict(getattr(self, "__dict__", {}))

        # 把“额外字段”也一并写出去，保证 config 快照可复现
        for k, v in getattr(self, "__dict__", {}).items():
            if k not in data:
                data[k] = v
        return data


@dataclass
class Batch:
    # DQN 从 replay buffer 里采样出来的一批数据
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor


Transition = Dict[str, Any]
