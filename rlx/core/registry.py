# rlx/core/registry.py
"""
该模块实现了强化学习 (RL) 算法的注册表 (Registry) 和工厂 (Factory) 模式。

这种设计模式是框架可扩展性的基石。它将算法的“注册”过程与“创建”过程解耦,
使得添加新算法变得非常简单, 无需修改任何核心训练代码。

核心功能:
1.  **算法注册**: 开发者可以使用 `@registry.register_agent` 装饰器轻松地将新的
    RL 算法 (Agent) 添加到框架中。注册时, 算法会获得一个唯一的字符串名称,
    例如 "qlearning" 或 "ppo"。

2.  **配置关联**: 在注册算法的同时, 必须关联一个 Pydantic 配置类。这使得算法的
    超参数 (hyperparameters) 结构化、类型安全且易于管理。框架可以自动处理
    配置的加载、验证和覆盖。

3.  **动态实例化**: 提供一个 `create_agent` 工厂方法, 它能根据算法名称和用户提供
    的配置动态地创建和初始化一个智能体实例。这使得训练脚本 (`train.py`) 可以
    保持通用, 无需硬编码任何特定的算法类。

主要组件:
- `Registry` 类: 一个封装了所有注册和创建逻辑的中心类。
- `registry` 实例: `Registry` 类的一个全局单例, 供整个项目使用。
"""
from typing import Any, Callable, Dict, Type, Optional

import gymnasium as gym
import torch
from pydantic import BaseModel

from rlx.core.base_agent import BaseAgent

class Registry:
    """
    一个中央注册表, 用于管理所有可用的 RL 算法及其配置。
    """
    def __init__(self):
        """初始化两个字典来存储算法和配置的映射关系。"""
        self.agent_registry: Dict[str, Type[BaseAgent]] = {}
        self.config_registry: Dict[str, Type[Any]] = {}

    def register_agent(self, name: str, config_cls: Type[Any]) -> Callable[[Type[BaseAgent]], Type[BaseAgent]]:
        """
        一个装饰器, 用于将一个 Agent 类及其配置类注册到全局注册表中。

        使用示例:
        ```python
        @registry.register_agent("my_algo", MyAlgoConfig)
        class MyAlgoAgent(BaseAgent):
            ...
        ```

        Args:
            name (str): 算法的唯一名称。
            config_cls (Type[Any]): 与该算法关联的 Pydantic 配置类。

        Returns:
            Callable: 返回一个接受 Agent 类并将其注册的装饰器函数。
        """
        def decorator(agent_cls: Type[BaseAgent]) -> Type[BaseAgent]:
            if name in self.agent_registry:
                raise ValueError(f"名称为 '{name}' 的智能体已被注册。")
            
            self.agent_registry[name] = agent_cls
            self.config_registry[name] = config_cls
            
            return agent_cls
        return decorator

    def create_agent(
        self,
        name: str,
        obs_space: gym.Space,
        act_space: gym.Space,
        config: BaseModel,
        device: torch.device,
    ) -> BaseAgent:
        """
        一个工厂方法, 用于根据名称、环境信息和配置来创建并初始化一个 Agent 实例。

        Args:
            name (str): 要创建的算法的名称。
            obs_space (gym.Space): 环境的观测空间。
            act_space (gym.Space): 环境的动作空间。
            config (BaseModel): 包含所有超参数的 Pydantic 配置实例。
            device (torch.device): 计算设备。

        Returns:
            BaseAgent: 一个已初始化的智能体实例。
        """
        if name not in self.agent_registry:
            raise ValueError(
                f"算法 '{name}' 在注册表中未找到。 "
                f"目前可用的算法: {list(self.agent_registry.keys())}"
            )
        
        agent_cls = self.agent_registry[name]
        # 动态创建智能体实例, 并传入所有必要的参数
        return agent_cls(
            obs_space=obs_space,
            act_space=act_space,
            config=config,
            device=device
        )

    def get_config_class(self, name: str) -> Type[Any]:
        """
        根据算法名称获取其注册的 Pydantic 配置类。

        Args:
            name (str): 算法的名称。

        Returns:
            Type[Any]: 对应的配置类。
        """
        if name not in self.config_registry:
             raise ValueError(
                f"算法 '{name}' 的配置类在注册表中未找到。 "
                f"目前已注册配置的算法: {list(self.config_registry.keys())}"
            )
        return self.config_registry[name]

    def list_agents(self) -> list[str]:
        """返回所有已注册算法的名称列表。"""
        return list(self.agent_registry.keys())

# 创建一个全局唯一的注册表实例, 供整个项目在导入时使用。
registry = Registry()
