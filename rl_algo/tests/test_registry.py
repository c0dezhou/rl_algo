import gymnasium as gym
import torch
import pytest

from rl_algo.core.registry import registry
from rl_algo.core.utils import dynamic_import_agents

# 在所有测试运行前执行一次动态导入的夹具
@pytest.fixture(scope="session", autouse=True)
def setup_registry():
    dynamic_import_agents()

def test_agents_are_registered():
    """验证智能体是否都被正确注册。"""
    registered_agents = registry.list_agents()
    for name in ("mc", "qlearning", "sarsa", "dqn", "ppo", "a2c", "reinforce"):
        assert name in registered_agents
    print(f"\n[TEST] Registered agents: {registered_agents}")

def test_create_agent_for_each_registered():
    """检查是否能为每个已注册算法成功创建智能体实例。"""
    env = gym.make("CartPole-v1")
    registered_agents = registry.list_agents()
    
    for agent_name in registered_agents:
        print(f"[TEST] Attempting to create agent: {agent_name}")
        config_class = registry.get_config_class(agent_name)
        assert config_class is not None, f"Config class for {agent_name} not found."
        
        config = config_class()
        agent = registry.create_agent(
            agent_name,
            env.observation_space,
            env.action_space,
            config,
            torch.device("cpu")
        )
        assert agent is not None, f"Failed to create agent: {agent_name}"
        print(f"[TEST] Successfully created {agent_name} agent.")
