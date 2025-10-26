import gymnasium as gym
import torch
import pytest

from rlx.core.registry import registry
from rlx.core.utils import dynamic_import_agents

# Fixture to run dynamic import once before all tests
@pytest.fixture(scope="session", autouse=True)
def setup_registry():
    dynamic_import_agents()

def test_agents_are_registered():
    """Test that our three agents are correctly registered."""
    registered_agents = registry.list_agents()
    assert "mc" in registered_agents
    assert "qlearning" in registered_agents
    assert "sarsa" in registered_agents
    print(f"\n[TEST] Registered agents: {registered_agents}")

def test_create_agent_for_each_registered():
    """Test if we can successfully create an agent for each registered algorithm."""
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
