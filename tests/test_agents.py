import gymnasium as gym
import torch

from rl_algo.algos.grpo.agent import GRPOAgent, GRPOConfig
from rl_algo.algos.ppo.agent import PPOAgent, PPOConfig


def test_ppo_select_action_smoke():
    env = gym.make("CartPole-v1")
    agent = PPOAgent(env.observation_space, env.action_space, PPOConfig(), torch.device("cpu"))

    obs, _ = env.reset(seed=0)
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    action, info = agent.select_action(obs_t, explore=True)

    assert tuple(action.shape) == (1,)
    assert "log_prob" in info
    assert "value" in info


def test_grpo_select_action_smoke():
    env = gym.make("CartPole-v1")
    agent = GRPOAgent(env.observation_space, env.action_space, GRPOConfig(), torch.device("cpu"))

    obs, _ = env.reset(seed=0)
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    action, info = agent.select_action(obs_t, explore=True)

    assert tuple(action.shape) == (1,)
    assert "log_prob" in info
