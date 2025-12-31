import gymnasium as gym
import torch

from rl_algo.algos import create_agent, list_algos, load_config


def test_factory_lists_expected_algos():
    algos = list_algos()
    for name in ("mc", "qlearning", "sarsa", "dqn", "ppo", "a2c", "reinforce", "grpo", "grpo_scaling"):
        assert name in algos


def test_factory_can_create_each_algo_smoke():
    env = gym.make("CartPole-v1")
    device = torch.device("cpu")

    for algo in list_algos():
        config = load_config(algo, None)
        agent = create_agent(algo, env.observation_space, env.action_space, config, device)
        assert agent is not None

        obs, _ = env.reset(seed=0)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action, _info = agent.select_action(obs_t, explore=True, global_step=0)

        if isinstance(action, torch.Tensor):
            assert int(action.detach().cpu().numpy()[0]) in range(env.action_space.n)
        else:
            assert int(action) in range(env.action_space.n)
