import yaml
from pathlib import Path


def list_algos():
    # 这里写死可用算法列表
    return [
        "ppo",
        "grpo",
        "grpo_scaling",
        "a2c",
        "reinforce",
        "dqn",
        "qlearning",
        "sarsa",
        "mc",
    ]


def default_config_path(algo: str):
    # 默认配置：rl_algo/config/{algo}.yaml（存在就读，不存在就用默认 Config）
    p = Path(__file__).resolve().parent.parent / "config" / f"{algo}.yaml"
    return p if p.exists() else None


def load_config(algo: str, config_path: str | None = None):
    # 1) 先选这个算法对应的 Config 类
    # 2) 再用 yaml 覆盖（yaml 里只写要改的字段即可）
    config_class = get_config_class(algo)

    cfg_path = Path(config_path) if config_path is not None else default_config_path(algo)
    data = {}
    if cfg_path is not None and cfg_path.exists():
        loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"配置文件必须是 yaml mapping：{cfg_path}")
        data = loaded
    if hasattr(config_class, "from_dict"):
        return config_class.from_dict(data)
    return config_class(**data)


def get_config_class(algo: str):
    if algo == "ppo":
        from rl_algo.algos.ppo.agent import PPOConfig

        return PPOConfig
    if algo == "grpo":
        from rl_algo.algos.grpo.agent import GRPOConfig

        return GRPOConfig
    if algo == "grpo_scaling":
        from rl_algo.algos.grpo_scaling.agent import ScalingGRPOConfig

        return ScalingGRPOConfig
    if algo == "a2c":
        from rl_algo.algos.a2c.agent import A2CConfig

        return A2CConfig
    if algo == "reinforce":
        from rl_algo.algos.reinforce.agent import ReinforceConfig

        return ReinforceConfig
    if algo == "dqn":
        from rl_algo.algos.dqn.agent import DQNConfig

        return DQNConfig
    if algo == "qlearning":
        from rl_algo.algos.qlearning.agent import QLearningConfig

        return QLearningConfig
    if algo == "sarsa":
        from rl_algo.algos.sarsa.agent import SarsaConfig

        return SarsaConfig
    if algo == "mc":
        from rl_algo.algos.mc.agent import MCConfig

        return MCConfig
    raise ValueError(f"未知算法：{algo}，可用算法：{list_algos()}")


def create_agent(algo: str, obs_space, act_space, config, device):
    # 你可以把它理解成一个“超简单工厂函数”：
    # 输入 algo 字符串，输出对应的 agent 实例
    if algo == "ppo":
        from rl_algo.algos.ppo.agent import PPOAgent

        return PPOAgent(obs_space, act_space, config, device)
    if algo == "grpo":
        from rl_algo.algos.grpo.agent import GRPOAgent

        return GRPOAgent(obs_space, act_space, config, device)
    if algo == "grpo_scaling":
        from rl_algo.algos.grpo_scaling.agent import ScalingGRPOAgent

        return ScalingGRPOAgent(obs_space, act_space, config, device)
    if algo == "a2c":
        from rl_algo.algos.a2c.agent import A2CAgent

        return A2CAgent(obs_space, act_space, config, device)
    if algo == "reinforce":
        from rl_algo.algos.reinforce.agent import ReinforceAgent

        return ReinforceAgent(obs_space, act_space, config, device)
    if algo == "dqn":
        from rl_algo.algos.dqn.agent import DQNAgent

        return DQNAgent(obs_space, act_space, config, device)
    if algo == "qlearning":
        from rl_algo.algos.qlearning.agent import QLearningAgent

        return QLearningAgent(obs_space, act_space, config, device)
    if algo == "sarsa":
        from rl_algo.algos.sarsa.agent import SarsaAgent

        return SarsaAgent(obs_space, act_space, config, device)
    if algo == "mc":
        from rl_algo.algos.mc.agent import MCAgent

        return MCAgent(obs_space, act_space, config, device)
    raise ValueError(f"未知算法：{algo}，可用算法：{list_algos()}")
