import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

def make_env(env_id: str, seed: int, render_mode: str | None = None):
    # 就做两件事：make 环境 + 套一层 RecordEpisodeStatistics（拿到 episode 回报/长度）
    env = gym.make(env_id, render_mode=render_mode)
    
    env = RecordEpisodeStatistics(env)
    
    # 设随机种子，保证可复现
    env.reset(seed=seed)
    
    return env
