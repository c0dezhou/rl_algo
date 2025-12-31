import random

import numpy as np
import torch

def set_seed(seed: int):
    # 设随机种子，保证每次跑结果差不多（至少不至于完全随机）
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 有 GPU 的话也一起设一下
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_schedule_fn(start: float, end: float, total_steps: float):
    # 线性调度：最常用的就是 epsilon 从 1 衰减到 0.05 这种
    
    def schedule(step: int):
        # 线性插值：超过 total_steps 就卡在 end
        fraction = min(float(step) / total_steps, 1.0)
        return start + fraction * (end - start)
    
    return schedule
