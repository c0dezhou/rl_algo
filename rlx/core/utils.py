# rlx/core/utils.py
"""
提供一系列在整个框架中可重用的、与具体算法无关的辅助函数。

该模块是框架的“工具箱”, 包含了各种通用功能, 以保持其他模块代码的
简洁和专注。

主要功能:
- `set_seed`: 设置全局随机种子, 以确保实验结果的可复现性。
- `get_schedule_fn`: 创建一个用于超参数 (如学习率、探索率) 线性调度的函数。
- `dynamic_import_agents`: 动态扫描并导入所有算法模块, 这是实现框架
  “即插即用”式扩展的关键。
"""

import importlib
import pkgutil
import random
from pathlib import Path

import numpy as np
import torch

import rlx.algos

def set_seed(seed: int):
    """
    为所有相关的随机数生成器设置种子, 以确保实验的可复现性。
    当使用相同的种子时, 随机数序列将完全相同, 从而使得模型初始化、
    环境交互和数据采样等所有随机过程都是确定的。

    Args:
        seed (int): 要设置的随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 如果 CUDA 可用, 也为所有 GPU 设置种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_schedule_fn(start: float, end: float, total_steps: float) -> callable:
    """
    创建一个线性调度函数 (闭包)。

    在强化学习中, 经常需要让某个超参数随训练步数线性变化, 例如:
    - **探索率 (epsilon)**: 在训练初期需要高探索率 (如 1.0), 随着训练进行
      逐渐降低, 以便智能体更多地利用已学到的知识 (如衰减到 0.05)。
    - **学习率 (learning rate)**: 在训练初期使用较高的学习率以快速收敛,
      后期逐渐降低以稳定训练过程。

    这个函数返回一个“调度器”函数, 该函数接收当前步数作为输入, 并返回
    该步数对应的超参数值。

    Args:
        start (float): 调度开始时的值。
        end (float): 调度结束时的最终值。
        total_steps (float): 从开始值到结束值所需的总步数。

    Returns:
        callable: 一个接收 `step` (int) 并返回 `float` 的调度函数。
    """
    
    def schedule(step: int) -> float:
        """
        根据当前步数计算线性插值后的超参数值。
        如果当前步数超过了总步数, 则始终返回结束值。
        """
        # 线性插值公式: y = y1 + (y2 - y1) * (x / x_total)
        fraction = min(float(step) / total_steps, 1.0)
        return start + fraction * (end - start)
    
    return schedule

def dynamic_import_agents():
    """
    动态地扫描并导入 `rlx.algos` 包下的所有算法模块。

    这个函数是实现框架“即插即用”功能的关键。它会自动查找 `rlx/algos`
    目录下的所有子目录 (每个子目录代表一个算法), 并尝试导入其中的
    `agent.py` 文件。

    导入这些文件会触发文件内定义的 `@registry.register_agent` 装饰器, 从而自动
    将新算法注册到全局的 `registry` 中, 无需在任何地方手动添加 `import` 语句。
    这使得添加一个新算法只需要在 `algos` 目录下创建一个新文件夹并实现相应文件即可。
    """
    # 获取 `rlx.algos` 包的物理路径
    algos_path = Path(rlx.algos.__file__).parent

    # 遍历 `rlx.algos` 目录下的所有模块
    for _, name, _ in pkgutil.iter_modules([str(algos_path)]):
        try:
            # 首先导入算法包本身, 以确保 __init__.py 被执行
            importlib.import_module(f'rlx.algos.{name}')
            # 然后再尝试导入 agent 模块, 这通常是注册器所在的位置
            importlib.import_module(f'rlx.algos.{name}.agent')
        except ImportError as e:
            # 如果导入失败 (例如, 某个子目录不是一个合法的 Python 包,
            # 或者缺少 agent.py), 则打印一个警告但允许程序继续运行。
            print(f"警告: 无法导入模块 'rlx.algos.{name}'. 原因: {e}")
