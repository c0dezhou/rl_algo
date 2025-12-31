# rl_algo（课程大作业：精简但可扩展）

本仓库保留“能跑多算法”的能力，同时把训练入口和代码结构压到尽量好讲的程度。

实现上不做“算法注册表/装饰器注册”，而是在 `rl_algo/algos/factory.py` 里用最直白的 `if/elif` 来选择算法并创建 agent/config。

课程大作业对比重点：
- PPO：Actor-Critic + GAE + PPO-Clip
- GRPO：Actor-only + group return 归一化优势 + PPO-Clip

## 快速开始
- 安装依赖：`pip install -r requirements.txt`
- 单次训练（默认自动读取 `rl_algo/config/{algo}.yaml`，也可用 `--config` 手动指定）：
  - `python -m rl_algo.train --algo ppo --seed 1 --total-steps 80000 --device auto`
  - `python -m rl_algo.train --algo grpo --seed 1 --total-steps 80000 --device auto`
- 其他算法示例：
  - `python -m rl_algo.train --algo dqn --seed 1 --total-steps 80000 --device auto`
  - `python -m rl_algo.train --algo qlearning --seed 1 --total-steps 80000 --device auto`
  - `python -m rl_algo.train --algo sarsa --seed 1 --total-steps 80000 --device auto`
- trajectory 算法每次更新用多少条轨迹（可选）：`--batch-episodes 8`（GRPO 会优先用配置里的 `group_size`）
- 可选：开启贪心评估（对所有算法都适用；结果写入 `models/{run_name}/eval_log.csv`）：
  - `--eval-frequency 10000 --eval-episodes 5`
- 备注：
  - 对 DQN，探索率衰减默认用 `epsilon_decay_fraction`（跟随 `--total-steps` 自动缩放），也可在 `rl_algo/config/dqn.yaml` 里手动改。
  - 对 DQN/Q-learning/SARSA，TimeLimit 截断不会被当成“终止状态”做 bootstrap 截断（更贴近理论定义）。
- 一键对比（PPO/GRPO × seeds=1/3/43 + 出图 + summary）：
  - 论文/报告一键跑全套（PPO / GRPO / GRPO-Scaling + DQN + SARSA，seeds=1/3/43，自动出图+汇总）：
    - `python scripts/run_cartpole_compare.py --total-steps 80000 --device auto`

## 输出产物
- `models/{run_name}/train_log.csv`：逐回合日志（episode_idx/episodic_return/episodic_length/elapsed_sec）
- `models/{run_name}/config.yaml`：本次运行的配置快照
- `models/{run_name}/final.pt`：最终模型权重与优化器状态
- `models/{run_name}/eval_log.csv`：评估日志（global_step/eval_return_mean；需开启 `--eval-frequency`）
- `results/summary.json`：对比脚本的聚合统计
- `figs/*`：对比曲线与报告图

## 支持算法
- trajectory-on-policy：`ppo` / `grpo` / `a2c` / `reinforce` / `mc`
- step-on-policy：`qlearning` / `sarsa`
- off-policy：`dqn`
