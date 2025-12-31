# （中文题目占位）CartPole-v1 上 PPO 与 GRPO 的实现与对比实验

作者：*TODO 姓名*  
单位：*TODO 单位/学院*  
邮箱：*TODO 邮箱*  
（可选）中图法分类号：*TODO*；DOI：*TODO*  

## 中文摘要
本文基于轻量强化学习代码库 `rl_algo/`，在 Gymnasium 的 CartPole-v1 环境上实现并对比两种同策略方法：PPO（Actor-Critic + GAE + PPO-Clip）[1][3] 与 GRPO（Actor-only + 组内回报归一化优势 + PPO-Clip 风格更新）[2]。工程实现方面，我们将训练入口统一到 `rl_algo/train.py`：支持按算法类型划分的三种训练流程（off-policy / step-on-policy / trajectory-on-policy），其中同策略轨迹类算法支持“按 *batch_episodes* 收集多条完整轨迹后再更新”，并输出逐回合日志 `models/{run_name}/train_log.csv`；同时提供对所有算法通用的可选贪心评估（`--eval-frequency`），输出 `models/{run_name}/eval_log.csv`，用于稳定地对比“训练曲线”和“评估曲线”。实验方面，本文提供三随机种子（1/3/43）的一键对比脚本与自动出图脚本；当前仓库内的 `results/summary.json` 来自 `--total-steps 80000` 的对比实验。结果显示：PPO 在 80k steps 的交互预算内达到回报=500 更快、后期更稳定；GRPO 能多次达到 500，但稳定满分平台（MA10=500）未在 80k steps 内出现。本文同时给出“按 episode 对齐”和“按 steps 对齐”的曲线与统计口径，便于在报告中解释样本效率、稳定性与 wall-clock 的差异来源。

关键词：强化学习；策略梯度；PPO；GRPO；CartPole；可复现；工程实现

## English Title (placeholder)
*TODO* PPO and GRPO on CartPole-v1: Implementation and Comparative Study

Author: *TODO*  
Affiliation: *TODO*  

## Abstract
This report implements and compares two on-policy methods on CartPole-v1 using the lightweight `rl_algo/` codebase: PPO (Actor-Critic with GAE and PPO-Clip) [1][3] and GRPO (Actor-only with group-normalized episode-return advantages and a PPO-Clip-style objective) [2]. We unify training into `rl_algo/train.py` with three easy-to-explain training modes (off-policy / step-on-policy / trajectory-on-policy). For on-policy trajectory methods, the runner supports collecting multiple complete episodes (*batch_episodes*) before each update, and logs per-episode statistics to `models/{run_name}/train_log.csv`. We also provide an optional greedy evaluation interface for all algorithms (`--eval-frequency`), writing `models/{run_name}/eval_log.csv`, so that training curves and evaluation curves can be compared consistently. We provide one-click scripts for running PPO vs. GRPO across three random seeds (1/3/43) and for generating all report figures. The current `results/summary.json` in the repository is generated with `--total-steps 80000`. Under this interaction budget, PPO reaches the 500-return threshold earlier and exhibits better stability, while GRPO improves but does not form a stable full-score plateau (MA10=500) within 80k steps. We also report both episode-aligned and step-aligned curves to fairly interpret sample efficiency and wall-clock costs.

Key words: Reinforcement learning; Policy gradient; PPO; GRPO; CartPole; Reproducibility; Engineering

---

# 1 引言

## 1.1 研究背景
强化学习在控制任务中面临的核心挑战是：在与环境交互产生的轨迹数据上，学习一个能够最大化长期回报的策略。CartPole-v1 是经典的基准控制任务，因其状态维度低、训练速度快、最优回报上限清晰（单回合最大回报为 500）而常用于同策略算法的实现与对比。

本作业选择对比 PPO 与 GRPO 的原因是：二者都可使用 PPO-Clip 风格的更新来提升策略优化的稳定性，但 GRPO 去除了 Critic，改用“组内回报归一化”的方式提供优势信号，从而形成“有/无 Critic”的对照。

## 1.2 问题定义与目标
任务：在 CartPole-v1 环境上实现 PPO 与 GRPO（Gym 适配版），并在公平设置下进行对比实验。  
评价指标：
- 收敛：达到回报=500 所需回合数（主判据：首次单回合回报=500；辅判据：最近10回合平均=500）。
- 效率：相同回合数下的墙钟时间（`elapsed_sec`，由训练过程记录）。
补充口径（用于更公平的样本效率比较）：
- 样本效率：达到回报=500 所需交互步数（*steps-to-500*），由 `episodic_length` 累加得到累计交互步数 *cum_steps* 后计算。

如表1所示，给出本文使用的数据字段与指标来源。

表1 指标与数据来源（定义口径）

| 指标/字段 | 定义 | 来源文件/字段 |
|---|---|---|
| *episodic_return* | 单回合未折扣回报（CartPole-v1 每步 reward=1，最大 500） | `models/{run_name}/train_log.csv:episodic_return` |
| *episodic_length* | 单回合长度（步数） | `models/{run_name}/train_log.csv:episodic_length` |
| *elapsed_sec* | 从训练开始到该回合结束的累计墙钟时间（秒） | `models/{run_name}/train_log.csv:elapsed_sec` |
| *episodes-to-500* | 首次出现 *episodic_return*≥500 的回合号；若不存在则“未达到” | 由 `train_log.csv` 计算（脚本实现） |
| *cum_steps* | 训练到某回合结束时累计与环境交互的步数（Σ *episodic_length*） | 由 `train_log.csv:episodic_length` 累加得到（脚本实现） |
| *steps-to-500* | 首次出现 *episodic_return*≥500 时对应的 *cum_steps* | 由 `train_log.csv` 计算（脚本实现） |

## 1.3 本文贡献
- 提供 `rl_algo/` 上 PPO/GRPO 的课程作业口径实现（PPO：Actor-Critic+GAE+Clip；GRPO：Actor-only+组内回报归一化优势+Clip）。
- 在统一训练入口实现“按 *batch_episodes* 收集多条完整轨迹后再更新”的同策略训练机制，便于扩展其他 on-policy 轨迹算法。
- 提供三随机种子（1/3/43）的一键对比脚本与自动出图脚本，输出可直接用于报告的曲线图（PNG+PDF）与汇总 JSON。
- 基于 `--total-steps 80000` 的真实实验日志，补齐 episodes-to-500 / steps-to-500 / time-to-500 等指标，并对“稳定性与运行时变慢”现象给出可复现的量化分析。

# 2 相关工作

## 2.1 策略梯度与 Actor-Critic 简述
策略梯度方法直接对策略 *π(a|s)* 的参数做梯度上升，通过采样轨迹估计期望回报的梯度。Actor-Critic 结构使用 Critic 近似价值函数 *V(s)* 作为基线以降低方差。

## 2.2 PPO 的核心思想（clipped surrogate）
PPO 通过对重要性采样比率 *ratio* 进行剪切（clip），限制单次更新造成的策略变化幅度，提升训练稳定性 [1]。该方法的工程特征是“采样一批数据后进行多 epoch 的 minibatch 优化”，以更充分利用同一批轨迹样本 [1]。

## 2.3 GRPO 思路与组内优势归一化
GRPO 在 DeepSeekMath 论文中被提出，用于在不训练 Critic 的情况下，通过“组内相对奖励”估计 baseline 并构造优势，从而减少训练资源开销 [2]。在 outcome supervision 描述中，对同一问题采样得到的 *G* 个输出奖励做归一化（减去组均值并除以组标准差），并将归一化奖励作为序列内所有 token 的优势 [2]。本文实现的是 Gym 适配版本：把一个 batch 采集到的多条完整轨迹视为一个 group，把每条轨迹的 *episode return* 视为“奖励分数”，并按作业要求使用组内均值/标准差归一化优势，广播给该轨迹内每个时间步。

与 `grpo.pdf` 的原始 LLM 场景相比，本文的 Gym 适配做了两点“口径替换”（不改变核心思想）：
- “问题-输出”的组：替换为“一次 update 收集的多条完整轨迹（episodes）”
- “奖励模型给出的分数”：替换为“环境给出的未折扣 episode return（CartPole 单回合最大 500）”

如表2所示，给出 PPO 与 GRPO 的方法对比。

表2 PPO vs GRPO 方法对比（实现口径）

| 维度 | PPO | GRPO（Gym 适配版） |
|---|---|---|
| 是否有 Critic | 有（Actor-Critic） | 无（Actor-only） |
| 优势估计 | GAE（逐步优势）[3] | 组内 *episode return* 归一化（轨迹级常数优势广播）[2] |
| 损失组成 | policy + value + entropy | policy + entropy |
| 更新稳定性 | PPO-Clip（clipped surrogate）[1] | PPO-Clip 风格更新（但无 value loss） |

# 3 主要成果与方法

## 3.1 PPO 方法

### 3.1.1 Actor-Critic 结构
PPO 的 Actor 输出动作分布 logits，Critic 输出 *V(s)*。本文实现中两者网络结构一致但参数独立，以便设置不同学习率（`actor_lr` 与 `critic_lr`）。

### 3.1.2 GAE 优势估计
本文在 `rl_algo/train.py` 中对每条轨迹独立计算 GAE（函数 `add_gae_advantages_and_returns`）。其核心是 TD 误差与递推：

公式(1)（GAE/TD-δ 递推）[3]
- *δ_t* = *r_t* + *γ*·*V(s_{t+1})* − *V(s_t)*
- *A_t* = *δ_t* + *γ*·*λ*·*A_{t+1}*  （IF *done* THEN 截断：*V(s_{t+1})=0*）

### 3.1.3 PPO-Clip 目标与损失函数
PPO 的 clipped surrogate objective 使用重要性比率：
- *ratio* = exp(*logp_new* − *logp_old*)
- *clip_ratio* = clip(*ratio*, 1−*ε*, 1+*ε*)

公式(2)（clipped surrogate；对应 `ppo.pdf` 中的 L^CLIP 形式）  
- *L_clip(θ)* = E[ min(*ratio_t(θ)*·*Â_t*, clip(*ratio_t(θ)*, 1−*ε*, 1+*ε*)·*Â_t* ) ]  [1]

本文实现中：
- policy loss = −mean(min(...))  
- value loss = MSE(*V(s)*, *returns*)  
- entropy bonus = mean(entropy)

## 3.2 GRPO 方法（Gym 适配版）

### 3.2.1 去除 Critic 的动机与结构
GRPO 去除 Critic，仅保留 Actor（策略网络），损失中不包含 value loss。

### 3.2.2 Group 优势估计（作业规定口径）
一次 update 收集的多条完整轨迹视为一个 group。令第 *i* 条轨迹的未折扣回报：
- *R_i* = Σ_t *r_{i,t}*

在 group 内进行均值/标准差归一化得到轨迹级优势，并广播给轨迹内所有 step：

公式(3)（group 优势归一化；Gym 适配版，思想来源于 GRPO 的组内归一化描述 [2]）
- *A_i* = (*R_i* − mean(*R*))/ (std(*R*) + *eps*)
- 对轨迹 *i* 的所有时间步：*A_{i,t} = A_i*

说明：在 `grpo.pdf` 的 4.1.2 节（Outcome Supervision）中，GRPO 将同一组内的奖励做归一化，并令序列内所有 token 的优势等于该归一化奖励（形式为 *Â_{i,t} = (r_i − mean(r))/std(r)*）[2]。本文按作业要求把 *r_i* 具体化为 CartPole 的未折扣 *episode return*（*R_i*），并对轨迹内每个时间步广播常数优势。

### 3.2.3 GRPO 的 PPO-Clip 风格更新
为保证稳定性，GRPO 仍采用 PPO-Clip 风格的 clipped surrogate objective，只是没有 value loss。

# 4 关键实现技术（结合 rl_algo 真实代码）

## 4.1 系统架构与模块划分
项目按“最小可讲解实现”划分为：训练入口、算法实现、环境封装、配置与脚本。

### 4.1.1 训练入口（精简版：显式工厂 + 三种训练模式 + 可选评估）
- `rl_algo/train.py`：统一训练主循环；按算法类型分三种更新模式（off-policy / step-on-policy / trajectory-on-policy）；写 `train_log.csv`；可选写 `eval_log.csv`；保存 `final.pt` 与 `config.yaml`。
- `rl_algo/algos/factory.py`：用最直白的 `if/elif` 写死算法列表，并按 `--algo` 创建对应的 `Config` 与 `Agent`（不做“注册表/装饰器注册”）。
- `rl_algo/algos/__init__.py`：仅导出 `factory.py` 里的几个函数（`list_algos/load_config/create_agent`），让训练入口导入更简单。
- `rl_algo/envs/make_env.py`：创建环境并包上 `RecordEpisodeStatistics`（便于拿到 episode return/length）。

### 4.1.2 Agent 接口（最小契约）
核心接口由 `rl_algo/core/base_agent.py` 约束，训练脚本只依赖以下最小方法：
- `select_action(obs, explore, global_step)`：返回 action 与 info（例如 `log_prob`；PPO/A2C 额外可能返回 `value`）。
- `update(batch, global_step)`：用于 off-policy 与 trajectory-on-policy 的批量更新。
- `train_step(transition)`：用于 step-on-policy 的逐步更新（Q-learning/SARSA）。
- `save()/load()`：模型与优化器状态的序列化/恢复。

## 4.2 数据采样与批量更新机制
`rl_algo/train.py` 按算法类型支持三种“可讲解”的训练流程：
- **off-policy**（如 DQN）：ReplayBuffer 存数据，按 `train_frequency` 采样 batch 调 `agent.update`。
- **step-on-policy**（Q-learning/SARSA）：每个时间步构造 `transition`，直接调用 `agent.train_step`。
- **trajectory-on-policy**（PPO/GRPO/A2C/REINFORCE/MC）：缓存一条完整轨迹，episode 结束后加入 batch；当 batch 的轨迹条数达到 `batch_episodes`（GRPO 用 `group_size`）时，扁平化为 step-level batch 并调用 `agent.update`。

其中 PPO 的优势/回报字段可由训练脚本计算并传入（GAE：`use_gae=true` 时生效）；GRPO 的 group 优势由训练脚本按 episode return 归一化后传入。

## 4.3 训练日志、评估与模型保存
### 4.3.1 train_log.csv 字段定义
`rl_algo/train.py` 在每个 episode 结束时写入：
- `episode_idx`：回合编号（从 1 开始）
- `episodic_return`：未折扣回报（CartPole-v1 上限 500）
- `episodic_length`：回合长度（步数）
- `elapsed_sec`：从训练开始到该回合结束的累计墙钟时间（秒）

### 4.3.2 模型保存策略
训练结束保存一次最终模型：`models/{run_name}/final.pt`（包含 `agent_state` 与运行元信息）。同时保存本次运行的配置快照：`models/{run_name}/config.yaml`（便于复现实验）。

### 4.3.3 可选：贪心评估与 eval_log.csv
为避免“训练时的探索噪声”与“是否真的学到策略”混在一起，`rl_algo/train.py` 提供对所有算法通用的可选评估：
- 开启方式：命令行参数 `--eval-frequency N --eval-episodes K`（`N=0` 表示关闭）。
- 评估口径：使用 `agent.select_action(..., explore=False)` 进行贪心动作选择；评估环境与训练环境独立创建，评估结果不参与训练更新。
- 输出文件：`models/{run_name}/eval_log.csv`，字段为 `global_step` 与 `eval_return_mean`。

实现细节补充：对需要 bootstrap 的价值更新算法（DQN / Q-learning / SARSA），TimeLimit 截断不视为“终止状态”来截断目标（仅对 `terminated=True` 截断），更贴近理论定义并避免接近满分时的价值被硬截断。

## 4.4 可复现实验脚本
`scripts/run_cartpole_compare.py` 负责：
1) 跑 PPO/GRPO × seeds(1,3,43)  
2) 写 `results/summary.json`  
3) 调用 `scripts/plot_report_figures.py` 一键生成报告用图与 `fig_list.md`  

## 4.5 伪代码（按模板风格：IF/THEN 大写，变量/函数名斜体）

### 过程1 一次更新的数据准备过程（采样→攒 batch→算优势→扁平化→minibatch）
1) 初始化空缓存：*episode_obs/actions/log_probs/rewards/dones/values*，以及 *batch_episodes_data*。  
2) REPEAT 每个时间步与环境交互：得到 *s_t, a_t, r_t, done_t*，并缓存到 *episode_*。  
3) IF *done_t* THEN  
   3.1) 计算该轨迹未折扣回报 *R_i = Σ_t r_t*；  
   3.2) 将该轨迹打包为 *episode_dict* 并 APPEND 到 *batch_episodes_data*；  
   3.3) CLEAR *episode_*，开始采集下一条轨迹。  
4) IF len(*batch_episodes_data*) == *batch_episodes*（或 *group_size*） THEN  
   4.1) 将多条轨迹 CONCAT 并扁平化为 step-level 大 batch：*observations/actions/log_probs/rewards/dones*；  
   4.2) IF *algo* == PPO THEN 逐轨迹计算 GAE 得到 *advantages/returns*；  
       ELSE IF *algo* == GRPO THEN 计算组内归一化优势 *A_i* 并广播到每个 step；  
   4.3) SHUFFLE indices 并按 *minibatch_size* 切分，进行多 epoch 的优化更新。  

### 算法1 PPO（Actor-Critic + GAE + PPO-Clip）
输入：环境 *Env*；策略参数 *θ*；价值参数 *ψ*；超参 *γ, λ, ε, c_v, c_e, batch_episodes, update_epochs, minibatch_size*。  
输出：训练后的 *θ, ψ*；日志 `train_log.csv`；模型 `final.pt`。

1) 初始化 *Actor(θ)* 与 *Critic(ψ)*。  
2) REPEAT 直到 *global_step* 达到 *total_steps*：  
   2.1) 按“过程1”收集 *batch_episodes* 条完整轨迹并得到 step-level batch。  
   2.2) 对每条轨迹按公式(1)计算 GAE：得到 *Â_t*，并计算 *return_t = Â_t + V_ψ(s_t)*。  
   2.3) FOR *epoch* = 1..*update_epochs* DO  
       2.3.1) FOR 每个 minibatch DO  
           - 计算 *logp_new = log π_θ(a_t|s_t)*，*logp_old* 来自采样时缓存；  
           - 计算 *ratio = exp(logp_new − logp_old)*；  
           - 计算 *policy_loss = − mean( min(ratio·Â, clip(ratio,1−ε,1+ε)·Â) )*；  
           - 计算 *value_loss = MSE(V_ψ(s), return)*；  
           - 计算 *entropy = H(π_θ)*；  
           - 更新 *θ* 以最小化 *policy_loss − c_e·entropy*；更新 *ψ* 以最小化 *c_v·value_loss*。  
       2.3.2) END FOR  
   2.4) END FOR  

### 算法2 GRPO（Actor-only + Group 优势归一化 + PPO-Clip 风格更新）
输入：环境 *Env*；策略参数 *θ*；超参 *γ, ε, c_e, group_size, update_epochs, minibatch_size*。  
输出：训练后的 *θ*；日志 `train_log.csv`；模型 `final.pt`。

1) 初始化 *Actor(θ)*。  
2) REPEAT 直到 *global_step* 达到 *total_steps*：  
   2.1) 按“过程1”收集 *group_size* 条完整轨迹并计算每条轨迹未折扣回报 *R_i*。  
   2.2) 计算组内优势（公式(3)）：*A_i = (R_i − mean(R))/ (std(R)+eps)*，并广播为 *A_{i,t}=A_i*。  
   2.3) FOR *epoch* = 1..*update_epochs* DO  
       2.3.1) FOR 每个 minibatch DO  
           - 计算 *logp_new = log π_θ(a_t|s_t)*，*logp_old* 来自采样时缓存；  
           - 计算 *ratio = exp(logp_new − logp_old)*；  
           - 计算 *policy_loss = − mean( min(ratio·A, clip(ratio,1−ε,1+ε)·A) )*；  
           - 计算 *entropy = H(π_θ)*；  
           - 更新 *θ* 以最小化 *policy_loss − c_e·entropy*（无 value loss）。  
       2.3.2) END FOR  
   2.4) END FOR  

# 5 验证与实验

## 5.1 实验环境与设置
### 5.1.1 环境
- Gym 环境：CartPole-v1
- 软件版本：*TODO 自动填表/手工填写（Python、PyTorch、Gymnasium、CUDA 等）*
- 硬件环境：*TODO（CPU/GPU 型号）*

### 5.1.2 公平性控制（公共超参一致）
本文通过 `rl_algo/config/ppo.yaml` 与 `rl_algo/config/grpo.yaml` 控制公平性：除算法特有参数外，其余公共参数一致。  
公平性方面，本文统一环境交互步数预算（`total_steps`），并尽量保持 PPO 与 GRPO 系列在网络宽度、学习率量级、更新轮数（`update_epochs`）等设置上的一致。为对齐每次更新的采样规模与更新频率，本文将 PPO 的 `batch_episodes` 与 GRPO 的 `group_size` 统一设为相同数值，并采用“每次更新前收集若干条完整回合（episode/trajectory）”的口径进行对比。需要注意：on-policy 方法每次更新的样本来自完整回合，回合长度随策略改善而增长，因此“同样的 `total_steps`”会对应不同的 episode 总数。为避免误读，本文同时报告“回报 vs episode”与“时间 vs episode”两类曲线，并在统计表中给出达到 500 的 episode 与耗时。  
算法特有参数：
- PPO：`critic_lr`
- GRPO：`group_size`

如表4所示，本文按要求控制变量：除算法特有超参外，其余公共超参保持一致（数值来自 `rl_algo/config/ppo.yaml` 与 `rl_algo/config/grpo.yaml`）。

表4 超参数对比表（公共/特有）

| 超参 | PPO | GRPO | 说明 |
|---|---:|---:|---|
| `gamma` | 0.99 | 0.99 | 公共参数 |
| `hidden_dim` | 128 | 128 | 公共参数（Actor 结构一致） |
| `batch_episodes` | 8 | 8 | 公共参数（一次更新收集的轨迹条数口径） |
| `actor_lr` | 3e-4 | 3e-4 | 公共参数 |
| `update_epochs` | 4 | 4 | 公共参数 |
| `minibatch_size` | 256 | 256 | 公共参数 |
| `clip_coef` | 0.2 | 0.2 | 公共参数（PPO-Clip） |
| `entropy_coef` | 0.01 | 0.01 | 公共参数 |
| `max_grad_norm` | 0.5 | 0.5 | 公共参数 |
| `critic_lr` | 1e-3 | — | PPO 特有（Critic 学习率） |
| `value_coef` | 0.5 | — | PPO 特有（value loss 系数） |
| `gae_lambda` | 0.95 | — | PPO 特有（GAE 系数） |
| `group_size` | — | 8 | GRPO 特有（group 轨迹条数） |

### 5.1.3 随机种子设置
对比实验固定 seeds = 1 / 3 / 43（见 `results/summary.json`）。

## 5.2 指标定义与统计口径
### 5.2.1 episodic_return
使用未折扣回报（CartPole-v1 每步 reward=1，最大 500），对应 `train_log.csv` 的 `episodic_return`。

补充说明：在 CartPole-v1 中，每个时间步 reward=1，因此单回合未折扣回报就是该回合的步数之和，数值上通常与 `episodic_length` 非常接近（甚至相等）。这属于环境奖励定义导致的“预期现象”，不是代码 bug。

### 5.2.2 episodes-to-500（收敛）
主判据：首次出现单回合 `episodic_return>=500` 的 `episode_idx`。  
辅判据：首次出现最近10回合平均 `>=500` 的 `episode_idx`（窗口=10）。

### 5.2.3 wall-clock（效率）
`elapsed_sec` 为训练开始到当前回合结束的累计墙钟秒数。对比“相同回合数下耗时”曲线，或对比到达 500 时的耗时（time-to-500）。

注意：训练内部是否使用折扣因子 γ 与“报告指标是否使用折扣回报”是两个不同问题。本项目中：
- **训练（PPO）**：GAE/returns 的计算使用 γ（见 `rl_algo/train.py` 的 `add_gae_advantages_and_returns`）。
- **统计与画图**：`episodic_return` 采用未折扣回报（CartPole 口径，最大 500），用于 episodes-to-500 的判据与曲线展示。

## 5.3 实验结果与分析（严禁编造：全部来自日志/汇总文件）
本仓库当前实验汇总来自：
- 命令：`python scripts/run_cartpole_compare.py --total-steps 80000`
- 汇总文件：`results/summary.json`（`total_steps=80000`）

若要进一步降低统计方差或获得更充分的收敛对比，可提高总步数（例如 `--total-steps 200000`）并重新出图（脚本会自动完成）。

### 5.3.1 平均回报曲线（3 seeds 平均 + std 阴影）
如图10所示，给出了 PPO 与 GRPO 的平均回合回报曲线（3 种子均值±标准差）。

![图10 CartPole-v1 平均回合回报曲线（3 种子，均值±标准差）](figs/fig10_cartpole_return_mean_std.png)

### 5.3.2 平均时间曲线（3 seeds 平均 + std 阴影）
如图11所示，给出了相同回合数下的累计墙钟时间曲线（3 种子均值±标准差）。

![图11 CartPole-v1 平均墙钟时间曲线（3 种子，均值±标准差）](figs/fig11_cartpole_time_mean_std.png)

### 5.3.3 稳定性分析（单种子/滑动平均/方差）
如图12–14所示，分别给出了单种子曲线、平均回报的 MA10/MA20 曲线，以及跨种子的标准差随回合变化曲线。

![图12(a) PPO 单种子回合回报曲线（3 seeds）](figs/fig12a_return_per_seed_ppo.png)

![图12(b) GRPO 单种子回合回报曲线（3 seeds）](figs/fig12b_return_per_seed_grpo.png)

![图13 平均回报的滑动平均曲线（MA10 / MA20）](figs/fig13_return_moving_average.png)

![图14 回报标准差随回合变化（3 种子）](figs/fig14_return_std_over_episode.png)

### 5.3.4 收敛速度与效率汇总（episodes-to-500 / time-to-500）
如图15–16所示，给出了“首次单回合回报达到 500”的回合数与对应墙钟时间。  
为了避免“回合变长导致 episode 对齐产生偏差”，本文额外统计了 *steps-to-500*（达到 500 所需交互步数），并在表6中同时给出“首次达到 500”与“MA10=500（稳定满分平台）”两种口径。

表6 收敛与效率汇总（`total_steps=80000`，来自 `train_log.csv` 计算；详见 `results/analysis_metrics_table.csv`）

| 算法 | 种子 | 记录回合数 | episodes-to-500 | steps-to-500 | time-to-500（秒） | first_ep_ma10_500 | steps-to-ma10 | time-to-ma10（秒） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO | 1 | 410 | 303 | 29385 | 68.68 | 362 | 56170 | 130.78 |
| PPO | 3 | 380 | 246 | 22486 | 54.39 | 354 | 66799 | 158.42 |
| PPO | 43 | 420 | 275 | 21723 | 51.80 | 395 | 67816 | 157.50 |
| GRPO | 1 | 592 | 475 | 36892 | 64.11 | — | — | — |
| GRPO | 3 | 535 | 450 | 51427 | 89.41 | — | — | — |
| GRPO | 43 | 536 | 383 | 32421 | 62.40 | — | — | — |

由表6可见，在 80k steps 的交互预算内：
- PPO 的平均 *steps-to-500* 约为 24.5k，而 GRPO 约为 40.2k（PPO 更省样本，约节省 39% 交互步数）。
- PPO 的 3 个种子均进入稳定满分平台（MA10=500），而 GRPO 在 80k steps 内未进入该平台，表现为“能多次到 500，但尾部仍有较多失败回合”。

![图15 达到回报=500 的回合数（首次单回合）](figs/fig15_episodes_to_500_bar.png)

![图16 达到回报=500 的墙钟时间（首次单回合）](figs/fig16_time_to_500_bar.png)

### 5.3.5 用 steps 口径重新看学习曲线（更公平的样本效率比较）
如图17所示，将横轴从 episode 改为累计交互步数（steps）后，可以更直接地回答“同等交互预算下谁学得更快”。该口径下 PPO 的回报曲线整体更靠左、更早达到 500；而 GRPO 曲线虽上升，但均值与方差表现出更强的不稳定性。

![图17 平均回合回报曲线（steps 对齐，3 种子，均值±标准差）](figs/fig17_return_mean_std_vs_steps.png)

如图18(a)–18(b)所示，PPO 的单种子曲线在后期更容易形成接近 500 的平台；GRPO 则常出现“达到 500 与跌回 300–400 区间交替”的现象，这与其优势估计方式更粗粒度、方差更大的理论预期一致（见 5.4）。

![图18(a) PPO 单种子回合回报曲线（steps 横轴，3 seeds）](figs/fig18a_ppo_seed_curves_vs_steps.png)

![图18(b) GRPO 单种子回合回报曲线（steps 横轴，3 seeds）](figs/fig18b_grpo_seed_curves_vs_steps.png)

### 5.3.6 运行时现象：后期“变慢”的来源与量化
训练日志中 `elapsed_sec` 是累计墙钟时间，因此每回合耗时增量 *dt* 可由相邻回合差分得到。由于本项目采用“攒够 *batch_episodes/group_size* 条完整轨迹后再 update”的训练方式，当策略变好、回合变长（接近 500）时，**每次 update 的 batch（按 transition 数）会显著增大**，从而导致后期 update 开销上升、体感变慢。

如图19所示，以“update 触发回合”的额外停顿 *dt_update - dt_non_update* 作为估计量，PPO 的额外停顿随训练阶段显著增大，而 GRPO 增幅较小。这与 PPO 额外包含 Critic 前向/反向与 value loss、并采用多 epoch minibatch 更新的计算结构一致。

![图19 update 额外停顿随训练阶段变化（dt_update - dt_non_update）](figs/fig19_update_extra_pause_by_stage.png)

如图20所示，吞吐（steps/sec）随训练进程会发生变化：当回合更长、update batch 更大时，吞吐曲线会下降或波动加大。对于 CartPole 这类“小网络 + 单环境”的任务，使用 GPU 未必总能显著加速；报告中应将该现象解释为“单步推理/频繁张量构造的开销 + update 批量变大带来的反向计算开销”的综合结果。

![图20 训练吞吐曲线（steps/sec，MA10，steps 对齐，均值±标准差）](figs/fig20_throughput_steps_per_sec.png)

## 5.4 讨论：Critic 是否必要
在 CartPole 这类低维控制任务中，PPO 的 Critic 并非“形式上可有可无”，而是与优势估计的方差控制/信用分配强相关：

1) **优势估计的粒度差异（credit assignment）**  
PPO 的 baseline 是状态相关的 *V(s)*，并通过 GAE 递推把时间步级别的 TD-δ 信息融合为更平滑的优势 *Â_t*，因此能更细粒度地反映“在哪些状态下的动作更好”，通常表现为更快进入稳定平台。  
相比之下，作业版 GRPO 去掉 Critic，用 group 内回合回报的均值/标准差构造“轨迹级”的相对优势 *A_i*，并广播给该轨迹内所有 step。这使得**同一条轨迹内的所有时间步共享同一个优势信号**，在经典控制任务里更像“episode-level baseline 的 REINFORCE”，信用分配更粗、梯度方差更大，容易出现“偶尔到 500，但平台稳不住”的现象。

2) **表6 与图18 的现象对应**  
表6显示：在 80k steps 内，GRPO 3 个种子都能首次达到 500，但均未达到 MA10=500；图18(b)也体现为尾部回合仍频繁跌回 300–400。相对地，PPO 在 3 个种子上均达到 MA10=500，说明其不仅“能达到”，而且“能稳定保持”。这一差异与上述方差控制机制一致。

3) **效率口径需要区分：episode 对齐 vs steps 对齐**  
若按 episode 对齐比较 wall-clock，弱策略往往回合更短，因此看起来“同样 episode 数耗时更少”；但这并不等价于“样本效率更高”。因此报告中应同时给出 steps 口径（图17、表6 的 *steps-to-500*）以做更公平的解释。

# 6 结论
## 6.1 结论总结
- 本文在 `rl_algo/` 代码库上实现了 PPO 与 GRPO（Gym 适配版），并提供三随机种子的可复现实验管线与报告出图脚本。
- PPO 采用 Actor-Critic+GAE+PPO-Clip；GRPO 采用 Actor-only+组内回报归一化优势+PPO-Clip 风格更新。
- 实验指标与统计口径均由 `train_log.csv` 与 `results/summary.json` 明确给出，可通过一条命令复现实验与图表。
- 在 `--total-steps 80000` 的交互预算内，PPO 相比 GRPO 具有更好的样本效率（更小 *steps-to-500*）与更强稳定性（达到 MA10=500），但后期 update 开销更大、速度下降更明显。

## 6.2 局限性与未来工作
- 当前仓库内的对比实验基于 `--total-steps 80000`；若要获得更充分的统计与更接近最终收敛的平台对比，可提高到 200k steps 并复现出图。
- 未来可扩展：在更复杂环境上对比；增加随机种子数量；或引入更严格的统计检验。

---

# 参考文献
[1] Schulman J, Wolski F, Dhariwal P, Radford A, Klimov O. Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347v2, 2017.  
[2] Shao Z, Wang P, Zhu Q, Xu R, Song J, Bi X, Zhang H, Zhang M, Li Y K, Wu Y, Guo D. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv preprint arXiv:2402.03300v3, 2024.（第 4.1 节给出 GRPO 的组内相对优势思想）  
[3] Schulman J, Moritz P, Levine S, Jordan M, Abbeel P. High-Dimensional Continuous Control Using Generalized Advantage Estimation. arXiv preprint arXiv:1506.02438, 2015.  
[4] Brockman G, Cheung V, Pettersson L, Schneider J, Schulman J, Tang J, Zaremba W. OpenAI Gym. arXiv preprint arXiv:1606.01540, 2016.

# 附：项目审计与代码地图（必须据实）

## A.1 项目目标与实现范围
### A.1.1 任务目标
在 CartPole-v1 上实现 PPO 与 GRPO，并在公平设置下对比收敛与效率。

### A.1.2 代码库实现的算法清单（按 `rl_algo/algos/` 目录）
- 策略梯度/Actor-Critic：PPO（`rl_algo/algos/ppo/agent.py`）、A2C（`rl_algo/algos/a2c/agent.py`）、REINFORCE（`rl_algo/algos/reinforce/agent.py`）、GRPO（`rl_algo/algos/grpo/agent.py`）
- 值函数/表格方法：DQN（`rl_algo/algos/dqn/agent.py`）、Q-learning（`rl_algo/algos/qlearning/agent.py`）、SARSA（`rl_algo/algos/sarsa/agent.py`）、MC 控制（`rl_algo/algos/mc/agent.py`）

## A.2 目录结构与代码地图（文件→职责→关键函数）
### A.2.1 `rl_algo/train.py`
- 职责：训练入口；创建环境与 agent；主循环交互；按算法类型选择训练流程；写 `train_log.csv`；保存 `final.pt` 与 `config.yaml`。
- 关键函数：`main()`；`build_flat_batch()`；`add_gae_advantages_and_returns()`；`add_grpo_group_advantages()`。
- 对应作业要求：支持同策略“多轨迹 batch 更新”；输出 wall-clock 与回报日志；3 seeds 实验可复现。

### A.2.2 core/envs 层
- `rl_algo/core/base_agent.py`：Agent 接口契约。
- `rl_algo/core/types.py`：Config/Batch/Transition 的统一类型。
- `rl_algo/core/buffers.py`：ReplayBuffer（DQN 使用）。
- `rl_algo/core/utils.py`：set_seed/线性调度等通用工具。
- `rl_algo/envs/make_env.py`
  - 职责：创建环境并包上 `RecordEpisodeStatistics`。

### A.2.3 algos 层
- `rl_algo/algos/factory.py`
  - 职责：最小“工厂函数集合”，根据 algo 名称创建 config/agent（写死列表 + if/elif）。
- `rl_algo/algos/ppo/agent.py`
  - 职责：PPO Agent（Actor-Critic + PPO-Clip），`select_action` 返回 `log_prob/value`。
- `rl_algo/algos/grpo/agent.py`
  - 职责：GRPO Agent（Actor-only + PPO-Clip 风格更新），`select_action` 返回 `log_prob`，`update` 使用外部传入的 group 优势。

### A.2.4 配置与脚本
- `rl_algo/config/ppo.yaml`、`rl_algo/config/grpo.yaml`
  - 职责：超参数配置；用于公平性控制（公共参数一致，特有参数单列）。
- `scripts/run_cartpole_compare.py`
  - 职责：一键运行 PPO/GRPO × seeds(1,3,43)；生成 `results/summary.json`；自动调用出图脚本。
- `scripts/plot_report_figures.py`
  - 职责：一键生成报告所需图表（>=10，PNG+PDF）并写 `fig_list.md`。

## A.3 运行方式与产物路径（据实）
### A.3.1 单跑命令
- PPO：`python -m rl_algo.train --algo ppo --env-id CartPole-v1 --seed 1 --total-steps 80000 --device auto`
- GRPO：`python -m rl_algo.train --algo grpo --env-id CartPole-v1 --seed 1 --total-steps 80000 --device auto`
  - 可选：如需手动指定配置文件，可加 `--config rl_algo/config/{ppo|grpo}.yaml`；默认会自动读取 `rl_algo/config/{algo}.yaml`（若存在）。
  - 若要强制使用 GPU：把 `--device auto` 改成 `--device cuda`；若机器不支持 CUDA，则用 `--device cpu`。
  - 若要跑更充分的对比：把 `--total-steps 80000` 改成 `--total-steps 200000`。

### A.3.2 一键对比命令（2算法×3种子 + 画图 + summary）
- `python scripts/run_cartpole_compare.py --total-steps 80000 --device auto`
  - 若要强制使用 GPU：`python scripts/run_cartpole_compare.py --total-steps 80000 --device cuda`
  - 若要跑更充分的对比：把 `--total-steps 80000` 改成 `--total-steps 200000`。

### A.3.3 输出产物
- 训练产物：`models/{run_name}/final.pt`、`models/{run_name}/config.yaml`、`models/{run_name}/train_log.csv`
- 汇总：`results/summary.json`
- 指标表（由日志计算）：`results/analysis_metrics_table.csv`
- 图表：`figs/*.png` 与 `figs/*.pdf`

## A.4 输入材料审计（必须据实）
本仓库根目录已包含输入材料：
- `ppo.pdf`（PPO 原论文；用于公式(2)等描述 [1]）
- `grpo.pdf`（DeepSeekMath 论文；包含 GRPO 章节与组内归一化优势描述 [2]）
- `大作业要求.jpg`（任务要求、指标定义、交付物、评分标准）
- `报告撰写模板.doc`（计算机学报风格 Word 模板）

## A.5 作业要求对照表（“要求→实现/文件”）

表A1 任务与交付物对照（来自 `大作业要求.jpg`）

| 要求条目 | 本仓库实现/产物 |
|---|---|
| PPO：Actor-Critic 架构 | `rl_algo/algos/ppo/agent.py`（ActorNet/CriticNet） |
| PPO：GAE 优势估计 | `rl_algo/train.py:add_gae_advantages_and_returns()` |
| PPO：PPO-Clip 目标 | `rl_algo/algos/ppo/agent.py:PPOAgent.update()`（ratio/clip/min） |
| GRPO：移除 Critic，仅 Actor | `rl_algo/algos/grpo/agent.py`（无 critic/value loss） |
| GRPO：group 优势（组内均值/标准差归一化） | `rl_algo/train.py:add_grpo_group_advantages()` |
| CartPole-v1 对比实验 | `scripts/run_cartpole_compare.py`（默认 env-id=CartPole-v1） |
| 超参公平性控制 | `rl_algo/config/ppo.yaml` vs `rl_algo/config/grpo.yaml`（公共/特有分离） |
| 三种子重复实验（1/3/43） | `scripts/run_cartpole_compare.py`（默认 seeds=[1,3,43]） |
| 指标：episodes-to-500 / wall-clock | `models/*/train_log.csv` + `results/summary.json` + `scripts/plot_report_figures.py` 计算/作图 |
| 可视化代码（出对比图） | `scripts/plot_report_figures.py`（>=10 张 PNG+PDF） |
| 打包提交 | 见 `checklist.md`：建议打包 `学号_姓名.zip`（按要求命名） |

## A.6 模板排版要点（Word 操作提示）
本文以 Markdown 草稿给出内容；实际提交 Word 时建议在 `报告撰写模板.doc` 中按模板要求完成排版。根据模板文件中的可见提示文本（例如 Title/Abstract/Key words、分级编号、IF/THEN 伪代码关键词、参考文献 [1] 格式等），需要重点检查：
- 章节编号：按 1、1.1、1.1.1 的层级编号
- 图表：先在正文引用“如图X所示/如表X所示”，再插入图表；图题/表题中文；坐标轴与图例清晰
- 伪代码：IF/THEN 大写；变量/函数名斜体（本文在 4.5 节已按该口径写）
- 参考文献：按引用顺序 [1][2]… 排列

## A.7 图表清单（写入正文；完整版见 `./fig_list.md`）
为满足“图表必须在正文先引用再出现、图例与坐标中文、同时输出 PNG+PDF”的要求，本文使用 `scripts/plot_report_figures.py` 统一生成图表。表A2列出报告使用的主要图号与文件路径（PNG 与 PDF 同名不同后缀）。

表A2 报告图表与文件路径

| 图号 | 图名（中文） | 输出路径（示例） |
|---|---|---|
| 图10 | 平均回合回报曲线（3 种子，均值±标准差） | `figs/fig10_cartpole_return_mean_std.png` |
| 图11 | 平均墙钟时间曲线（3 种子，均值±标准差） | `figs/fig11_cartpole_time_mean_std.png` |
| 图12(a) | PPO 单种子回合回报曲线（3 seeds） | `figs/fig12a_return_per_seed_ppo.png` |
| 图12(b) | GRPO 单种子回合回报曲线（3 seeds） | `figs/fig12b_return_per_seed_grpo.png` |
| 图13 | 平均回报的滑动平均曲线（MA10 / MA20） | `figs/fig13_return_moving_average.png` |
| 图14 | 回报标准差随回合变化（3 种子） | `figs/fig14_return_std_over_episode.png` |
| 图15 | 达到回报=500 的回合数（首次单回合） | `figs/fig15_episodes_to_500_bar.png` |
| 图16 | 达到回报=500 的墙钟时间（首次单回合） | `figs/fig16_time_to_500_bar.png` |
| 图17 | 平均回合回报曲线（steps 对齐，均值±标准差） | `figs/fig17_return_mean_std_vs_steps.png` |
| 图18(a) | PPO 单种子回合回报曲线（steps 横轴） | `figs/fig18a_ppo_seed_curves_vs_steps.png` |
| 图18(b) | GRPO 单种子回合回报曲线（steps 横轴） | `figs/fig18b_grpo_seed_curves_vs_steps.png` |
| 图19 | update 额外停顿随训练阶段变化 | `figs/fig19_update_extra_pause_by_stage.png` |
| 图20 | 训练吞吐曲线（steps/sec） | `figs/fig20_throughput_steps_per_sec.png` |

## A.8 提交前自检（写入正文；完整版见 `./checklist.md`）
提交前建议逐项核对：
- 能否单跑 PPO/GRPO（命令见 A.3.1）
- 是否满足 3 seeds（1/3/43）重复实验（命令见 A.3.2）
- 公共超参是否一致、特有超参是否独立可调（表4与配置文件）
- `models/*/train_log.csv` 字段是否齐全；`final.pt` 是否落盘
- `results/summary.json`、`figs/*.png`、`figs/*.pdf` 是否齐全

## A.9 评分标准提示（摘自 `大作业要求.jpg`）
根据要求截图中的评分标准，评估通常包含三部分：
- 代码实现（40%）：算法逻辑正确、可运行并收敛；代码规范与模块化设计
- 实验严谨性（20%）：多种子重复实验；控制变量、对比公平
- 报告质量（40%）：理论分析深度；对实验现象的分析是否深入；结构规范、图表清晰
