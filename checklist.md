# 提交前自检清单（rl_algo：PPO vs GRPO，CartPole-v1）

> 说明：带 `[ ]` 的条目建议在最终打包前逐项勾选核对。

## 1 代码与运行
- [ ] `python -m rl_algo.train --algo ppo --env-id CartPole-v1 --config rl_algo/config/ppo.yaml --seed 1 --total-steps 200000 --eval-frequency 0` 可运行
- [ ] `python -m rl_algo.train --algo grpo --env-id CartPole-v1 --config rl_algo/config/grpo.yaml --seed 1 --total-steps 200000 --eval-frequency 0` 可运行
- [ ] `scripts/run_cartpole_compare.py` 可运行并完成 2 算法 × 3 seeds（1/3/43）

## 2 公平性与配置
- [ ] 公共超参一致：`gamma/hidden_dim/batch_episodes/actor_lr/update_epochs/minibatch_size/clip_coef/entropy_coef/max_grad_norm`
- [ ] 算法特有超参独立可调：PPO 的 `critic_lr`；GRPO 的 `group_size`
- [ ] `rl_algo/config/ppo.yaml` 与 `rl_algo/config/grpo.yaml` 已按要求填写并可被训练脚本读取

## 3 日志与模型产物
- [ ] 每次 run 输出目录存在：`models/{run_name}/`
- [ ] 模型文件存在：`models/{run_name}/best.pt` 与 `models/{run_name}/final.pt`
- [ ] 日志文件存在且字段完整：`models/{run_name}/train_log.csv`
  - [ ] `episode_idx`
  - [ ] `episodic_return`
  - [ ] `episodic_length`
  - [ ] `elapsed_sec`

## 4 汇总与出图（报告素材）
- [ ] 汇总文件存在：`results/summary.json`
- [ ] 报告图表一键生成：`python scripts/plot_report_figures.py --summary results/summary.json`
- [ ] `figs/` 下至少包含图1–图16对应的 PNG+PDF（见 `fig_list.md`）
- [ ] 图例/坐标轴为中文，且能在本机正常显示（若中文显示为方框，需补装字体或改用已安装中文字体）

## 5 报告草稿与清单文件（必须落盘）
- [ ] `./report_draft.md` 已生成（正文含图引用，且图先引用再出现）
- [ ] `./fig_list.md` 已生成（每张图含数据来源/生成命令/输出路径）
- [ ] `./checklist.md` 已生成

## 6 最终打包建议
- [ ] 打包命名：`学号_姓名.zip`（或按老师要求）
- [ ] 包含内容：
  - [ ] 代码（`rl_algo/`、`scripts/`、`setup.py` 等）
  - [ ] 配置（`rl_algo/config/*.yaml`）
  - [ ] 模型与日志（`models/`、`results/summary.json`）
  - [ ] 图表（`figs/*.png`、`figs/*.pdf`）
  - [ ] 报告草稿与清单（`report_draft.md`、`fig_list.md`、`checklist.md`）

