# 第 6 章：使用强化学习训练推理模型

&nbsp;

&nbsp;
## 补充材料

- [rlvr_grpo_original_no_kl.py](rlvr_grpo_original_no_kl.py)：实现原始 GRPO 算法的脚本，用带可验证奖励的强化学习（RLVR）训练推理模型。[DeepSeek R1](https://arxiv.org/abs/2501.12948) 使用了该算法，它最初由 [DeepSeekMath](https://arxiv.org/abs/2402.03300) 论文提出。不过，该脚本省略了 KL 散度项（[DAPO](https://arxiv.org/abs/2503.14476)、[Dr. GRPO](https://arxiv.org/abs/2503.20783)、[Olmo 3](https://arxiv.org/abs/2512.13961) 等工作也建议这样做）
  - KL 散度项确保训练后的模型不会偏离原始模型太远，但可能损害性能（尤其是在数学任务上）
  - 该脚本在概念上实现了与第 6 章相同的代码，不过包含两项小的性能调整：
    1. 移除 `torch.multinomial` 采样器中的 `.cpu()` 转换，使吞吐量提升 20%；有关具体实现，请参阅 [PR #178](https://github.com/rasbt/reasoning-from-scratch/pull/178)
    2. 使用 `--skip-zero-advantage-updates` 标志时，如果所有奖励都相等，就跳过模型更新。这会进一步加快训练，并可能降低内存需求（因为超过 `--max_new_tokens` 的长序列代价最高，而且通常会得到零奖励：在生成正确答案之前就已达到 token 上限，导致输出中不包含正确答案）；有关具体实现，请参阅 [PR #186](https://github.com/rasbt/reasoning-from-scratch/pull/186)
    - 如果你希望查看不包含上述两项改进的脚本，可以在[这里](https://github.com/rasbt/reasoning-from-scratch/blob/da009e41aacb17a433968cf84a4a6cf2a0fa4655/ch06/02_rlvr_grpo_scripts_intro/rlvr_grpo_original_no_kl.py)查看原始代码
- [rlvr_grpo_original_no_kl_batched.py](rlvr_grpo_original_no_kl_batched.py)：与上面相同，但支持批量训练。请注意，这会增加内存需求，因此可能需要减少 rollout 数量和 rollout 长度。用法与上面的脚本相同，只是增加了 `--num_batches`。
  - 请注意，与第 3 章的 [evaluate_math500_batched.py](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch03/02_math500-verifier-scripts/evaluate_math500_batched.py) 代码不同，这段代码无需从 [qwen3_batched.py](https://github.com/rasbt/reasoning-from-scratch/blob/main/reasoning_from_scratch/qwen3_batched.py) 导入 `Qwen3Model`；[PR #179](https://github.com/rasbt/reasoning-from-scratch/pull/179) 中提供了更详细的说明

- [rlvr_grpo_original_no_kl_batched_fsdp.py](rlvr_grpo_original_no_kl_batched_fsdp.py)：与上面相同，但支持使用 PyTorch 的 FSDP 在多个 GPU 上训练。如果可以使用多个 GPU，建议采用该脚本。用法与上面的脚本相同，只是增加了 `--num_gpus`。

这些脚本从 [`reasoning_from_scratch`](../../reasoning_from_scratch) 包导入了一些功能，以避免代码重复。（安装详情请参阅[第 2 章配置说明](../../ch02/02_setup-tips/python-instructions.md)。）不过，在这里，代码还重新实现了本章的核心函数，便于查看和修改。



<br>

---

**注意**：如果你不是 `uv` 用户，请将下面示例中的 `uv run ...py` 替换为 `python ...py`。

---


&nbsp;

|      | 方法                                 | 步数 | 最大 token 数 | Rollout 数量 | MATH-500 准确率 | 平均 token 数 |
| ---- | -------------------------------------- | ---- | ---------- | ------------ | ------------ | --------------- |
| 1    | 基础模型（第 3 章）                       | -    |            |              | 15.2%        | 78.85           |
| 2    | 推理模型（第 3 章）                  | -    |            |              | 48.2%        | 1369.79         |
| 3    | 原始 GRPO（第 7 章）              | 50   | 512        | 8            | 33.4%        | 910.33          |
| 4    | 原始 GRPO（第 7 章）              | 100  | 512        | 8            | 0.4%         | 1168.05         |
| 5    | 原始 GRPO，去掉 KL（本章） | 50   | 512        | 8            | 47.4%        | 586.11          |
| 6    | 原始 GRPO，去掉 KL（本章） | 100  | 512        | 8            | 44.0%        | 555.95          |
| 7    | GRPO 的 Olmo 3 修改版（第 7 章）           | 50   | 512        | 8            | 46.4%        | 601.61          |
| 8    | GRPO 的 Olmo 3 修改版（第 7 章）           | 100  | 512        | 8            | 45.4%        | 589.51          |
| 9    | GRPO 的 DeepSeek V3.2 修改版（第 7 章）    | 50   | 512        | 8            | 44.2%        | 618.49          |
| 10   | GRPO 的 DeepSeek V3.2 修改版（第 7 章）    | 100  | 512        | 8            | 45.2%        | 676.96          |

每 50 步保存一个检查点。如果用 KeyboardInterrupt 中断脚本，也会将最后一步保存为检查点。

请注意，为了降低所需计算内存、使训练更容易运行，训练最多只允许生成 512 个 token（即上表的最大 token 数）。

不过，评估脚本（使用与第 3 章相同的方法）允许最多生成 2048 个 token；上表的“平均 token 数”列衡量的是在 MATH-500 测试数据集上平均使用了多少 token。（训练使用 MATH 数据集里与 MATH-500 测试集不重叠的 12,000 个示例。更多详情请参阅 [https://github.com/rasbt/math_full_minus_math500](https://github.com/rasbt/math_full_minus_math500)。）

**第 1 行**

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model base
```

- 提示：可以在上面的代码执行命令中添加 `--show_eta`，以显示针对你机器估算的脚本总运行时间

**第 2 行**

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model reasoning
```

**第 3、4 行**

```bash
uv run ../../ch07/02_rlvr_grpo_scripts_advanced/rlvr_grpo_original.py \
--num_rollouts 8 \
--max_new_tokens 512
```

然后，针对生成的检查点运行 `evaluate_math500.py` 脚本来评估模型。例如：

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model base \
--checkpoint_path checkpoints/rlvr_grpo_original/qwen3-0.6B-rlvr-grpo-step00050.pth
```

**第 5、6 行**

```bash
uv run rlvr_grpo_original_no_kl.py \
--num_rollouts 8 \
--steps 100 \
--max_new_tokens 512
```

**第 7、8 行**

```bash
uv run ../../ch07/02_rlvr_grpo_scripts_original/rlvr_grpo_olmo3.py \
--num_rollouts 8 \
--max_new_tokens 512
```

**第 9、10 行**

```bash
uv run ../../ch07/02_rlvr_grpo_scripts_original/rlvr_grpo_deepseek_v32.py \
--num_rollouts 8 \
--max_new_tokens 512
```


<br>

如果 RAM 不足，可以考虑减少 rollout 数量（`--num_rollouts`）或回答长度（`--max_new_tokens`）。下表列出了部分资源需求，供参考。



| num_rollouts | max_new_tokens | 所需 RAM（GB） |
| ------------ | -------------- | ----------------- |
| 8            | 1024           | 30.50 GB          |
| 8            | 512            | 20.31 GB          |
| 8            | 256            | 15.60 GB          |
| 4            | 1024           | 14.60 GB          |
| 4            | 512            | 12.80 GB          |
| 4            | 256            | 10.59 GB          |


请注意，减少 token 数或 rollout 数量很可能会降低性能。如果 rollout 数量较少，可以将 `--accum_steps` 从 1 增加到 2 或 4（梯度累积），在一定程度上改善训练稳定性；不过，这会需要更多计算时间。

请注意，采用这些设置的原始（“vanilla”）GRPO 方法在训练超过 50 步时并不十分稳定。如果希望训练超过 50 步，可以考虑第 7 章的改进版本。


<br>

原始 GRPO 算法可以通过多种方式改善训练稳定性并提升训练效果，这也是[下一章](../../ch07)的主题。



&nbsp;
## 绘制训练曲线

可以使用 [plot_metrics.py](plot_metrics.py) 绘制 CSV 格式的训练记录。`logs` 文件夹中包含一次 200 步训练的示例记录（该日志使用默认设置生成，仅将 `--max_new_tokens 2048` 调高）：

```bash
uv run plot_metrics.py \
--csv logs/rlvr_grpo_original_no_kl_metrics.csv \
--moving_average 20
```

（`--moving_average 20` 设置会对之前 20% 的步数取平均，以获得更平滑的趋势线。）

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/ch06/other/plot.webp?1" width="600px">
