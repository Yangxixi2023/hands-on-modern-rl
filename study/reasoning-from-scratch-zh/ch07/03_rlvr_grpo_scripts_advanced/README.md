# 第 7 章：改进强化学习中的策略优化

本节包含高级 GRPO 脚本，在第 6 章实现的基础上加入额外的跟踪、稳定化和奖励建模变体。

&nbsp;
## 脚本概览

&nbsp;
### 主要脚本

- `7_3_plus_tracking.py`（*7.3 跟踪更高级的 GRPO 性能指标*）：跟踪额外的性能指标（优势统计量和熵）
- `7_4_plus_clip_ratio.py`（*7.4 使用裁剪策略比率稳定序列级 GRPO*）：与上面类似，但使用裁剪后的策略比率计算策略梯度损失
- `7_5_plus_kl.py`（*7.5 使用 KL 项控制模型变化程度*）：与上面类似，但增加 KL 损失项
- `7_6_plus_format_reward.py`（*7.6 添加显式格式奖励*）：与上面类似，但为 `<think>` 词元增加额外的格式奖励（与其他脚本的关键区别是，该奖励作用于推理模型而不是基础模型，因为正如主章节所述，推理模型已经熟悉这些词元）

<br>

&nbsp;
### GRPO 技巧与窍门补充脚本

GRPO 于 2024 年 4 月首次发布（[DeepSeekMath](https://arxiv.org/abs/2402.03300)），并于 2025 年 1 月随着 [DeepSeek-R1](https://arxiv.org/abs/2501.12948) 受到关注，此后文献提出了许多改进。下面列出一些较为重要的改进：

1. 零梯度信号过滤（[DAPO，Yu 等，2025](https://arxiv.org/abs/2503.14476)）
2. 主动采样（DAPO）
3. 词元级损失（DAPO）
4. 无 KL 损失（DAPO 与 [Dr. GRPO，Liu 等，2025](https://arxiv.org/abs/2503.20783)）
5. 提高裁剪上限（DAPO）
6. 截断重要性采样（[Yao 等，2025](https://fengyao.notion.site/off-policy-rl)）
7. 不进行标准差归一化（Dr. GRPO）
8. 使用领域特定的 KL 强度进行 KL 调整；数学任务设为零（[DeepSeek V3.2](https://arxiv.org/abs/2512.02556)）
9. 重新加权 KL（DeepSeek V3.2）
10. 离线策略序列掩码（DeepSeek V3.2）
11. 保留 top-p / top-k 的采样掩码（DeepSeek V3.2）
12. 保留原始 GRPO 优势归一化（DeepSeek V3.2）
13. 聚合前按每个奖励进行分组归一化（[GDPO，Liu 等，2026](https://arxiv.org/abs/2601.05242)）
14. 序列级重要性采样与裁剪（[GSPO，Zheng 等，2025](https://arxiv.org/abs/2507.18071)）
15. 裁剪重要性采样权重而不是词元更新（[CISPO，MiniMax 等，2025](https://arxiv.org/abs/2506.13585)）

（计划在完成主要内容后，找时间撰写更详细的说明。）

<br>

下面的脚本实现了其中一些改进：

- `7_7_improvements/olmo3_style.py`：在 [7_5_plus_kl.py](7_5_plus_kl.py) 基础上，实现类似 [Olmo 3](https://arxiv.org/abs/2512.13961) 的改进 1–7

- `7_7_improvements/deepseek_v32_style.py`：在 [7_5_plus_kl.py](7_5_plus_kl.py) 基础上，实现类似 [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) 的改进 8–12

- `7_7_improvements/gdpo.py`：在 [7_6_plus_format_reward.py](7_6_plus_format_reward.py) 基础上实现 [GDPO](https://arxiv.org/abs/2601.05242)（因为 GDPO 是针对多重奖励的调整）

---

**注意**：如果你不使用 `uv`，请在下面的示例中将 `uv run ...py` 替换为 `python ...py`。

---
&nbsp;
