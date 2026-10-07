# 第 5 章：通过自我细化进行推理时扩展


&nbsp;
## 补充材料

- [self_refinement_math500.py](self_refinement_math500.py)：使用自我细化在 MATH-500 数据集上评估模型的独立脚本

该脚本从 [`reasoning_from_scratch`](../../reasoning_from_scratch) 包导入功能，以避免代码重复。（安装详情请参阅[第 2 章配置说明](../../ch02/02_setup-tips/python-instructions.md)。）



<br>

---

**注意**：如果你不是 `uv` 用户，请将下面示例中的 `uv run ...py` 替换为 `python ...py`。

---



&nbsp;

## 自我细化

[`self_refinement_math500.py`](self_refinement_math500.py) 脚本实现了第 5 章介绍的自我细化方法。


&nbsp;

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/ch05/CH05_F21_raschka.webp" width=600>

&nbsp;



| #  | 方法          | 评分器    | 迭代次数 | 模型     | 准确率 | 时间      |
|----|-----------------|-----------|------------|-----------|----------|-----------|
| 1  | 基线（ch03） | -         | -          | 基础      | 15.2%    | 10.1 min  |
| 2  | 自我细化 | None      | 1          | 基础      | 25.0%    | 84.8 min  |
| 3  | 自我细化 | None      | 2          | 基础      | 22.0%    | 165.4 min |
|    |                 |           |            |           |          |           |
| 4  | 自我细化 | Heuristic | 1          | 基础      | 21.6%    | 84.7 min  |
| 5  | 自我细化 | Heuristic | 2          | 基础      | 20.8%    | 151.4 min |
|    |                 |           |            |           |          |           |
| 6  | 自我细化 | Logprob   | 1          | 基础      | 21.4%    | 85.3 min  |
| 7  | 自我细化 | Logprob   | 2          | 基础      | 22.0%    | 165.3 min |
|    |                 |           |            |           |          |           |
| 8  | 自我细化 | Logp-ex   | 1          | 基础      | 20.4%    | 85.0 min  |
| 9  | 自我细化 | Logp-ex   | 2          | 基础      | 21.2%    | 160.2 min |
|    |                 |           |            |           |          |           |
| 10 | 基线（ch03） | -         | -          | 推理      | 48.2%    | 182.1 min |
| 11 | 自我细化 | None      | 1          | 推理      | 56.6%    | 498.8 min |
| 12 | 自我细化 | None      | 2          | 推理      | 56.6%    | 713.9 min |
|    |                 |           |            |           |          |           |
| 13 | 自我细化 | Heuristic | 1          | 推理      | 57.8%    | 498.6 min |
| 14 | 自我细化 | Heuristic | 2          | 推理      | 57.8%    | 713.9 min |
|    |                 |           |            |           |          |           |
| 15 | 自我细化 | Logprob   | 1          | 推理      | 48.4%    | 499.7 min |
| 16 | 自我细化 | Logprob   | 2          | 推理      | 48.6%    | 753.0 min |

表中的准确率和运行时间是在 MATH-500 测试集的全部 500 个样本上，使用 “cuda” GPU（DGX Spark）计算得到的。

以下代码说明如何运行第 4–12 行的自洽性实验（如果你不是 `uv` 用户，请将 `uv run` 替换为 `python`）。

**第 2 行：**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "none"
```

**第 3 行：**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "none"
```

**第 4 行：**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "heuristic"
```

**第 5 行：**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "heuristic"
```

**第 6 行：**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "logprob"
```

**第 7 行：**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "logprob"
```

**第 8 行：**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "logprob_extract"
```

**第 9 行：**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "logprob_extract"
```

**第 11 行：**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "none"
```

**第 12 行：**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "none"
```

**第 13 行：**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "heuristic"
```

**第 14 行：**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "heuristic"
```

**第 15 行：**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "logprob"
```

**第 16 行：**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "logprob"
```




&nbsp;

## 使用评分器打破平局的自洽性

[`self_consistency_scorer_math500.py`](self_consistency_scorer_math500.py) 在第 5 章实现的评分器基础上扩展了自洽性方法，用于打破平局。


&nbsp;

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/appendix-b/majority-vote.webp" width=600>

&nbsp;



|   | 方法                                   | 模型 | 准确率 | 时间      |
|---|------------------------------------------|-------|----------|-----------|
| 1 | 第 4 章使用 CoT 提示的基线    | 基础  | 33.4%    | 129.2 min |
| 2 | 自洽性（n=3）+ 多数投票   | 基础  | 43.2%    | 328.2 min |
| 3 | 自洽性（n=3）+ 启发式       | 基础  | 43.4%    | 326.5 min |
| 4 | 自洽性（n=3）+ 平均 logprob    | 基础  | 44.8%    | 327.7 min |


表中的准确率和运行时间是在 MATH-500 测试集的全部 500 个样本上，使用 “cuda” GPU（DGX Spark）计算得到的。

以下代码说明如何运行第 2–4 行的自洽性实验（如果你不是 `uv` 用户，请将 `uv run` 替换为 `python`）。

**第 2 行：**

```bash
uv run self_consistency_scorer_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step." \
    --scoring "none"
```

**第 3 行：**

```bash
uv run self_consistency_scorer_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step." \
    --scoring "heuristic"
```

**第 4 行：**

```bash
uv run self_consistency_scorer_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step." \
    --scoring "logprob"
```

&nbsp;

## N 选最佳

[`self_consistency_scorer_math500.py`](self_consistency_scorer_math500.py) 实现了 N 选最佳的推理时扩展方法。

N 选最佳与自洽性类似，都会生成多个答案。但它不是通过多数投票选择最终答案，而是使用评分函数为所有生成的答案评分。

[`best_of_n_math500.py`](best_of_n_math500.py) 在第 5 章实现的评分器基础上扩展了自洽性方法，用于打破平局。


&nbsp;

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/appendix-b/best-of-n.webp" width=600>

&nbsp;

|   | 方法                                   | 模型 | 准确率 | 时间      |
|---|------------------------------------------|-------|----------|-----------|
| 1 | 使用思维链提示的基线 | 基础  | 33.4%    | 129.2 min |
| 2 | N 选最佳（n=3）+ 启发式              | 基础  | 40.6%    | 327.7 min |
| 3 | N 选最佳（n=3）+ 平均 logprob           | 基础  | 43.2%    | 330.2 min |


表中的准确率和运行时间是在 MATH-500 测试集的全部 500 个样本上，使用 “cuda” GPU（DGX Spark）计算得到的。

以下代码说明如何运行第 2、3 行的自洽性实验（如果你不是 `uv` 用户，请将 `uv run` 替换为 `python`）。

**第 2 行：**

```bash
uv run best_of_n_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
    --scoring "heuristic"
)
```

**第 3 行：**

```bash
uv run best_of_n_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
    --scoring "logprob"
)
```
s
