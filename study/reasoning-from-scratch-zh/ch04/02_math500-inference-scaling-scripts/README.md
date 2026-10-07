# 第 4 章：通过推理时扩展改进推理能力


&nbsp;
## 补充材料

- [cot_prompting_math500.py](cot_prompting_math500.py)：使用思维链提示在 MATH-500 数据集上评估模型的独立脚本
- [self_consistency_math500.py](self_consistency_math500.py)：使用自洽性采样在 MATH-500 数据集上评估模型的独立脚本
- [run_all_experiments_math500.sh](run_all_experiments_math500.sh)：用于运行下方本 README 中列出的全部实验（第 4 至第 12 行）的便捷 bash 脚本

两个评估脚本都从 [`reasoning_from_scratch`](../../reasoning_from_scratch) 包导入功能，以避免代码重复。（安装详情请参阅[第 2 章配置说明](../../ch02/02_setup-tips/python-instructions.md)。）



<br>

---

**注意**：如果你不是 `uv` 用户，请将下面示例中的 `uv run ...py` 替换为 `python ...py`。

---



&nbsp;

## 思维链提示

[`cot_prompting_math500.py`](self_consistency_math500.py) 脚本实现了第 4 章介绍的思维链提示方法。

&nbsp;

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/ch04/CH04_F04_raschka.webp" width=600>

&nbsp;

下表将这种方法（第 3 行）与第 3 章的基线进行比较：

|    | 方法                                       | 模型     | 准确率 | 时间       |
|----|----------------------------------------------|-----------|----------|------------|
| 1  | 基线（第 3 章），贪心解码        | 基础      | 15.2%    | 10.1 min   |
| 2  | 基线（第 3 章），贪心解码        | 推理 | 48.2%    | 182.1 min  |
| 3  | 思维链提示（“CoT”）           | 基础      | 40.6%    | 84.5 min   |

表中的准确率和运行时间是在 MATH-500 测试集的全部 500 个样本上，使用 “cuda” GPU（DGX Spark）计算得到的。

运行第 1 行实验：

```bash
python cot_prompting_math500.py \
--which_model "base" \
--dataset_size 500
```

或者使用 `uv`：


```bash
uv run cot_prompting_math500.py \
--which_model "base" \
--dataset_size 500
```

更多选项请使用 `--help` 标志。



&nbsp;
## 自洽性采样

[`self_consistency_math500.py`](self_consistency_math500.py) 脚本实现了第 4 章介绍的采样方法。

（可选地，还有一个 [`self_consistency_math500_batched.py`](self_consistency_math500_batched.py) 变体，会将全部 `--num_samples` 作为批次执行，从而加快处理速度。但请注意，这需要更多计算内存。）

&nbsp;

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/ch04/CH04_F17_raschka.webp" width=600>

&nbsp;

下表将这种方法（第 4–12 行）与第 3 章的基线（第 1–2 行）进行比较：

|      | 方法                                    | 模型     | 准确率 | 时间      |
| ---- | ----------------------------------------- | --------- | -------- | --------- |
| 1    | 基线（第 3 章），贪心解码     | 基础      | 15.2%    | 10.1 min  |
| 2    | 基线（第 3 章），贪心解码     | 推理 | 48.2%    | 182.1 min |
| 3    | 思维链提示（“CoT”）        | 基础      | 40.6%    | 84.5 min  |
| 4    | 温度和 top-p（“Top-p”）           | 基础      | 17.8%    | 30.7 min  |
| 5    | “Top-p” + 自洽性（n=3）          | 基础      | 29.6%    | 97.6 min  |
| 6    | “Top-p” + 自洽性（n=5）          | 基础      | 27.8%    | 116.8 min |
| 7    | “Top-p” + 自洽性（n=10）         | 基础      | 31.6%    | 300.4 min |
| 8    | “Top-p” + “CoT”                           | 基础      | 33.4%    | 129.2 min |
| 9    | 自洽性（n=3）+ “Top-p” + “CoT”  | 基础      | 42.2%    | 211.6 min |
| 10   | 自洽性（n=5）+ “Top-p” + “CoT”  | 基础      | 48.0%    | 452.9 min |
| 11   | 自洽性（n=10）+ “Top-p” + “CoT” | 基础      | 52.0%    | 862.6 min |
| 12   | 自洽性（n=3）+ “Top-p” + “CoT”  | 推理 | 55.2%    | 544.4 min |

表中的准确率和运行时间是在 MATH-500 测试集的全部 500 个样本上，使用 “cuda” GPU（DGX Spark）计算得到的。

以下代码说明如何运行第 4–12 行的自洽性实验（如果你不是 `uv` 用户，请将 `uv run` 替换为 `python`）。

**第 4 行：**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 1 \
    --dataset_size 500
```

**第 5 行：**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500
```

**第 6 行：**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 5 \
    --dataset_size 500
```

**第 7 行：**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 10 \
    --dataset_size 500
```

**第 8 行：**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 1 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
```

**第 9 行：**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
```

**第 10 行：**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 5 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
```

**第 11 行：**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 10 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
```

**第 12 行：**

```bash
uv run self_consistency_math500.py \
    --which_model "reasoning" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
```


更多选项请使用 `--help` 标志。
