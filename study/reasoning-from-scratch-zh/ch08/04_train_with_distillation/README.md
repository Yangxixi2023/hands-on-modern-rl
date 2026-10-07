# 第 8 章奖励材料：使用蒸馏训练

本文件夹包含一个简单的蒸馏脚本，用教师模型生成的推理轨迹训练 Qwen3 0.6B 模型，具体内容见第 8 章。

&nbsp;
## 文件

- [distill.py](distill.py)：使用 JSON 格式的蒸馏数据训练 Qwen3 0.6B（下一节会详细说明格式）。
  - 默认使用基础分词器训练基础模型
  - 如果传入 `--use_think_tokens`，则使用推理分词器，并在最终答案前将推理轨迹包在 `<think>...</think>` 中，方式与第 8 章类似
  - 每个 epoch 后，将检查点保存到 `checkpoints/distill/`，并将训练指标追加到 `logs/distill_metrics.csv`
  - 如果使用可选的 `--checkpoint_path` 从已有检查点初始化，而不是从基础模型初始化，则可以继续训练已有检查点
- [distill_batched.py](distill_batched.py)：上面脚本的批量版本。
  - 使用考虑填充的批量 Qwen3 实现，因此可以将不同长度的样例放在一起训练
  - 增加 `--batch_size` 参数，在每个优化步骤中处理多个样例
  - 将检查点保存到 `checkpoints/distill_batched/`，并将指标追加到 `logs/distill_batched_metrics.csv`
  - 需要注意，批量版本会使用多得多的 GPU 内存（取决于批量大小）

脚本从 [`reasoning_from_scratch`](../../reasoning_from_scratch) 包导入共享功能，以避免重复模型加载和提示词格式化代码。（安装详情请见[第 2 章设置说明](../../ch02/02_setup-tips/python-instructions.md)。）


<br>

---

**注意**：如果你不是 `uv` 用户，请将示例中的 `uv run ...py` 替换为 `python ...py`。

---


&nbsp;
## 输入数据格式

输入是 [`../02_generate_distillation_data`](../02_generate_distillation_data) 生成的 JSON 输出。每行应类似于：
```json
{
  "problem": "Compute 1/2 + 1/6.",
  "gtruth_answer": "2/3",
  "message_thinking": "I will rewrite the fractions with a common denominator.",
  "message_content": "The final answer is \\boxed{\\tfrac{2}{3}}."
}
```


训练时只使用以下字段：

- `problem`：插入第 3 章使用的同一个数学提示词模板
- `message_content`：必需；作为监督目标答案
- `message_thinking`：可选；如果存在，会放在 `message_content` 之前

缺少字段或字段格式错误的行会自动跳过；提示词加答案序列长度超过 `--max_seq_len` 的样例，会在训练/验证集拆分前被过滤。


&nbsp;
## 运行示例

如需快速进行健全性检查，可以在前一个文件夹生成的小样例上训练：
```bash
uv run distill.py \
  --data_path ../02_generate_distillation_data/sample_openrouter_outputs.json \
  --dataset_size 5 \
  --validation_size 1 \
  --epochs 2 \
  --log_every 1
```


执行过程会：

- 加载 Qwen3 0.6B 基础权重
- 对提示词/答案对进行分词
- 留出 1 个样例用于验证
- 每个 epoch 后将检查点保存到 `checkpoints/distill/`
- 将 CSV 指标写入 `logs/distill_metrics.csv`

如果想使用显式推理标签和推理分词器进行训练，请添加 `--use_think_tokens`：
```bash
uv run distill.py \
  --data_path ../02_generate_distillation_data/sample_openrouter_outputs.json \
  --dataset_size 5 \
  --validation_size 1 \
  --epochs 2 \
  --log_every 1 \
  --use_think_tokens
```


如果希望改为批量训练，请运行：
```bash
uv run distill_batched.py \
  --data_path ../02_generate_distillation_data/sample_openrouter_outputs.json \
  --dataset_size 5 \
  --validation_size 1 \
  --epochs 2 \
  --batch_size 2 \
  --log_every 1
```


&nbsp;
## 有用的选项
```bash
uv run distill.py --help
```


重要参数：

- `--data_path`：蒸馏 JSON 文件路径
- `--dataset_size`：拆分前截断数据集（`0` 表示使用全部行）
- `--validation_size`：验证样例的绝对数量
- `--epochs`：遍历训练集的次数
- `--batch_size`：`distill_batched.py` 每个优化步骤处理的样例数量
- `--lr`：AdamW 学习率
- `--max_seq_len`：丢弃提示词加答案序列超过该长度限制的样例
- `--checkpoint_path`：从之前的蒸馏检查点初始化
- `--grad_clip_norm`：可选的梯度裁剪
- `--use_think_tokens`：切换到推理分词器和 `<think>...</think>` 格式

下面“实验”一节提供了实际操作示例。

&nbsp;

## 评估蒸馏检查点

训练完成后，可以使用第 3 章的评估脚本在 MATH-500 上评估检查点。

如果训练时没有使用 `--use_think_tokens`，请将它作为 `base` 模型评估：
```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
  --dataset_size 500 \
  --which_model base \
  --checkpoint_path checkpoints/distill/qwen3-0.6B-distill-step00004-epoch1.pth
```


**重要：** 如果训练时使用了 `--use_think_tokens`，请将它作为 `reasoning` 模型评估，以使用推理分词器：
```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
  --dataset_size 500 \
  --which_model reasoning \
  --checkpoint_path checkpoints/distill/qwen3-0.6B-distill-step00004-epoch1.pth
```


&nbsp;
## 实验

第 8 章使用的蒸馏数据集位于我的 Hugging Face 仓库 [rasbt/math_distill](https://huggingface.co/datasets/rasbt/math_distill)。第 8 章通过一个下载分区的辅助函数加载这些数据，例如：
````python
from reasoning_from_scratch.ch08 import load_distill_data

_ = load_distill_data(
    partition="deepseek-r1-math-train.json",
    local_path="deepseek-r1-math-train.json"
)
_ = load_distill_data(
    partition="qwen3-235b-a22b-math-train.json",
    local_path="qwen3-235b-a22b-math-train.json"
)
```



下面的实验使用了该数据集中的 `deepseek-r1-math-train.json` 和 `qwen3-235b-a22b-math-train.json` 文件。


&nbsp;

|      | 教师数据                         | Epoch | MATH-500 准确率 | 最终验证损失 |
| ---- | -------------------------------- | ----- | --------------- | ------------ |
| 1    | 基础模型（第 3 章）              | -     | 15.2%           | -            |
| 2    | 推理模型（第 3 章）              | -     | 48.2%           | -            |
| 3    | DeepSeek R1 蒸馏数据             | 1     | 30.6%           | 0.5436       |
| 4    | DeepSeek R1 蒸馏数据             | 2     | 32.4%           | 0.5349       |
| 5    | DeepSeek R1 蒸馏数据             | 3     | 33.6%           | 0.5343       |
| 6    | Qwen3 235B A22B 蒸馏数据         | 1     | 45.0%           | 0.4043       |
| 7    | Qwen3 235B A22B 蒸馏数据         | 2     | 43.8%           | 0.3963       |
| 8    | Qwen3 235B A22B 蒸馏数据         | 3     | 44.2%           | 0.3948       |

在 H100 上训练约需 30 分钟，在 DGX Spark 上约需 3 小时，最多使用 15 GB 内存。

下面给出复现表格结果的代码片段。

&nbsp;
**第 1 行**
```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model base
```


&nbsp;
**第 2 行**
```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model reasoning
```


&nbsp;
**第 3、4、5 行**
```bash
uv run distill.py \
--data_path deepseek-r1-math-train.json \
--validation_size 25 \
--epochs 3 \
--lr 1e-5 \
--max_seq_len 2048 \
--use_think_tokens \
--grad_clip 1.0
```

然后，要评估各 epoch 的检查点，请运行：

&nbsp;
```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model reasoning \
--max_new_tokens 4096 \
--checkpoint_path run-1/checkpoints/distill/qwen3-0.6B-distill-step06682-epoch1.pth
```

第 4 行和第 5 行分别将检查点路径替换为 `...step13364-epoch2.pth` 和 `...step20046-epoch3.pth`。

&nbsp;
**第 6、7、8 行**
```bash
uv run distill.py \
--data_path qwen3-235b-a22b-math-train.json \
--validation_size 25 \
--epochs 3 \
--lr 1e-5 \
--max_seq_len 2048 \
--use_think_tokens \
--grad_clip 1.0
```

然后，要评估各 epoch 的检查点，请运行：
```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model reasoning \
--max_new_tokens 4096 \
--checkpoint_path run_11/checkpoints/distill/qwen3-0.6B-distill-step05746-epoch1.pth
```

第 7 行和第 8 行分别将检查点路径替换为 `...step11492-epoch2.pth` 和 `...step17238-epoch3.pth`。
