# 第 3 章：评估推理模型

&nbsp;


&nbsp;
## 补充材料

- [evaluate_math500.py](evaluate_math500.py)：在 MATH-500 数据集上评估模型的独立脚本
- [evaluate_math500_batched.py](evaluate_math500_batched.py)：与上面相同，但在生成过程中并行处理多个示例（以获得更高吞吐量）
- [evaluate_json.py](evaluate_json.py)：评估已保存的 JSON/JSONL 记录文件并报告准确率

两个评估脚本都从 [`reasoning_from_scratch`](../../reasoning_from_scratch) 包导入功能，以避免代码重复。（安装详情请参阅[第 2 章配置说明](../../ch02/02_setup-tips/python-instructions.md)。）



<br>

---

**注意**：如果你不是 `uv` 用户，请将下面示例中的 `uv run ...py` 替换为 `python ...py`。

---



&nbsp;

## `evaluate_math500.py` 的使用方法

运行：

```bash
python evaluate_math500.py
```

或者使用 `uv`：


```bash
uv run evaluate_math500.py
```

选项：

```bash
uv run evaluate_math500.py --help

options:
  -h, --help            show this help message and exit
  --device DEVICE       Device to use: "auto" (default) or any torch device string
                        (e.g., "cpu", "cuda", "cuda:0", "mps").
  --which_model {base,reasoning}
                        Model variant to load (default: "base").
  --dataset_size DATASET_SIZE
                        Number of MATH-500 examples to evaluate (default: 10).
  --max_new_tokens MAX_NEW_TOKENS
                        Max new tokens to generate (default: 2048).
  --compile             Enable torch.compile.
  --verbose             Print per-sample correctness while evaluating.
```

&nbsp;
## `evaluate_math500_batch.py` 的使用方法

这个版本将批处理扩展到了生成过程本身，从而可以并行解码：

```bash
uv run evaluate_math500_batched.py --help
```

额外选项：

```bash
  --batch_size BATCH_SIZE
                        Number of examples to generate in parallel (default: 4).
  --disable_efficient_mode
                        Use a simpler batched inference method. Slower and more
                        memory-intensive, but easier to debug.
```


&nbsp;


**实现说明：**
默认情况下，批量生成会在序列输出停止标记后停止对应序列。使用 `--disable_efficient_mode` 时，所有序列会一直运行到最长序列结束。这只影响计算效率，不影响定性结果，因为停止标记之后的 token 会被丢弃。

&nbsp;

**提示（MPS 设备）：**
运行：

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run evaluate_math500_batched.py
```

高效批量推理使用的某些 PyTorch 算子目前尚未在 MPS 上受支持。作为替代方案，也可以使用 `--disable_efficient_mode`。



&nbsp;

- `evaluate_math500.py --dataset_size 500`


| 设备 / 数据集大小                       | 基础模型 | 推理模型 |
| ------------------------------------------- | ---------- | --------------- |
| **Mac Mini M4 CPU**（500 个示例，顺序执行） | 43.6 min | 未运行（温度过高）           |
| **Mac Mini M4 GPU**（500 个示例，顺序执行） | 37.5 min | 未运行（温度过高） |
| **DGX Spark**（500 个示例，顺序执行） | 10.0 min  | 182.2 min      |
| **H100 GPU**（500 个示例，顺序执行） | 13.3 min  | 185.4 min      |

<br>
<br>

- `evaluate_math500_batched.py --dataset_size 500 --batch_size 128`

| 设备 / 数据集大小                                        | 基础模型 | 推理模型 |
| ------------------------------------------------------------ | ---------- | ---------- |
| **Mac Mini M4 CPU**（500 个示例，批处理，`--batch_size 128`） | 167.2 min | 未运行（温度过高）           |
| **Mac Mini M4 GPU**（500 个示例，批处理，`--batch_size 128`） | 错误*     | 错误           |
| **DGX Spark**（500 个示例，批处理，`--batch_size 128`）    | 16.3 min  | 119.3 min      |
| **H100 GPU**（500 个示例，批处理，`--batch_size 128`）     | 3.3 min   | 14.6 min       |



- 基础模型的准确率为 15.6%（78/500）；推理模型的准确率为 48.2%（241/500）。


&nbsp;
## `evaluate_json.py` 的使用方法

如果已经保存了记录，只想（重新）计算准确率，可以使用此脚本：

```bash
uv run evaluate_json.py --json_path math500_base-mps-evaluate-script.jsonl
# Accuracy 15.6% (78/500)

Optional keys:

```bash
uv run evaluate_json.py \
  --json_path my_records.json \
  --gtruth_answer "gtruth_answer" \
  --generated_text "generated_text"
```
