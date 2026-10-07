# 优化版 Qwen3

本书使用的 Qwen3 从零实现，在高效（CPU 和 GPU 上都适用）、精简与易读性之间取得了平衡。

作为替代方案，你可以使用可选的 `Qwen3Model` 即插即用替代实现，它在 GPU 上的效率略高。 [`qwen3_optimized.py`](../../reasoning_from_scratch/qwen3_optimized.py) 中的优化版本（附录 C 会进一步讨论）与基线实现 [`qwen3.py`](../../reasoning_from_scratch/qwen3.py) 有两处关键差异：

- 使用 PyTorch 内置的 `torch.nn.functional.scaled_dot_product` 实现注意力，而不是自定义实现。
- 引入了修改后的 `KVCache`，预先分配键和值张量。这会增加内存使用，但可以避免执行过程中反复分配新的存储空间。


要了解两者的差异，建议并排打开 [`qwen3.py`](../../reasoning_from_scratch/qwen3.py) 和 [`qwen3_optimized.py`](../../reasoning_from_scratch/qwen3_optimized.py)，以及/或者查看文件差异：

<br>

![](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/optimized-LLM/vscode.webp)

<br>

&nbsp;
## 如何使用

如下所示，优化代码可以直接替换本章主要代码中的实现。

**替换前：**

```python
from reasoning_from_scratch.qwen3 import Qwen3Model
from reasoning_from_scratch.ch02 import generate_text_basic_stream_cache
```


**替换后：**

```python
from reasoning_from_scratch.qwen3_optimized import Qwen3Model
from reasoning_from_scratch.ch02 import generate_text_basic_stream_cache
```

&nbsp;
## 如何运行对比

要评估它在你系统上的性能，可以使用本目录中的 [`compare_inference.py`](compare_inference.py) 函数：

```python
python compare_inference.py
```

或者：

```python
uv run compare_inference.py
```

然后添加以下标志：

- `--device`：选择设备，例如 `cpu`、`mps` 或 `cuda`
- `--cache`：启用 KV 缓存
- `--compile`：使用 `torch.compile`
- `--reasoning`：使用 Qwen3 推理变体，而不是基础模型。基础模型针对给定提示生成约 50 个 token，推理变体生成约 2000 个 token。
- `--optimize`：使用 `qwen3_optimized.py` 中的优化模型，而不是 `qwen3.py` 中的标准模型。

<br>

&nbsp;
### 标准模型



| 模型    | 模式              | 命令                         | 硬件        | Token/秒    | GPU 内存（VRAM） |
| -------- | ----------------- | ------------------------------- | --------------- | ------------- | ----------------- |
| qwen3.py | 常规           | --device cpu                    | Mac Mini M4 CPU | 6             | -                 |
| qwen3.py | 常规编译  | --device cpu --compile          | Mac Mini M4 CPU | 6             | -                 |
| qwen3.py | KV 缓存          | --device cpu --cache            | Mac Mini M4 CPU | 28            | -                 |
| qwen3.py | KV 缓存编译 | --device cpu --compile --cache  | Mac Mini M4 CPU | 68            | -                 |
|          |                   |                                 |                 |               |                   |
| qwen3.py | 常规           | --device mps                    | Mac Mini M4 GPU | 17            | -                 |
| qwen3.py | 常规编译  | --device mps --compile          | Mac Mini M4 GPU | InductorError | -                 |
| qwen3.py | KV 缓存          | --device mps --cache            | Mac Mini M4 GPU | 18            | -                 |
| qwen3.py | KV 缓存编译 | --device mps --compile --cache  | Mac Mini M4 GPU | InductorError | -                 |
|          |                   |                                 |                 |               |                   |
| qwen3.py | 常规           | --device cuda                   | NVIDIA H100 GPU | 51            | 1.55 GB           |
| qwen3.py | 常规编译  | --device cuda --compile         | NVIDIA H100 GPU | 164           | 1.81 GB           |
| qwen3.py | KV 缓存          | --device cuda --cache            | NVIDIA H100 GPU | 48            | 1.52 GB           |
| qwen3.py | KV 缓存编译 | --device cuda --compile --cache | NVIDIA H100 GPU | 141           | 1.81 GB           |

<br>

&nbsp;
### 优化模型


| 模型              | 模式              | 命令                                     | 硬件        | Token/秒 | GPU 内存（VRAM） |
| ------------------ | ----------------- | ------------------------------------------- | --------------- | ---------- | ----------------- |
| qwen3_optimized.py | 常规           | --optimized --device cpu                    | Mac Mini M4 CPU | 5          | -                 |
| qwen3_optimized.py | 常规编译  | --optimized --device cpu --compile          | Mac Mini M4 CPU | 7          | -                 |
| qwen3_optimized.py | KV 缓存          | --optimized --device cpu --cache            | Mac Mini M4 CPU | 49         | -                 |
| qwen3_optimized.py | KV 缓存编译 | --optimized --device cpu --compile --cache  | Mac Mini M4 CPU | 51         | -                 |
|                    |                   |                                             |                 |            |                   |
| qwen3_optimized.py | 常规           | --optimized --device mps                    | Mac Mini M4 GPU | 21         | -                 |
| qwen3_optimized.py | 常规编译  | --optimized --device mps --compile          | Mac Mini M4 GPU | NameError  | -                 |
| qwen3_optimized.py | KV 缓存          | --optimized --device mps --cache            | Mac Mini M4 GPU | 29         | -                 |
| qwen3_optimized.py | KV 缓存编译 | --optimized --device mps --compile --cache  | Mac Mini M4 GPU | 38         | -                 |
|                    |                   |                                             |                 |            |                   |
| qwen3_optimized.py | 常规           | --optimized --device cuda                   | NVIDIA H100 GPU | 55         | 1.50 GB           |
| qwen3_optimized.py | 常规编译  | --optimized --device cuda --compile         | NVIDIA H100 GPU | 173        | 1.81 GB           |
| qwen3_optimized.py | KV 缓存          | --optimized --device cuda --cache           | NVIDIA H100 GPU | 56         | 5.85 GB           |
| qwen3_optimized.py | KV 缓存编译 | --optimized --device cuda --compile --cache | NVIDIA H100 GPU | 177        | 5.85 GB           |

<br>

对比上面的两张表可以看出，在大多数情况下，优化变体的 Token/秒明显更高。

不过请注意，在使用带 KV 缓存的编译版本时，未优化版本（68 token/秒）比优化版本（51 token/秒）更快。

优化版本还使用了更多基础 RAM（使用 KV 缓存时为 5.85 GB），而未优化版本为 1.5 GB。这是因为它会为支持的最大上下文长度预先分配保存 KV 值的张量。（因此，当未优化版本运行上下文长度为 41k 的提示时，RAM 使用量大致相当。）

**最佳建议可能是：使用 CPU 时采用未优化版本（带 `--cache` 和 `--compile`）；使用 GPU 时采用优化版本（带 `--cache` 和 `--compile`）。**
