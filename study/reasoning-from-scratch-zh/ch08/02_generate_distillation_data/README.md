# 第 8 章奖励材料：生成蒸馏数据

本文件夹包含用于为数学题生成教师模型输出的脚本。正如第 8 章所述，这些输出可以作为训练较小推理模型的蒸馏数据。

&nbsp;
**目录：**

- [文件](#files)
- [输入数据格式](#input-data-format)
- [输出格式](#output-format)
- [1. 使用 Ollama 本地生成](#1-local-generation-with-ollama)
  - [1.1 Ollama 设置](#11-ollama-setup)
  - [1.2 使用 Ollama 本地生成数据](#12-local-data-generation-with-ollama)
  - [1.3 Ollama 排障](#13-ollama-troubleshooting)
    - [1.3.1 Ollama 未运行](#131-ollama-not-running)
    - [1.3.2 Ollama 模型尚未下载](#132-ollama-model-not-downloaded)
- [2. 使用 OpenRouter 托管生成](#2-hosted-generation-with-openrouter)
  - [2.1 OpenRouter 设置](#21-openrouter-setup)
  - [2.2 使用 OpenRouter 生成数据](#22-data-generation-with-openrouter)
- [用于蒸馏的数据集](#datasets-for-distillation)
- [数据集统计](#dataset-statistics)
- [教师模型准确率](#teacher-accuracy)
- [生成 MATH-500 蒸馏数据集](#generating-a-math-500-distillation-dataset)
- [生成包含 12,000 道 MATH 题目的蒸馏数据集](#generating-a-distillation-dataset-of-12000-math-samples)


&nbsp;
## 文件

- [average_field_lengths_json.py](average_field_lengths_json.py)：打印生成数据集基本统计信息的工具脚本。
- [generate_with_ollama.py](generate_with_ollama.py)：使用 Ollama 生成用于蒸馏的模型答案。如果想从可在本地运行的较小模型进行蒸馏，例如 Qwen3 4B、gpt-oss 20B、DeepSeek R1 32B 等，推荐使用此脚本。
- [generate_with_openrouter.py](generate_with_openrouter.py)：通过 OpenRouter API 使用模型生成用于蒸馏的模型答案。如果想使用 DeepSeek R1（671B）或 Kimi K2.5（1T）等无法在本地运行的大模型，推荐使用此脚本。
- [math_train_sample.json](math_train_sample.json)：用于快速健全性检查的小型示例数据集。

&nbsp;
## 输入数据格式

两个脚本都通过 `--math_json` 接收 JSON 文件。每个对象至少应包含：

- `problem`（字符串）：数学问题。
- `answer`（字符串）：真实答案。

`level`、`type`、`unique_id` 等额外键会被忽略。可以查看 [math_train_sample.json](math_train_sample.json) 文件了解示例结构；该结构基于我们在第 6、7、8 章使用的 [math_full_minus_math500.json](https://github.com/rasbt/math_full_minus_math500/blob/main/math_full_minus_math500.json)。

若要应用到完整的 12,000 个样例，只需下载 [math_full_minus_math500.json](https://github.com/rasbt/math_full_minus_math500/blob/main/math_full_minus_math500.json)，并通过 `--math_json math_full_minus_math500.json` 传给脚本。注意，这会花费很长时间，因此建议先将文件截断为几百或一千个样例。


&nbsp;
## 输出格式

两个脚本都会写出一个 JSON 数组，其中每一行类似于：
```
{
  "problem": "...",             # The original "problem"
  "gtruth_answer": "...",       # The original "answer"
  "message_thinking": "...",    # The model's thinking stream
  "message_content": "..."      # The model's final answer
}
```


说明：

- 输入 JSON 文件中的原始 `"answer"` 被重命名为 `"gtruth_answer"`，以避免歧义（因为“answer”是一个通用术语，也可能指模型的答案）。
- 文件会在每个样例完成后增量写入，因此可以使用中间文件，或中断运行。
- 脚本提供 `--resume` 选项，可以继续执行被中断的运行。

&nbsp;
## 1. 使用 Ollama 本地生成


- Ollama 是一个高效运行 LLM 的开源应用。
- 它是 llama.cpp（[https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)）的封装，llama.cpp 使用纯 C/C++ 实现 LLM，以最大化效率。
- 注意，它是用于让 LLM 生成文本（推理）的工具，不用于训练或微调 LLM。

&nbsp;
### 1.1 Ollama 设置


- 运行下面的代码前，请访问 [https://ollama.com](https://ollama.com) 安装 ollama，并按说明操作（例如点击“Download”按钮，下载适合你操作系统的 ollama 应用）。
- macOS 和 Windows 用户点击下载的 ollama 应用；如果系统提示安装命令行用法，请选择“是”。
- Linux 用户可以使用 ollama 网站提供的安装命令。
- 在计算机上运行 ollama 有 3 种方式：


&nbsp;
**1. `ollama serve`**

- 这会将 ollama 后端作为服务器运行，通常地址为 `http://localhost:11434`。在通过 API 调用之前，它不会加载模型。如果想通过 Python 使用 ollama，就采用这种方式。

&nbsp;
**2. `ollama run deepseek-r1:8b`**

- 这是一个便捷封装。如果服务器尚未运行，它会启动服务器，然后（第一次运行时）下载模型，并打开交互式终端，让我们与模型聊天。在后台，它使用相同的服务器 API。
- `deepseek-r1:8b` 模型在 `--max_new_tokens 8192` 设置下约需要 30 GB 内存。
  - 如果内存更多，建议尝试更大的模型以获得更高质量的答案，例如 `deepseek-r1:32b`（约需要 60 GB）。
  - 如果内存较少，可以选择更小的模型；[这里](https://ollama.com/library/deepseek-r1) 有较小 R1 模型的列表。除了 DeepSeek 模型，也可以使用 [Ollama 网站](https://ollama.com/)上的“Search model”字段，选择其他感兴趣的模型。
  - 还可以将 `--max_new_tokens 8192` 改为 `--max_new_tokens 2048`，以降低内存使用，但这可能会过早截断一些答案。

&nbsp;
**3. Ollama 桌面应用**

- 这会自动运行同一个后端，并在其上提供图形界面（如上图所示）。它还会应用默认设置（系统提示词、温度、停止序列），这可以解释为什么回答看起来与直接使用 API 时不同。

&nbsp;
### 1.2 使用 Ollama 本地生成数据
```bash
uv run generate_with_ollama.py \
  --math_json math_train_sample.json \
  --dataset_size 5 \
  --model deepseek-r1:8b \
  --max_new_tokens 8192 \
  --out_file sample_ollama_outputs.json
```

如果你不是 `uv` 用户，请将 `uv run` 替换为 `python`。

预期输出如下：
```
Loading model: deepseek-r1:8b
Using CUDA:0
Model ready
5/5 | MATH-500: 5/5 | ETA: 00s
Total time: 3.2 min

Wrote 5 rows to: /home/rasbt/reasoning-from-scratch-codedev/ch08/sample_ollama_outputs.json
```

生成的 [sample_ollama_outputs.json](sample_ollama_outputs.json) 文件中的条目如下：
```json
  {
    "problem": "A rectangular band formation...",
    "gtruth_answer": "98",
    "message_thinking": "I need to find the largest number of...",
    "message_content": "The function is continuous..."
  },
```

`"message_thinking"` 字段包含思维链解释，`"message_content"` 包含最终答案。例如，可以将两者连接为
```python
complete_answer = f"<think>{data['message_thinking']}</think>\n\n{data['message_content']}"
```

也就是：
```
"<think>I need to find the largest number of...</think>

The function is continuous..."
```

&nbsp;
### 1.3 Ollama 排障

下面列出运行 Ollama 数据生成脚本时的一些常见问题。

&nbsp;
#### 1.3.1 Ollama 未运行

如果看到类似下面的错误：
```
Loading model: deepseek-r1:32b
Using CUDA:0
Traceback (most recent call last):
  File "/home/rasbt/reasoning-from-scratch-codedev/ch08/generate_with_ollama.py", line 379, in <module>
    query_ollama_chat(
  File "/home/rasbt//reasoning-from-scratch-codedev/ch08/generate_with_ollama.py", line 235, in query_ollama_chat
    raise RuntimeError(
RuntimeError: Failed to query Ollama after 3 attempt(s). Last error: <urlopen error [Errno 111] Connection refused>
```

请确保 `ollama serve` 正在运行（在另一个终端标签页中）。

&nbsp;
#### 1.3.2 Ollama 模型尚未下载

如果看到以下错误：
```
Loading model: deepseek-r1:8b
Using CUDA:0
Traceback (most recent call last):
  File "/home/rasbt/reasoning-from-scratch-codedev/ch08/generate_with_ollama.py", line 379, in <module>
    query_ollama_chat(
  File "/home/rasbt/reasoning-from-scratch-codedev/ch08/generate_with_ollama.py", line 235, in query_ollama_chat
    raise RuntimeError(
RuntimeError: Failed to query Ollama after 3 attempt(s). Last error: HTTP 404 from Ollama at http://localhost:11434/api/chat: {"error":"model 'deepseek-r1:8b' not found"}
```

这表示模型尚未下载。在这种情况下，请在单独的终端中运行 `ollama run deepseek-r1:8b`；它会下载模型并启动聊天会话。你可以在聊天中试用模型，然后通过 `\bye` 退出。


&nbsp;
## 2. 使用 OpenRouter 托管生成

如果想在本地运行模型，Ollama 很方便。不过，有些大模型（例如 671B 参数的 DeepSeek R1）太大，无法在我们的硬件上本地运行。对于这些情况，推荐使用 [OpenRouter](https://openrouter.ai)：它通过类似 ChatGPT 的 API，在云端托管大量开放权重模型和专有 LLM。

截至本文撰写时，[DeepSeek R1](https://openrouter.ai/deepseek/deepseek-r1) 的价格是每 100 万个输入 token 0.70 美元，每 100 万个输出 token 2.50 美元。注意，OpenRouter 上还有许多更便宜（也更快）的模型；甚至较新的 [DeepSeek V3.2](https://openrouter.ai/deepseek/deepseek-v3.2) 每 100 万个输出 token 只需 0.40 美元。

下面进行一个简单的成本计算。假设平均输入提示词长度为 11 个 token、平均回答长度为 1524 个 token，那么生成 1000 道 MATH 题的答案约需 3.82 美元。

具体计算如下：

- 输入 token 总数：11 × 1000 = 11,000
- 输出 token 总数：1524 × 1000 = 1,524,000
- 输入费用：`(11,000 / 1,000,000) × $0.70 = $0.0077`
- 输出费用：`(1,524,000 / 1,000,000) × $2.50 = $3.81`
- 总费用：`$3.81 + $0.0077 ≈ $3.82`

&nbsp;
### 2.1 OpenRouter 设置

设置非常简单。只需在 [OpenRouter](https://openrouter.ai/) 创建账户，在 [https://openrouter.ai/settings/keys](https://openrouter.ai/settings/keys) 的账户设置中生成 API 密钥，并将 API 密钥安全保存（例如保存在密码管理器中）。


&nbsp;
### 2.2 使用 OpenRouter 生成数据

OpenRouter 脚本的工作方式与 Ollama 脚本类似，只是要先将 API 密钥设置为环境变量：
```bash
OPENROUTER_API_KEY="YOUR_API_KEY" uv run generate_with_openrouter.py \
  --math_json math_train_sample.json \
  --dataset_size 5 \
  --model deepseek/deepseek-r1 \
  --num_processes 1 \
  --out_file sample_openrouter_outputs.json
```

如果你不是 `uv` 用户，请将 `uv run` 替换为 `python`。

输出如下：
```
Loading model: deepseek/deepseek-r1
Using OpenRouter API: https://openrouter.ai/api/v1/chat/completions
Model ready
5/5 | MATH-500: 5/5 | ETA: 00s
Total time: 2.2 min

Wrote 5 rows to: /Users/sebastian/Developer/reasoning-from-scratch/ch08/02_generate_distillation_data/sample_openrouter_outputs.json
```

[sample_openrouter_outputs.json](sample_openrouter_outputs.json) 输出文件的结构与 Ollama 脚本生成的文件相同。

**提示：** 如果要生成大量数据，按顺序运行这个蒸馏过程可能非常慢（例如使用 DeepSeek R1 生成 12,000 个答案约需 100 小时）。此时建议通过 `--num_processes` 运行多个并行的数据生成线程。例如，对 DeepSeek R1 模型使用 `--num_processes 50`，可以将运行时间从 100 小时缩短到约 2 小时。


&nbsp;
## 用于蒸馏的数据集

通过上述 OpenRouter 方法生成的一组数据集位于：[https://huggingface.co/datasets/rasbt/math_distill](https://huggingface.co/datasets/rasbt/math_distill)。

&nbsp;
## 数据集统计

要查看数据集统计信息，请使用 [average_field_lengths_json.py] 脚本：
```bash
uv run average_field_lengths_json.py \
--json_path sample_openrouter_outputs.json
```

```
tokenizer-reasoning.json: 100% (10 MiB / 10 MiB)
Records: 5
Tokenizer: reasoning
Field             AvgTokens  MinTokens  MaxToken  Count
gtruth_answer          9.40          9        10      5
message_content      196.00        166       259      5
message_thinking     933.20        449      1676      5
problem               77.80         30       121      5
```

&nbsp;
## 教师模型准确率

要计算生成该数据集的模型（即教师模型）的准确率，请使用 [../../ch03/02_math500-verifier-scripts/evaluate_json.py](../../ch03/02_math500-verifier-scripts/evaluate_json.py) 脚本：
```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_json.py \
--json_path sample_openrouter_outputs.json \
--gtruth_answer gtruth_answer \
--generated_text message_content
```

```
Accuracy: 100.0% (5/5)
```

&nbsp;
## 生成 MATH-500 蒸馏数据集

要为包含 500 个样例的 MATH-500 集生成教师答案，可以省略 `--math_json`；两个脚本都会自动加载 `math500_test.json`（首次使用时还会保存一个本地副本）。

**Ollama**
```bash
uv run generate_with_ollama.py \
  --dataset_size 500 \
  --model deepseek-r1:8b \
  --max_new_tokens 8192 \
  --out_file math500_ollama_distill.json
```

**OpenRouter**
```bash
OPENROUTER_API_KEY="YOUR_API_KEY" uv run generate_with_openrouter.py \
  --dataset_size 500 \
  --model deepseek/deepseek-r1 \
  --num_processes 1 \
  --out_file math500_openrouter_distill.json
```

&nbsp;
## 生成包含 12,000 道 MATH 题目的蒸馏数据集

这里使用第 6、7、8 章中的同一组互不重叠的 12,000 个训练样例。如果还没有该数据集，请先下载：
```bash
curl -fL -o math_full_minus_math500.json \
https://raw.githubusercontent.com/rasbt/math_full_minus_math500/refs/heads/main/math_full_minus_math500.json
```

**Ollama**
```bash
uv run generate_with_ollama.py \
  --math_json math_full_minus_math500.json \
  --dataset_size 12000 \
  --model deepseek-r1:8b \
  --max_new_tokens 8192 \
  --resume \
  --out_file math12000_ollama_distill.json
```

**OpenRouter**
```bash
OPENROUTER_API_KEY="YOUR_API_KEY" uv run generate_with_openrouter.py \
  --math_json math_full_minus_math500.json \
  --dataset_size 12000 \
  --model deepseek/deepseek-r1 \
  --num_processes 50 \
  --resume \
  --out_file math12000_openrouter_distill.json
```

对于大规模 OpenRouter 运行，请根据账户限制和期望吞吐量，减少或增加 `--num_processes`。
