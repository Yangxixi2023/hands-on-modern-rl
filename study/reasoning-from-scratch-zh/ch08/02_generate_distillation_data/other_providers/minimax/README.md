# MiniMax 蒸馏服务提供商

本文件夹包含用于第 8 章蒸馏数据生成的 MiniMax 专用托管生成脚本。

请在 `ch08/02_generate_distillation_data/` 目录下运行下面的命令，这样 `math_train_sample.json` 等相对路径才能按原样工作。

输入和输出 JSON 格式与[主 README](../../README.md#input-data-format)中的说明相同。

&nbsp;
## 文件

- [generate_with_minimax.py](generate_with_minimax.py)：使用 MiniMax 云 API 生成模型答案用于蒸馏。MiniMax 通过兼容 OpenAI 的 API 提供 MiniMax-M3（512K 上下文窗口）等模型。

&nbsp;
## MiniMax 设置

1. 在 [MiniMax Platform](https://platform.minimaxi.com/) 创建账户
2. 在账户设置中生成 API 密钥
3. 将 API 密钥安全保存（例如保存在密码管理器中）

可用模型：
- `MiniMax-M3` — 最新模型，512K 上下文窗口，最大输出 128K（默认）
- `MiniMax-M2.7` — 上一代模型，1M 上下文窗口
- `MiniMax-M2.7-highspeed` — M2.7 的高速版本，针对吞吐量优化

&nbsp;
## 使用 MiniMax 生成数据
```bash
MINIMAX_API_KEY="YOUR_API_KEY" uv run other_providers/minimax/generate_with_minimax.py \
  --math_json math_train_sample.json \
  --dataset_size 5 \
  --model MiniMax-M3 \
  --num_processes 1 \
  --out_file sample_minimax_outputs.json
```


如果你不是 `uv` 用户，请将 `uv run` 替换为 `python`。

输出文件的结构与 Ollama 和 OpenRouter 脚本生成的文件相同。

**注意：** MiniMax 要求 temperature 参数处于 (0.0, 1.0] 范围内。脚本会自动将超出范围的值限制在该范围。

&nbsp;
## 生成 MATH-500 蒸馏数据集

要为包含 500 个样例的 MATH-500 集生成教师答案，可以省略 `--math_json`；MiniMax 脚本会自动加载 `math500_test.json`（首次使用时还会保存一个本地副本）。
```bash
MINIMAX_API_KEY="YOUR_API_KEY" uv run other_providers/minimax/generate_with_minimax.py \
  --dataset_size 500 \
  --model MiniMax-M3 \
  --num_processes 1 \
  --out_file math500_minimax_distill.json
```


&nbsp;
## 生成包含 12,000 道 MATH 题目的蒸馏数据集

这里使用第 6、7、8 章中的同一组互不重叠的 12,000 个训练样例。如果还没有该数据集，请先下载：
```bash
curl -fL -o math_full_minus_math500.json \
https://raw.githubusercontent.com/rasbt/math_full_minus_math500/refs/heads/main/math_full_minus_math500.json
```


```bash
MINIMAX_API_KEY="YOUR_API_KEY" uv run other_providers/minimax/generate_with_minimax.py \
  --math_json math_full_minus_math500.json \
  --dataset_size 12000 \
  --model MiniMax-M3 \
  --num_processes 50 \
  --resume \
  --out_file math12000_minimax_distill.json
```

对于大规模运行，请根据账户限制和期望吞吐量，减少或增加 `--num_processes`。
