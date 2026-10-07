# 由大语言模型担任评审

本补充材料实现了由大语言模型担任评审的方法：通过开源 Ollama 库使用 `gpt-oss:20b`，在 MATH-500 上评估 Qwen3 0.6B 的基础模型和推理模型变体。

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/appendix-f/Appendix_F_F06_raschka.webp" width="500px">

- Ollama 是用于高效运行大语言模型的开源应用
- 它是 llama.cpp 的封装（[https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)）；llama.cpp 使用纯 C/C++ 实现大语言模型，以最大限度提高效率
- 请注意，Ollama 是用于生成文本（推理）的工具，而不是用于训练或微调大语言模型的工具
- 运行下面的代码前，请访问 [https://ollama.com](https://ollama.com) 并按照说明安装 Ollama（例如点击“Download”按钮，下载适合你操作系统的 Ollama 应用）
- macOS 和 Windows 用户请点击下载的 Ollama 应用；如果应用提示安装命令行功能，请选择“是”
- Linux 用户可以使用 Ollama 网站提供的安装命令
- 在计算机上运行 Ollama 有三种方式：

**1. `ollama serve`**

- 这会将 Ollama 后端作为服务器运行，通常地址为 `http://localhost:11434`。在通过 API 调用模型之前，它不会加载模型。如果想通过 Python 使用 Ollama，就应采用这种方式。

**2. `ollama run gpt-oss:20b`**

- 这是一个便捷封装。如果服务器尚未运行，它会启动服务器，然后在首次运行时下载模型，并进入可以与模型聊天的交互式终端。底层使用的是同一个服务器 API。

**3. Ollama 桌面应用**

- 该应用会自动运行同一个后端，并在其上提供图形界面（如上图所示）。它还会应用默认的系统提示词、温度和停止序列，这可以解释为什么它的回答与直接调用原始 API 时不同。

## 用法

下面列出可用选项及其默认值。

<br>

---

**注意**：如果你不使用 `uv`，请在下面的示例中将 `uv run ...py` 替换为 `python ...py`。

---

```bash
uv run ollama-judge.py --help
usage: ollama-judge.py [-h] [--device DEVICE]
                       [--which_model {base,reasoning}]
                       [--dataset_size DATASET_SIZE]
                       [--max_new_tokens MAX_NEW_TOKENS]
                       [--url URL]
                       [--judge_model JUDGE_MODEL]

options:
  -h, --help            show this help message and
                        exit
  --device DEVICE       Device e.g., "cpu",
                        "cuda", "cuda:0", "mps".
  --which_model {base,reasoning}
                        Candidate variant to use.
                        Defaults to "base".
  --dataset_size DATASET_SIZE
                        Number of MATH-500
                        examples to evaluate.
                        Default: 10
  --max_new_tokens MAX_NEW_TOKENS
                        Max new tokens for
                        candidate generation.
                        Default: 2048
  --url URL             Ollama chat endpoint for
                        the judge. Default: "http:
                        //localhost:11434/api/chat
                        "
  --judge_model JUDGE_MODEL
                        Judge model name (Ollama).
                        Used only for scoring.
                        Default: "gpt-oss:20b"
```

**基础模型**

```bash
➜  uv run ollama-judge.py
Using Apple Silicon GPU (MPS)
Model: base
Device: mps
✓ qwen3/qwen3-0.6B-base.pth already up-to-date
✓ qwen3/tokenizer-base.json already up-to-date
Ollama running: True
[1/10] score=5
[2/10] score=1
[3/10] score=5
[4/10] score=5
[5/10] score=3
[6/10] score=5
[7/10] score=5
[8/10] score=3
[9/10] score=5
[10/10] score=1

Summary
-------
Average score: 3.800 over 10 example(s)
Counts: 1:2 2:0 3:2 4:0 5:6
```

**推理模型**

```bash
➜  uv run ollama-judge.py --which_model reasoning
Using Apple Silicon GPU (MPS)
Model: reasoning
Device: mps
✓ qwen3/qwen3-0.6B-reasoning.pth already up-to-date
✓ qwen3/tokenizer-reasoning.json already up-to-date
Ollama running: True
[1/10] score=5
[2/10] score=5
[3/10] score=5
[4/10] score=5
[5/10] score=4
[6/10] score=5
[7/10] score=5
[8/10] score=1
[9/10] score=5
[10/10] score=3

Summary
-------
Average score: 4.300 over 10 example(s)
Counts: 1:1 2:0 3:1 4:1 5:7
```
