# 故障排查指南

本页汇总阅读本书时遇到的常见问题和设置建议。

&nbsp;
## JupyterLab 滚动错误

如果你是在 JupyterLab 而不是 VSCode 中查看 Notebook 代码，请注意 JupyterLab 最近版本的默认设置存在滚动问题。建议前往 Settings -> Settings Editor，将“Windowing mode”改为“none”（如下图所示），这样似乎可以解决问题。


![Jupyter Glitch 1](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_1.webp)

<br>

![Jupyter Glitch 2](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_2.webp)


&nbsp;
## 第 2 章

&nbsp;
### 文件下载问题

如果文件下载遇到问题，请使用[此讨论页](https://github.com/rasbt/reasoning-from-scratch/discussions/145)。

代码从以下 Hugging Face 地址下载文件。你也可以在浏览器中手动打开这些地址，检查机器或网络是否阻止了访问：

- 第 2 章模型和分词器文件： [rasbt/qwen3-from-scratch](https://huggingface.co/rasbt/qwen3-from-scratch/tree/main)
- 基础模型文件： [qwen3-0.6B-base.pth](https://huggingface.co/rasbt/qwen3-from-scratch/resolve/main/qwen3-0.6B-base.pth)
- 基础分词器文件： [tokenizer-base.json](https://huggingface.co/rasbt/qwen3-from-scratch/resolve/main/tokenizer-base.json)
- 推理模型文件： [qwen3-0.6B-reasoning.pth](https://huggingface.co/rasbt/qwen3-from-scratch/resolve/main/qwen3-0.6B-reasoning.pth)
- 推理分词器文件： [tokenizer-reasoning.json](https://huggingface.co/rasbt/qwen3-from-scratch/resolve/main/tokenizer-reasoning.json)
- 第 7 章 GRPO 检查点： [rasbt/qwen3-from-scratch-grpo-checkpoints](https://huggingface.co/rasbt/qwen3-from-scratch-grpo-checkpoints/tree/main)
- 第 8 章蒸馏检查点： [rasbt/qwen3-from-scratch-distill-checkpoints](https://huggingface.co/rasbt/qwen3-from-scratch-distill-checkpoints/tree/main)

&nbsp;
#### SSL / 代理 / 证书错误

如果模型下载失败并出现包含 `SSL`、`CERTIFICATE_VERIFY_FAILED` 或 `ProxyError` 的错误，问题通常来自环境，而不是文件缺失。

这种情况总体并不常见，但在 VPN、代理、防火墙或杀毒软件拦截 HTTPS 流量的工作或学校电脑上可能发生。此时可尝试：

- 检查上面列出的 Hugging Face 地址是否能在浏览器中打开。
- 如果分词器可以下载，但 `.pth` 模型文件不能，代理可能阻止了较大的文件或 `.pth` 扩展名。
- 请 IT 团队允许下载，或让 Python 信任代理证书。
- 在一些受管理的机器上，读者报告使用 `pip install pip-system-certs` 成功；该命令让 Python 使用操作系统证书存储。

&nbsp;
### `InductorError: CppCompileError`
如果你是 Linux 用户，在执行 `InductorError: CppCompileError: C++ compile error` 时看到 `torch.compile`，并且错误包含以下行：

```python
Python.h: No such file or directory
81 | #include <Python.h>
| ^~~~~~~~~~
compilation terminated.
```

这表示 Python 运行时可能缺少为 CPU 编译模型所需的一些 C++ 头文件。

例如，可以检查文件是否存在：`ls -l /usr/include/python3.12/Python.h`。

如果文件不存在，可以尝试通过以下命令安装另一套 Python 运行时：

```bash
sudo apt-get install -y python3.12-dev build-essential
```

或者在调用 `torch.compile` 前禁用 PyTorch 的 C++ 要求：

```python
import torch
import torch._inductor.config as inductor_config

inductor_config.cpp_wrapper = False

compiled_model = torch.compile(model)
```

更多背景请参阅 [#192](https://github.com/rasbt/reasoning-from-scratch/issues/192)。


&nbsp;
### Windows CPU：`fatal error C1083` 或 `algorithm` 导致 `omp.h`

如果你在 Windows 上使用 `torch.compile()` 时出现以下错误：

```text
fatal error C1083: Cannot open include file: 'algorithm': No such file or directory
```

或者

```text
fatal error C1083: Cannot open include file: 'omp.h': No such file or directory
```

问题通常出在 TorchInductor 使用的本地 Windows 编译器或 OpenMP 配置（而不是本书或仓库中的代码）。

一位读者在仅使用 CPU 的 Intel 系统上，于[论坛](https://livebook.manning.com/forum?product=raschka2&comment=583365)报告了以下经验：

- 升级 PyTorch 解决了缺失的 `algorithm` 头文件。
- 但 `omp.h` 头文件仍然缺失。
- 使用 `"eager"` 或 `"aot_eager"` 等备用后端可以运行代码。

例如：

```python
compiled_model = torch.compile(model, backend="eager")

# or

compiled_model = torch.compile(model, backend="aot_eager")
```

请注意，这是一种变通方法，不是完整修复。它可能有帮助，但不会使用完整的 TorchInductor 编译路径，因此加速效果可能小于正常工作的 `torch.compile()`。

**还请记住，torch.compile 对本书并非必需，你完全可以跳过这一节。**

无论如何，如果你想让它运行，在花大量时间调试之前，先运行一个最小的健全性检查会很有帮助：

```python
import torch

device = "cpu"  # or "xpu"

def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b


opt_foo = torch.compile(foo)
out = opt_foo(torch.randn(10, 10).to(device), torch.randn(10, 10).to(device))
print(out.shape)
```

如果这个小示例已经失败，问题很可能在 PyTorch 或编译器配置，而不是本书或仓库中的模型代码。

更多设置建议还可参阅[在 Windows 上使用 `torch.compile()`](ch02/04_torch-compile-windows/README.md)和 PyTorch 的 [Windows CPU/XPU 指南](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html)。如前所述，如果 `torch.compile()` 在你的系统上仍不稳定，跳过它并运行本书示例也完全可以。



&nbsp;
## 第 6 章

&nbsp;
### 检查点损坏

在 `train_rlvr_grpo`（第 6 章）中，`Ctrl+C` 会触发 `KeyboardInterrupt` 处理器，保存一个 `-interrupt` 检查点。如果在保存完成前再次按 `Ctrl+C`，可能中断 `torch.save` 的写入，留下被截断的 `.pth` 文件。退出前请等待 `-interrupt` 检查点提示。

损坏的模型检查点通常会在加载时产生错误或在评估时失败；另一个明显迹象是文件远小于预期的约 1.5 GB。

&nbsp;
## 其他问题

如有其他问题，欢迎新建 GitHub [Issue](https://github.com/rasbt/reasoning-from-scratch/issues)。
