# 在 Windows 上使用 `torch.compile()`

`torch.compile()` 依赖 *TorchInductor*。TorchInductor 会对内核进行 JIT 编译，因此需要可用的 C/C++ 编译器工具链。

因此，在 Windows 上让 `torch.compile` 正常工作所需的配置可能比 Linux 或 macOS 更复杂；后两者通常只需安装 PyTorch，无需额外步骤。

如果你是 Windows 用户，觉得使用 `torch.compile` 太棘手或复杂，也不用担心：本仓库中的所有代码示例都可以在不编译的情况下正常运行。

下面的提示整理自 [Daniel Kleine](https://github.com/d-kleine) 的建议和以下 [PyTorch 指南](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html)。

&nbsp;
## 1 基础配置（CPU 或 CUDA）

&nbsp;
### 1.1 安装 Visual Studio 2022

- 选择 **“Desktop development with C++”** 工作负载。
- 确保包含 **English language pack**（否则可能遇到 UTF-8 编码错误）。

&nbsp;
### 1.2 打开正确的命令提示符


从以下命令提示符启动 Python：

**“x64 Native Tools Command Prompt for VS 2022”**

或

**“Visual Studio 2022 Developer Command Prompt”**。

也可以手动运行下面的命令初始化环境：

```bash
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

&nbsp;
### 1.3 验证编译器是否正常

运行

   ```bash
   cl.exe
   ```

如果看到打印出的版本信息，说明编译器已经就绪。

&nbsp;
## 2 常见错误排查

&nbsp;
### 2.1 错误：`cl not found`

安装带有“C++ build tools”工作负载的 **Visual Studio Build Tools**，并从开发者命令提示符运行 Python。（详情请参阅 Microsoft 的[指南](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170)。）

&nbsp;
### 2.2 错误：`triton not found`（使用 CUDA 时）

手动安装 Windows 版本的 Triton：

```bash
pip install "triton-windows<3.4"
```

或者，如果你使用 `uv`：

```bash
uv pip install "triton-windows<3.4"
```

（如前文所述，TorchInductor 在编译 CUDA 内核时需要 triton。）



&nbsp;
## 3 其他说明

在 Windows 上，`cl.exe` 编译器只能在 Visual Studio Developer 环境中访问。这意味着，除非从 Developer Command Prompt 启动 Notebook，否则在 Jupyter 等 Notebook 中使用 `torch.compile()` 可能无法工作。

正如本文开头提到的，还有一份 [PyTorch 指南](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html)，一些用户发现它有助于在 Windows CPU 构建上运行 `torch.compile()`。不过请注意，该指南针对 PyTorch 的不稳定分支，因此仅应作为参考。

**如果编译仍然引发问题，可以直接跳过。它是很好的附加功能，但不是学习本书的必要条件。**
