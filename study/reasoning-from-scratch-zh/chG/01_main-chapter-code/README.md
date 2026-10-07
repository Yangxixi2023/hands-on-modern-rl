# 附录 G：构建聊天界面



本目录包含运行类似 ChatGPT 用户界面的代码，用于与本书使用或开发的大语言模型交互，如下图所示。



![Chainlit UI example](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen3-chainlit.gif)



我们使用开源的 [Chainlit Python 包](https://github.com/Chainlit/chainlit)实现此用户界面。

&nbsp;
## 步骤 1：安装依赖

本附录请使用 Python 3.13。Chainlit 目前不支持 Python 3.14（见[上游 issue](https://github.com/Chainlit/chainlit/issues/2952)）。在 Python 3.14 中安装仓库的 `extra` 依赖时会跳过 Chainlit。

首先，从仓库根目录进入此目录：

```bash
cd chG/01_main-chapter-code
```

如果你使用 `pip`，请在此目录创建 Python 3.13 虚拟环境并安装依赖。在 macOS 和 Linux 上：

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e "../..[extra]"
```

在 Windows（PowerShell）上：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e "../..[extra]"
```

如果你使用 `uv`，请跳过 `pip` 命令。`uv run` 命令会创建隔离的 Python 3.13 环境并安装仓库的 `extra` 依赖。即使主项目环境使用 Python 3.14，这种方式也能工作。

&nbsp;

## 步骤 2：运行 `app` 代码

此目录包含 2 个文件：

1. [`qwen3_chat_interface.py`](qwen3_chat_interface.py)：此文件以思考模式加载并使用 Qwen3 0.6B 模型。
2. [`qwen3_chat_interface_multiturn.py`](qwen3_chat_interface_multiturn.py)：与上面相同，但配置为记住消息历史。

（打开并检查这些文件以了解更多信息。）

在终端运行以下任一命令启动 UI 服务器：

```bash
chainlit run qwen3_chat_interface.py
```

如果使用 `uv`，则运行：

```bash
uv run --isolated --python 3.13 --extra extra chainlit run qwen3_chat_interface.py
```

运行上述任一命令后应打开新的浏览器标签页，你可以在其中与模型交互。如果浏览器标签页没有自动打开，请查看终端命令，并将本地地址复制到浏览器地址栏（通常地址为 `http://localhost:8000`）。

## 使用自定义检查点

由于 `chainlit run ...` 占用命令行参数，这些脚本从 `CHECKPOINT_PATH` 环境变量读取自定义检查点路径，而不是使用 `argparse`。

终端示例：

```bash
CHECKPOINT_PATH=/absolute/path/to/qwen3-0.6B-distill-step06682-epoch1.pth \
uv run --isolated --python 3.13 --extra extra \
  chainlit run qwen3_chat_interface.py
```

注意：

- 确保 `WHICH_MODEL` 与检查点所需的分词器保持一致。
- 第 8 章的检查点来自 [`ch08/05_download_training_checkpoints`](../../ch08/05_download_training_checkpoints) 使用推理分词器，因此请使用 `WHICH_MODEL = "reasoning"`。
- 设置 `CHECKPOINT_PATH` 后，脚本只会将分词器下载到 `LOCAL_DIR`；不会重新下载默认模型权重。
