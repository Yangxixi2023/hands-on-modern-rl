# Python 配置建议

本书中的代码基本是自包含的，我也尽力减少外部依赖。不过，为了让本书易于阅读、适合学习，并将篇幅控制在 2000 页以内，仍需要使用少量 Python 包。

本节介绍两种适合初学者的安装所需包的方法，以便运行代码示例。

当然，安装和管理 Python 包还有许多其他方式。如果你已经熟悉 Python，并有自己的配置方式或偏好，可以跳过本节。

如果下面两种方法都无法使用，欢迎联系我，例如在 [Discussion](https://github.com/rasbt/reasoning-from-scratch/discussions) 中发帖。

&nbsp;
## 选项 1：使用 `pip`（内置，适用于所有环境）

如果你已经在使用较新的 Python 版本，可以使用内置的 `pip` 安装器来安装包。

本书使用 Python 3.12。不过，只要 PyTorch 支持，更新的 Python 3.13 和 3.14，以及较旧的 3.11 和 3.10 也可以正常工作。运行以下命令检查 Python 版本：

```bash
python --version
```

对于[附录 G 聊天界面](../../chG/01_main-chapter-code)，请使用 Python 3.13，因为 Chainlit 目前不支持 Python 3.14。

如果你使用的是 Python 3.9 或更早版本，可以考虑从 [python.org](https://www.python.org/downloads/) 安装最新版本，或使用 [`pyenv`](https://github.com/pyenv/pyenv) 等工具管理版本。不过，如果要安装新的 Python 版本，请先查看 [PyTorch 官方网站](https://pytorch.org/get-started/locally/)上的建议，确保该版本受到 PyTorch 支持。PyTorch 通常会比最新的 Python 版本晚几个月支持，因此刚发布的 Python 版本不会立即获得支持或推荐使用。

需要时（例如安装 PyTorch 和 Jupyter Lab），运行以下命令安装新包：

```bash
pip install torch jupyterlab
```

你也可以通过 [`requirements.txt`](https://github.com/rasbt/reasoning-from-scratch/blob/main/requirements.txt) 文件一次性安装本书所需的全部 Python 包：

```bash
pip install -r https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/refs/heads/main/requirements.txt
```


&nbsp;
## 选项 2：使用 `uv`（更快，广受推荐）

虽然 `pip` 仍是经典且官方的 Python 包安装方式，但 [`uv`](https://github.com/astral-sh/uv) 是现代且广受推荐的 Python 包管理器，可以自动：

- 创建和管理虚拟环境
- 快速安装包
- 保存锁定文件，保证安装可复现
- 支持类似 `pip` 的命令

&nbsp;
### 安装 `uv` 和 Python 包

可以使用下面的命令安装 `uv`（如需最新建议，也可以参阅官方的[安装说明](https://docs.astral.sh/uv/getting-started/installation/)页面）。

&nbsp;
**macOS / Linux：**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

&nbsp;
**Windows（PowerShell）：**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

安装完成后，可以像前一节使用 `pip` 那样安装新的 Python 包，只需将 `pip` 替换为 `uv pip`。例如：

```bash
uv pip install torch jupyterlab
```

不过，如果你使用 `uv`（我本人也推荐并使用它），最好按照下面的说明使用 `uv` 的原生语法，而不是 `uv pip`。

&nbsp;
### 推荐的 `uv` 工作流

我建议并使用 `uv` 的原生工作流，而不是 `uv pip`。

首先，将 GitHub 仓库克隆到本地：



```bash
git clone https://github.com/rasbt/reasoning-from-scratch.git
```

接着进入该文件夹，例如在 Linux 和 MacOS 上：

```bash
cd reasoning-from-scratch
```

`pyproject.toml` 文件列出了所需的包。第一次运行脚本或打开 Jupyter Lab 时，`uv` 会自动在 `reasoning-from-scratch` 项目中创建一个（默认不可见的）虚拟环境文件夹（`.venv`），并将所有依赖安装到其中。

仓库不包含 `.python-version` 文件。如果希望本地 `uv` 环境使用 Python 3.13，可以运行以下命令创建版本固定：

```bash
uv python pin 3.13
uv sync
```

通常不需要这样做，但一般来说，你可以通过 `uv add` 安装 `pyproject.toml` 中尚未列出的其他包：


```bash
uv add llms_from_scratch
```

上述命令会将该包添加到虚拟环境和 `pyproject.toml` 文件中。

&nbsp;
### 使用 `uv` 运行代码

本节介绍使用 `uv` 运行 Jupyter Lab 和 Python 脚本的命令。

打开 Jupyter Lab：

```bash
uv run jupyter lab
```

运行 Python 脚本：

```bash
uv run python script.py
```

提交到仓库的 `uv.lock` 文件使用 PyTorch 2.10.0，这是本书代码最初使用的版本。因此，普通的 `uv run` 命令默认使用 PyTorch 2.10.0。由于 `uv` 会为每个平台选择适配的包，因此该锁定文件适用于所有受支持的操作系统。

`pyproject.toml` 文件也允许使用更新的 PyTorch 2.x 版本。要在本地锁定文件中更新 PyTorch，请运行：

```bash
uv lock --upgrade-package torch
```

只需执行一次即可。之后的 `uv run` 命令都会使用更新后的版本。注意，这会修改本地的 `uv.lock` 文件。若要恢复仓库中包含的版本，请运行：

```bash
git restore uv.lock
uv sync
```




> **高级用法：** 本节介绍了一种对 `pip` 用户来说较熟悉的 `uv` 简单用法。如果你希望了解更高级的用法，请参阅[这份文档](https://github.com/rasbt/LLMs-from-scratch/tree/main/setup/01_optional-python-setup-preferences)，其中有关于在 `uv` 中管理虚拟环境的更详细说明。
> 如果你是 macOS 或 Linux 用户，并且更喜欢原生 uv 命令，请参阅[本教程](https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/01_optional-python-setup-preferences/native-uv.md)。还建议阅读 [uv 官方文档](https://docs.astral.sh/uv/)以获取更多信息。



&nbsp;
### JupyterLab 使用提示

如果你在 JupyterLab 而不是 VSCode 中查看 Notebook 代码，请注意，JupyterLab（默认设置）在近期版本中出现过滚动问题。建议进入 Settings -> Settings Editor，将“Windowing mode”改为“none”（如下图所示），这似乎可以解决问题。


![Jupyter Glitch 1](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_1.webp)

<br>

![Jupyter Glitch 2](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_2.webp)

&nbsp;
## 有问题？

如有任何问题，欢迎通过该 GitHub 仓库的 [Discussions](https://github.com/rasbt/reasoning-from-scratch/discussions) 论坛联系。
