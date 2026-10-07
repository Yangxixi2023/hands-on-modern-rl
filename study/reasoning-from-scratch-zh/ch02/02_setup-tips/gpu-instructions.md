# GPU 云端资源

本节介绍运行本书代码的云端替代方案。

虽然代码可以在没有独立 GPU 的普通笔记本电脑和台式机上运行，但配备 NVIDIA GPU 的云平台可以显著缩短代码运行时间，尤其是第 5 至第 7 章。

&nbsp;

## 使用 Lightning Studio

为了获得顺畅的云端开发体验，我推荐使用 [Lightning AI Studio](https://lightning.ai/) 平台。它可以让用户配置持久化环境，并在云端 CPU 和 GPU 上同时使用 VSCode 与 Jupyter Lab。

启动新的 Studio 后，可以打开终端，执行以下配置步骤来克隆仓库并安装依赖：

```bash
git clone https://github.com/rasbt/reasoning-from-scratch.git
cd reasoning-from-scratch
pip install -r requirements.txt
```

（与 Google Colab 不同，这些步骤只需执行一次，因为 Lightning AI Studio 环境是持久化的，即使在 CPU 和 GPU 机器之间切换也不需要重复执行。）

然后，进入想要运行的 Python 脚本或 Jupyter Notebook。你也可以选择轻松地连接 GPU 来加速代码运行，例如在第 5 章预训练大语言模型，或在第 6、7 章微调模型时。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/studio.webp" alt="1" width="700">

&nbsp;

## 使用 Google Colab

要在云端使用 Google Colab 环境，请访问 [https://colab.research.google.com/](https://colab.research.google.com/)，然后从 GitHub 菜单打开相应章节的 Notebook，或者像下图所示将 Notebook 拖入 *Upload* 字段。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_1.webp" alt="1" width="700">


还要确保将相关文件（数据集文件，以及 Notebook 导入的 .py 文件）一并上传到 Colab 环境，如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_2.webp" alt="2" width="700">


你可以按照下图所示更改 *Runtime*，选择在 GPU 上运行代码。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_3.webp" alt="3" width="700">


&nbsp;
## 有问题？

如有任何问题，欢迎通过该 GitHub 仓库的 [Discussions](https://github.com/rasbt/reasoning-from-scratch/discussions) 论坛联系。
