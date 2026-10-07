# 《从零构建推理模型》中文学习副本

这是 Sebastian Raschka《Build a Reasoning Model (From Scratch)》的中文代码学习副本。代码、模型实现、数据、脚本和测试来自官方仓库最新版本；第 2–8 章、附录 C–F 的 Notebook 教学文字，以及仓库中的辅助 README、安装、测试和排障文档，均以官方最新英文源为原文重新翻译；代码单元格和代码块保持原样。

- [官方英文仓库](https://github.com/rasbt/reasoning-from-scratch)
- [中文翻译仓库](https://github.com/xbsheng/reasoning-from-scratch-zh)
- [Manning 书页](https://www.manning.com/books/build-a-reasoning-model-from-scratch)
- [本地学习指南](学习指南.md)

## 内容目录

| 部分 | 内容 | 主 Notebook |
|---|---|---|
| 第 1 章 | 理解推理模型 | 无代码 |
| 第 2 章 | 使用预训练 LLM 生成文本 | `ch02/01_main-chapter-code/ch02_main.ipynb` |
| 第 3 章 | 评估推理模型 | `ch03/01_main-chapter-code/ch03_main.ipynb` |
| 第 4 章 | 使用推理时扩展改进推理 | `ch04/01_main-chapter-code/ch04_main.ipynb` |
| 第 5 章 | 通过自我修订进行推理时扩展 | `ch05/01_main-chapter-code/ch05_main.ipynb` |
| 第 6 章 | 使用强化学习训练推理模型 | `ch06/01_main-chapter-code/ch06_main.ipynb` |
| 第 7 章 | 改进 GRPO 强化学习 | `ch07/01_main-chapter-code/ch07_main.ipynb` |
| 第 8 章 | 蒸馏推理模型以实现高效推理 | `ch08/01_main-chapter-code/ch08_main.ipynb` |
| 附录 C | Qwen3 LLM 源码 | `chC/01_main-chapter-code/chC_main.ipynb` |
| 附录 D | 使用更大的 LLM | `chD/chD_main.ipynb` |
| 附录 E | 批处理与吞吐量导向的执行 | `chE/chE_main.ipynb` |
| 附录 F | LLM 评估的常见方法 | `chF/01_main-chapter-code/chF_main.ipynb` |
| 附录 G | 构建聊天界面 | `chG/01_main-chapter-code/` |

每章的 `*_exercise-solutions.ipynb` 是习题解答。先完成自己的实现，再打开解答 Notebook 对照。

## 安装与运行

项目要求 Python 3.10–3.14，当前官方配置使用 PyTorch 2.10 或更高版本。Windows PowerShell 中可以执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
jupyter lab
```

然后打开 `ch02/01_main-chapter-code/ch02_main.ipynb`。第 2–4 章可以先用 CPU；第 5–7 章的完整实验建议使用 GPU。模型权重和部分数据会从 Hugging Face 下载。

## 许可证

本项目代码沿用官方仓库的 Apache-2.0 许可证。书籍正文的版权和授权以 Manning 书页为准。
