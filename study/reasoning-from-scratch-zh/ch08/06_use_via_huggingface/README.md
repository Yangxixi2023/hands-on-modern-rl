# 第 8 章奖励材料：通过 Hugging Face 使用 Qwen3

本文件夹提供两种方式：使用本仓库中的从零实现 [`Qwen3Model`](../../reasoning_from_scratch/qwen3.py) 和兼容的 `.pth` 检查点，并接入 Hugging Face `transformers`。

两种方式都支持 Hugging Face 风格的推理和训练。区别在于：你是需要一个可复用的 Hugging Face 模型目录，还是需要在现有 PyTorch 模型外包一层更轻量的本地封装。

&nbsp;
## 方案


&nbsp;
### 1) `wrapper_approach`

[./wrapper_approach](./wrapper_approach) 将模型保留为本地 `.pth` 文件，并在 `Qwen3Model` 外封装一个轻量的本地 `PreTrainedModel`，使其能够使用部分 Hugging Face API。

如果你希望：

- 尽量少写额外代码
- 在本仓库内进行本地实验
- 不经过导出步骤，直接使用 `model.generate(...)` 和 `transformers.Trainer`
- 直接从 `.pth` 加载基础模型或第 6–8 章的检查点


&nbsp;
### 2) `export_approach`

[./export_approach](./export_approach) 将从零实现的 Qwen3 权重或兼容检查点转换为 Hugging Face 兼容的模型目录。

如果你希望：

- 保存包含 `config.json`、分词器文件和权重的模型目录
- 使用 `AutoConfig`、`AutoTokenizer` 和 `AutoModelForCausalLM`
- 使用更接近 Hugging Face 模型通常打包方式的工作流



&nbsp;
## 选择哪种方案？

- 如果目的是学习，或需要与 `transformers` 进行轻量本地集成，请选择 [wrapper_approach](wrapper_approach)。
- 如果目的是创建 Hugging Face 模型包并优化计算性能，请选择 [export_approach](export_approach)。
