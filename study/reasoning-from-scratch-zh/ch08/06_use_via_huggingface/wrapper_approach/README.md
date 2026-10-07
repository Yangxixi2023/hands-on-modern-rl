# 第 8 章奖励材料：通过本地 Hugging Face 封装使用 Qwen3

本文件夹展示如何将从零实现的 Qwen3Model（../../../reasoning_from_scratch/qwen3.py）封装在轻量的本地 PreTrainedModel 类中，以兼容 Hugging Face transformers 库。

这样就可以直接使用本仓库中的本地 .pth 模型文件，包括 Qwen3 基础权重和第 6–8 章生成的兼容检查点：

- model.generate(...)
- transformers.Trainer

&nbsp;
## 文件

- [hf_wrapper.py](hf_wrapper.py)：围绕本书使用的从零实现 Qwen3Model 的本地 PreTrainedModel 封装
- [hf_inference.py](hf_inference.py)：使用该封装和仓库分词器生成文本
- [hf_trainer.py](hf_trainer.py)：使用该封装和第 8 章蒸馏 JSON 格式的 Trainer 示例

---

**注意**：如果你不是 uv 用户，请将示例中的 uv run ...py 替换为 python ...py。

---

&nbsp;
## 这个封装做什么

该封装将模型保留在本仓库中，并使其适配 Hugging Face API。

具体来说，它会：

- 将本地 .pth 模型文件直接加载到 Qwen3Model
- 用 PreTrainedModel 接口封装该模型
- 暴露与 Trainer 兼容的 forward(...) 方法
- 启用 model.generate(...)
- 继续使用仓库中的 Qwen3Tokenizer

为什么需要它？有些读者希望在 transformers 中进一步探索模型，而 transformers 比本仓库的从零代码提供了更多功能。

&nbsp;
## 限制

这是围绕从零实现模型的小型本地封装。

重要影响：

- 适用于安装了 reasoning_from_scratch 的环境
- 不提供 AutoTokenizer.from_pretrained(...) 工作流
- 不会创建包含 config.json 和分词器文件的可复用模型目录
- 生成逻辑保持有意的简洁：它会重新计算完整前缀，而不是将从零实现的 KV 缓存适配到 Hugging Face 缓存类；如果需要完整支持，应改用 [../export_approach](../export_approach)

这些限制使代码保持简短，并专注于在本仓库内本地使用。

&nbsp;
## 第 1 步：安装依赖

本指南除了仓库依赖外，还使用 Hugging Face Transformers。
```bash
pip install transformers accelerate
```


如果使用 uv，则运行：
```bash
uv add --dev transformers accelerate
```


&nbsp;
## 第 2 步：运行本地封装推理

要通过封装运行基础模型，请使用：
```bash
  uv run hf_inference.py \
    --tokenizer_kind base \
    --prompt "If x + 7 = 19, what is x?"
```


要运行推理变体，请使用：
```bash
  uv run hf_inference.py \
    --tokenizer_kind reasoning \
    --prompt "If x + 7 = 19, what is x?"
```


要改为运行本地检查点，请使用：
```bash
uv run hf_inference.py \
  --tokenizer_kind reasoning \
  --model_path ../../04_train_with_distillation/checkpoints/distill/qwen3-0.6B-distill-step00004-epoch1.pth \
  --prompt "If x + 7 = 19, what is x?"
```


如果省略 --model_path，脚本会根据选定的 --tokenizer_kind 下载默认的基础模型或推理模型。如果提供 --model_path，它可以指向基础 Qwen3 .pth 文件，或第 6–8 章生成的任何兼容检查点。

推理脚本内部会：

1. 构建本地封装模型
2. 将选定的 .pth 模型文件加载到封装后的 Qwen3Model 中
3. 使用仓库分词器对提示词分词
4. 调用 model.generate(...)

&nbsp;
## 第 3 步：使用 Trainer 继续训练

同一个封装也可以与 transformers.Trainer 一起使用：
```bash
uv run hf_trainer.py \
  --tokenizer_kind reasoning \
  --model_path ../../04_train_with_distillation/checkpoints/distill/qwen3-0.6B-distill-step00004-epoch1.pth \
  --data_path ../../02_generate_distillation_data/sample_openrouter_outputs.json \
  --dataset_size 5 \
  --validation_size 1 \
  --epochs 1 \
  --logging_steps 1
```


与推理一样，--model_path 可以指向基础 Qwen3 权重或第 6–8 章生成的兼容检查点。

训练器保留第 8 章其他位置使用的“只训练答案”目标：

- 屏蔽提示词 token
- 只有答案 token 参与损失
- 推理模式会将教师轨迹包在 <think>...</think> 中

输入 JSON 格式与 [../../02_generate_distillation_data](../../02_generate_distillation_data) 生成的蒸馏数据一致。

&nbsp;
