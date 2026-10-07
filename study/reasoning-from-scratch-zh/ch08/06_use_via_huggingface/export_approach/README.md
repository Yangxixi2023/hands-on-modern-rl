# 第 8 章奖励材料：通过 Hugging Face Transformers 使用从零实现的 Qwen3 代码

本文件夹展示如何将从零实现的 Qwen3Model（../../../reasoning_from_scratch/qwen3.py）以及第 6–8 章生成的兼容 .pth 检查点，转换为 Hugging Face Transformers 兼容的目录，并使用 Hugging Face 推理函数和 Trainer 运行它。

导出功能通过自定义的 transformers 架构实现，因此支持 Hugging Face 的标准 API，例如 AutoConfig、AutoTokenizer、AutoModelForCausalLM、model.generate(...) 和 Trainer。但由于它是自定义代码，加载时需要使用 trust_remote_code=True。

&nbsp;
## 文件

- [hf_export.py](hf_export.py)：将从零实现的 Qwen3 权重或保存的 .pth 检查点转换为 Hugging Face 模型目录
- [hf_inference.py](hf_inference.py)：使用 AutoModelForCausalLM 运行文本生成
- [hf_trainer.py](hf_trainer.py)：在第 8 章蒸馏 JSON 格式的数据上，使用 transformers.Trainer 继续训练导出的模型
- [hf_qwen3.py](hf_qwen3.py)：为导出的 Qwen3 架构实现自定义 Hugging Face PretrainedConfig 和 PreTrainedModel

导出脚本将 Hugging Face 专用的模型代码保留在本文件夹中，并从 reasoning_from_scratch（../../../reasoning_from_scratch）包导入第 3 章提示词模板、RoPE 辅助函数和 Qwen3 下载函数等共享工具。（安装详情请见[第 2 章设置说明](../../../ch02/02_setup-tips/python-instructions.md)。）

---

**注意**：如果你不是 uv 用户，请将示例中的 uv run ...py 替换为 python ...py。

---

&nbsp;
## 第 1 步：安装依赖

本指南除了仓库依赖外，还使用 Hugging Face Transformers。若要使用 transformers.Trainer，还需要 accelerate。
```bash
pip install transformers accelerate
```


如果使用 uv，则运行：
```bash
uv add --dev transformers accelerate
```


&nbsp;
## 第 2 步：导出原始 Qwen3 模型

要将原始基础模型导出为 Hugging Face 目录，请运行：
```bash
uv run hf_export.py \
  --output_dir hf-qwen3-base \
  --tokenizer_kind "base"  # or use "reasoning"
```


如果本地已经有原始 .pth 模型和分词器，可以避免下载：
```bash
uv run hf_export.py \
  --output_dir hf-qwen3-base \
  --tokenizer_kind base \
  --model_path ../../../ch02/01_main-chapter-code/qwen3/qwen3-0.6B-base.pth \
  --tokenizer_path ../../../ch02/01_main-chapter-code/qwen3/tokenizer-base.json
```


第 6–8 章的 .pth 检查点同样适用。

导出的目录将包含：

- config.json
- generation_config.json
- 分词器文件
- 模型权重（默认文件名为 model.safetensors）
- trust_remote_code=True 所需的自定义 Python 模块副本

&nbsp;
### 导出代码的工作方式

导出器不会将模型转换为 Hugging Face 官方的 Qwen 实现，也不会修改学习到的权重。它会执行以下操作：

1. 构建自定义 Hugging Face PretrainedConfig 和 PreTrainedModel，复现从零实现的 Qwen3Model 架构
2. 将原始 .pth 的 state_dict 直接加载到自定义 Hugging Face 模型中，不重命名或重塑可训练参数
3. 使用标准 Hugging Face 目录格式保存结果，使 AutoConfig、AutoTokenizer、AutoModelForCausalLM、generate(...) 和 Trainer 都能加载它

主要新增或封装的内容包括：

- Hugging Face 配置文件（config.json）
- forward(...) 签名与 transformers 兼容的 Hugging Face 模型类
- Hugging Face 分词器文件
- Hugging Face 生成元数据（generation_config.json）
- 由 trust_remote_code=True 加载的自定义 Python 源文件

导出过程中还有一个细节：从零实现的检查点只保存可训练权重，而 Hugging Face 导出还会打包预计算的 RoPE cos 和 sin 缓冲区，从而使重新加载的导出模型在数值上保持一致。

对于 --tokenizer_kind reasoning，导出器还会将推理聊天模板附加到分词器，因此推理脚本可以自动按预期的聊天格式包装提示词。

由于这个自定义 Hugging Face 模块会导入已安装的 reasoning_from_scratch（../../../reasoning_from_scratch）包，只有在 Python 环境中安装了该包时，导出的目录才可正常使用。

&nbsp;
## 第 3 步：导出保存的检查点

同一个导出器也适用于第 8 章蒸馏检查点，或本仓库生成的其他兼容 .pth 文件。

例如，如果使用推理分词器训练了一个第 8 章检查点：
```bash
uv run hf_export.py \
  --output_dir hf-qwen3-distill \
  --model_path ../../04_train_with_distillation/checkpoints/distill/qwen3-0.6B-distill-step00004-epoch1.pth \
  --tokenizer_kind reasoning
```


重要说明：

- 第 8 章蒸馏检查点，以及使用推理分词器训练的其他检查点，请使用 --tokenizer_kind reasoning。
- 使用基础分词器训练的检查点，请使用 --tokenizer_kind base。
- 如果匹配的分词器 JSON 已在磁盘上，可以通过 --tokenizer_path 传入，以避免下载。

&nbsp;
## 第 4 步：运行 Hugging Face 推理

导出后，使用 AutoTokenizer 和 AutoModelForCausalLM 运行推理：
```bash
uv run hf_inference.py \
  --model_dir hf-qwen3-base \
  --prompt "If x + 7 = 19, what is x?"
```


脚本内部会：

1. 使用 trust_remote_code=True 加载导出的模型
2. 使用第 3 章相同的数学提示词模板格式化提示词
3. 当导出模型使用推理分词器时，自动应用推理聊天包装
4. 调用 model.generate(...)

如果希望直接使用原始 Hugging Face API，对应模式如下：
```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("hf-qwen3-base")
config = AutoConfig.from_pretrained("hf-qwen3-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "hf-qwen3-base",
    trust_remote_code=True,
)
```


&nbsp;
## 第 5 步：使用 Trainer 继续训练

可以在与 ../../04_train_with_distillation（../../04_train_with_distillation）相同的 JSON 格式上，使用 Hugging Face Trainer 继续训练导出的检查点。

示例：
```bash
uv run hf_trainer.py \
  --model_dir hf-qwen3-base \
  --data_path ../../02_generate_distillation_data/sample_openrouter_outputs.json \
  --dataset_size 5 \
  --validation_size 1 \
  --epochs 1 \
  --logging_steps 1 \
  --save_steps 10
```


脚本保留了从零实现蒸馏代码中相同的“只训练答案”目标：

- 将提示词 token 从损失中屏蔽
- 只有蒸馏得到的答案 token 参与交叉熵损失
- 对推理导出模型，脚本会在最终答案前将教师轨迹包在 <think>...</think> 中



&nbsp;
## 在其他位置加载导出结果

导出后，可以将目录复制到另一台机器，或上传到 Hugging Face Hub 并在那里加载，只要目标环境安装了 reasoning_from_scratch：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "path-or-hub-repo",
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "path-or-hub-repo",
    trust_remote_code=True,
)
```
