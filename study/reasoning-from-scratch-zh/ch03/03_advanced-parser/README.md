# 第 3 章：高级解析器（补充材料）

本目录包含来自 [issue #133](https://github.com/rasbt/reasoning-from-scratch/issues/133) 的解析器实验。该 issue 提议使用混合 LaTeX 解析器，以处理当前章节解析器可能遗漏的边界情况。



&nbsp;

## 文件

- [compare_with_current_parser.ipynb](compare_with_current_parser.ipynb)：包含使用示例的 Notebook
- [math500_gpt_answers.json](math500_gpt_answers.json)：包含大语言模型答案的 MATH-500 示例，供上面的 Notebook 中某一节使用
- [gen_llm_answers.py](gen_llm_answers.py)：从 Qwen3 模型获取带框答案并保存为 JSON 格式的便捷脚本
- [evaluate_math500_advanced.py](evaluate_math500_advanced.py)：与第 3 章的大语言模型评估脚本 [evaluate_math500.py](../02_math500-verifier-scripts/evaluate_math500.py) 相同，但额外支持 `--hybrid_parser` 参数，用于使用替代的混合解析器，例如：

```python
uv run evaluate_math500_advanced.py --dataset_size 500 --hybrid_parser
```



&nbsp;
## 与第 3 章解析器的差异
evaluate_math500_advanced.py
第 3 章的解析器位于 [reasoning_from_scratch/ch03.py](../../reasoning_from_scratch/ch03.py)，设计目标是保持精简并便于教学：

- 重点是轻量级规范化和符号等价性检查
- 主要将答案视为算术或符号表达式

本目录中的混合解析器（`latex_normalizer_hybrid.py`）采用模式优先的方法，覆盖范围更广：

- 在回退解析前先识别答案格式。
- 增加对区间、并集、方程、矩阵、集合表示、隶属关系（`\\in`）和 `\\pm` 的支持
- 更好地保留重要边界情况，例如下标答案（`52_8`）和文本大小写（`\\text{Evelyn}`）

行为差异示例：

- `52_8` -> 章节解析路径通常解析为 `528`；混合解析器保留 `52_8`
- `11,\\! 111,\\! 111,\\! 100` -> 章节解析路径可能变成元组；混合解析器规范化为 `11111111100`
- `(0,9) \\cup (9,36)` -> 章节解析路径通常保留为文本；混合解析器返回符号并集

取舍：

- 章节解析器：更简单、更快，也更容易理解
- 混合解析器：对 LaTeX 边界情况的覆盖更好，但规则和复杂度更多；同时增加 SymPy LaTeX 后端依赖

&nbsp;
## 使用方法

可以直接从包中导入混合解析器：

```python
from reasoning_from_scratch.bonus.parser import normalize_text_hybrid, sympy_parser_hybrid
```

更多使用示例请参阅 [compare_with_current_parser.ipynb](compare_with_current_parser.ipynb)。
