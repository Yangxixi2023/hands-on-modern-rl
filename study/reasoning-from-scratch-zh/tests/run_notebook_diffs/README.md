# Notebook 输出比较

这些便捷脚本使用两个不同的 PyTorch 版本运行 Notebook，并创建一份 Markdown 报告，记录发生变化的单元格源代码和输出，以便调查差异。这主要用于手动兼容性检查，不会由 pytest 或 GitHub CI 运行。

&nbsp;
## 使用两个 PyTorch 版本运行 Notebook

在仓库根目录运行：

```bash
uv run python tests/run_notebook_diffs/run.py \
  --torch-version 2.7.1 \
  --torch-version 2.13.0 \
  ch02/01_main-chapter-code/ch02_main.ipynb
```

`uv` 会针对每个版本创建一个安装了基础依赖的独立环境。

可以使用 `--with` 添加额外的 Notebook 依赖：

```bash
uv run python tests/run_notebook_diffs/run.py \
  --torch-version 2.7.1 \
  --torch-version 2.13.0 \
  --with transformers \
  --with datasets \
  ch08/01_main-chapter-code/ch08_main.ipynb
```

如果任一 PyTorch 版本没有针对默认 Python 版本提供 wheel，也可以使用 `--python 3.11`。

还可以在一次命令中传入多个 Notebook。例如：

```bash
notebooks=(
  ch02/01_main-chapter-code/ch02_main.ipynb
  ch02/01_main-chapter-code/ch02_exercise-solutions.ipynb

  ch03/01_main-chapter-code/ch03_main.ipynb
  ch03/01_main-chapter-code/ch03_exercise-solutions.ipynb
  ch03/03_advanced-parser/compare_with_current_parser.ipynb

  ch04/01_main-chapter-code/ch04_main.ipynb
  ch04/01_main-chapter-code/ch04_exercise-solutions.ipynb

  ch05/01_main-chapter-code/ch05_main.ipynb
  ch05/01_main-chapter-code/ch05_exercise-solutions.ipynb

  ch06/01_main-chapter-code/ch06_main.ipynb
  ch06/01_main-chapter-code/ch06_exercise-solutions.ipynb

  # ch07/01_main-chapter-code/ch07_main.ipynb
  # ch07/01_main-chapter-code/ch07_exercise-solutions.ipynb

  ch08/01_main-chapter-code/ch08_main.ipynb
  ch08/01_main-chapter-code/ch08_exercise-solutions.ipynb

  chC/01_main-chapter-code/chC_main.ipynb
  chD/chD_main.ipynb
  chE/chE_main.ipynb
  # chF/01_main-chapter-code/chF_main.ipynb
)

UV_PYTHON=3.13 uv run python tests/run_notebook_diffs/run.py \
  --allow-errors \
  --torch-version 2.7.1 \
  --torch-version 2.13.0 \
  "${notebooks[@]}"
```

每个 Notebook 都从自己的目录运行，因此相对路径的行为与在 Jupyter 中相同。

&nbsp;
## 结果

默认情况下，结果写入 `tests/run_notebook_diffs/results/` 下方：

```text
results/
└── ch02__01_main-chapter-code__ch02_main/
    ├── torch-2.7.1.ipynb
    ├── torch-2.13.0.ipynb
    └── comparison.md
```

`comparison.md` 的结构如下：

---

#### 单元格 23

##### 输出

```diff
--- left outputs
+++ right outputs
@@ -2,6 +2,6 @@
   {
     "name": "stdout",
     "output_type": "stream",
-    "text": "PyTorch version 2.7.1\nApple Silicon GPU\n"
+    "text": "PyTorch version 2.13.0\nApple Silicon GPU\n"
   }
 ]
```

#### 单元格 87

##### 输出

```diff
--- left outputs
+++ right outputs
@@ -207,6 +207,6 @@
   {
     "name": "stdout",
     "output_type": "stream",
-    "text": "\n\nTime: 8.20 sec\n5 tokens/sec\n"
+    "text": "\n\nTime: 8.16 sec\n5 tokens/sec\n"
   }
 ]
```

---

`results/` 目录会被 Git 忽略。也可以使用 `--output-dir PATH` 将结果写到其他位置。

默认情况下，Notebook 出错时执行会停止。对于某些出于教学目的而故意包含错误的 Notebook，可以使用 `--allow-errors` 继续执行剩余单元格。

每个单元格的默认超时为 1 小时，但可以使用 `--timeout SECONDS` 覆盖。

&nbsp;
## 比较已有 Notebook

如果已经有可用的已执行 Notebook，也可以独立使用比较器：

```bash
uv run python tests/run_notebook_diffs/compare.py \
  first.ipynb second.ipynb --output comparison.md
```
