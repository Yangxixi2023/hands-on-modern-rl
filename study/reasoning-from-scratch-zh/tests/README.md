# 测试

本目录包含仓库的 Python 测试套件。

## 本地运行

先安装开发环境：

```bash
uv sync --group dev
```

### 1. 常规测试套件，忽略高开销测试（推荐）

建议使用此方式快速测试和开发新功能。

```bash
SKIP_EXPENSIVE=1 RUN_REAL_DOWNLOAD_TESTS=0 uv run pytest tests
```

运行单个测试文件：

```bash
SKIP_EXPENSIVE=1 RUN_REAL_DOWNLOAD_TESTS=0 uv run pytest tests/test_ch03.py
```


这是本地运行方式中最接近默认 GitHub 测试矩阵的一种。

### 2. 常规测试套件加高开销测试

有些代码默认被忽略，因为运行开销较高。完成基本调试后，建议运行这些测试。

```bash
SKIP_EXPENSIVE=0 RUN_REAL_DOWNLOAD_TESTS=0 uv run pytest tests
```

请注意，这会运行测试文件中由 `SKIP_EXPENSIVE` 控制的测试，但仍排除实际下载大型模型检查点的真实网络/下载集成测试。

### 3. 仅运行下载测试

有些测试会检查模型检查点文件是否可下载，以及服务器是否（仍然）正常工作。无需在本地或定期运行这些测试；它们主要用于偶尔检查。

运行这些下载测试：

```bash
SKIP_EXPENSIVE=0 RUN_REAL_DOWNLOAD_TESTS=1 uv run pytest tests -k real_download
```

其工作方式如下：

- `pytest tests` 从 `tests/` 目录收集测试
- `-k real_download` 只保留名称中包含 `real_download` 的测试

如果希望更有针对性地运行，例如直接运行附录 D 的真实快照测试，可以使用：

```bash
SKIP_EXPENSIVE=0 RUN_REAL_DOWNLOAD_TESTS=1 uv run pytest tests/test_appendix_d.py -k real_download_1_7b
```

目前需要主动启用的真实下载测试覆盖以下内容：

- `tests/test_ch03.py`：真实的 `math500_test.json` 下载和分词器下载
- `tests/test_ch06.py`：真实的数学训练集下载
- `tests/test_ch07.py`：真实的 GitHub raw 文件下载
- `tests/test_ch08.py`：真实的蒸馏数据集和分词器下载
- `tests/test_appendix_d.py`：真实的 `Qwen/Qwen3-1.7B-Base` 快照下载
- `tests/test_qwen3.py`：真实的 `Qwen/Qwen3-0.6B` 分词器比较


### 4. 全部测试（不推荐）

这会运行测试套件中的全部测试。请注意，它同时包含计算开销较高的测试（第 3 节）和高开销下载测试（第 4 节）。

```bash
SKIP_EXPENSIVE=0 RUN_REAL_DOWNLOAD_TESTS=1 uv run pytest tests
```

不建议在代码修改后的日常测试中使用此方式，因为文件下载开销很高，没有必要定期运行。


## GitHub CI 中运行的内容

默认 GitHub 测试矩阵运行常规测试套件，并省略开销较高的测试：

- `.github/workflows/tests-linux.yml`
- `.github/workflows/tests-macos.yml`
- `.github/workflows/tests-windows.yml`
- `.github/workflows/basic-tests-pip.yml`

这些工作流设置 `SKIP_EXPENSIVE=1`，因此其中会跳过高开销测试。原因是 GitHub CI 没有运行高开销测试所需的计算资源（例如 GPU）。

真实网络/下载集成测试在单独的工作流中运行：

- `.github/workflows/real-download-tests.yml`

该工作流设置 `RUN_REAL_DOWNLOAD_TESTS=1`，且只运行由 `-k real_download` 选中的测试。
它不属于默认的 PR/push 测试矩阵。它按每周计划运行，也可以通过 `workflow_dispatch` 手动启动。
