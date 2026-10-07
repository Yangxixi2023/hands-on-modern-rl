# 排行榜排名

本补充材料用两种不同的方法，根据成对比较结果构建 LM Arena（原名 Chatbot Arena）风格的排行榜。

两种实现都通过 `--path` 参数从 JSON 文件读取成对偏好（左侧为胜者，右侧为败者）。下面是所提供 [votes.json](votes.json) 文件的节选：

```json
[
  ["GPT-5", "Claude-3"],
  ["GPT-5", "Llama-4"],
  ["Claude-3", "Llama-3"],
  ["Llama-4", "Llama-3"],
  ...
]
```

<br>

---

**注意**：如果你不使用 `uv`，请在下面的示例中将 `uv run ...py` 替换为 `python ...py`。

---

&nbsp;
## 方法 1：Elo 评分

- 实现流行的 Elo 评分方法（受国际象棋排名启发），这是 LM Arena 最初采用的方法
- 详情请参阅[主 Notebook](../01_main-chapter-code/chF_main.ipynb)

```bash
➜  03_leaderboards git:(main) ✗ uv run 1_elo_leaderboard.py --path votes.json

Leaderboard (Elo)
-----------------------
 1. GPT-5       1095.9
 2. Claude-3    1058.7
 3. Llama-4      958.2
 4. Llama-3      887.2
```

&nbsp;
## 方法 2：Bradley-Terry 模型

- 实现 [Bradley-Terry 模型](https://en.wikipedia.org/wiki/Bradley–Terry_model)，类似于新的 LM Arena 排行榜；后者的定义见官方论文（[Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132)）
- 与 LM Arena 排行榜一样，分数会重新缩放，使其接近原始 Elo 分数
- 此处代码使用 PyTorch 的 Adam 优化器拟合模型，以便读者熟悉代码并提高可读性

```bash
➜  03_leaderboards git:(main) ✗ uv run 2_bradley_terry_leaderboard.py --path votes.json

Leaderboard (Bradley-Terry)
-----------------------------
 1. GPT-5       1140.6
 2. Claude-3    1058.7
 3. Llama-4      950.3
 4. Llama-3      850.4
```
