# Song Yuchen

**哈尔滨工业大学（HIT）研究生 · 大模型推理与系统优化（efficient LLM inference & systems）**

我的研究方向是**高效大模型推理**——把量化、KV-cache 管理、投机解码这些在生产推理系统里最关键的技术，做成**读得懂、跑得起来、可复现**的工程实现。偏好离线优先、依赖精简、可落地的开源方案，而不是难以复现的一次性脚本。

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white" />
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" />
  <img alt="Focus" src="https://img.shields.io/badge/focus-LLM%20inference%20%26%20systems-6E56CF?style=flat" />
</p>

---

## 🔬 研究方向

- **训练后量化（PTQ）** — 对称 / 非对称 INT8、逐通道量化、INT4 分组量化与 nibble 打包，权衡精度与显存。
- **KV-cache 管理** — 分页显存管理、块表（block table）与分页注意力（paged attention），思路对齐 vLLM 的 PagedAttention。
- **投机解码（speculative decoding）** — draft/target 双模型 + 精确接受/拒绝采样，保证输出分布与目标模型严格一致（无损加速）。
- **工程取向** — 可复现、离线可跑、依赖精简、带测试与 CI 的参考实现。

---

## 📦 代表项目

### [liteinfer](https://github.com/luke-ward88/liteinfer) · 大模型高效推理工具包

纯 Python 的高效推理**参考实现**，把生产推理系统里最常见的三类加速手段拆成读得懂、可验证的内核。默认只依赖 `numpy`，离线即可运行，PyTorch 为可选后端；面向教学与实验，而非替代 vLLM / TensorRT-LLM。

| 模块 | 内容 |
| --- | --- |
| `quant` | 对称/非对称 INT8、逐通道量化、INT4 分组量化（nibble 打包），量化线性层前向与显存核算 |
| `kvcache` | 块分配器、块表、分页注意力，对齐 PagedAttention 的分页 KV-cache 思路 |
| `specdec` | draft/target 投机解码 + 精确接受/拒绝采样，输出分布与目标模型严格一致 |

配套命令行 demo（`liteinfer quant / kv / spec / bench`）、文档（架构总览、使用指南、设计笔记、API 参考）、示例脚本与 pytest 测试，附 CI、ruff、mypy 与 pre-commit 工程配置。

`Python` · `NumPy` · `PTQ / INT8 / INT4` · `Paged KV-cache` · `Speculative Decoding` · `MIT`

---

## 🛠️ 技术栈

| 方向 | 关键词 |
| --- | --- |
| 语言 | Python |
| 科学计算 | NumPy · PyTorch（可选后端） |
| 推理技术 | 训练后量化 · 分页 KV-cache · 投机解码 |
| 工程 | pytest · ruff · mypy · pre-commit · GitHub Actions CI |

---

## 🌱 近期方向

- 扩展量化算子的精度/显存基准，覆盖更多分组粒度与打包方案。
- 打磨分页注意力与投机解码在长序列、不同提议长度（gamma）下的接受率评测。
- 持续沉淀「高效推理」相关的可复现参考实现与设计笔记。

---

<sub>偏好可复现、离线优先、工程可落地的开源实现。</sub>
