# 教育领域大模型可信问答系统

> LLM Fine-tuning + 检索增强 + Workflow 可信回答

本项目面向初中科学教材问答场景，构建集 QLoRA 微调、检索增强、Workflow 控制、可信回答和自动评测于一体的教育领域大模型问答系统。

项目重点解决通用大模型在教材问答中容易出现的回答无依据、资料不足时强行生成、引用不稳定和学生问题口语化等问题，使模型能够基于教材证据回答问题，并在缺少有效资料时进行拒答。

系统整体采用“数据构造 -> QLoRA 微调 -> 检索增强 -> Workflow 控制 -> 固定评测 -> 失败分析 -> 数据补丁迭代”的主线展开。RAG 提供可追溯证据，QLoRA 用于教育问答风格和拒答能力适配，Workflow 负责把问题分类、检索、上下文判断、生成和拒答拆分为可观测流程。

## 项目亮点

| 模块 | 说明 |
| --- | --- |
| QLoRA 教育问答微调 | 基于 Qwen2.5-3B-Instruct 构造 SFT 数据，覆盖教材依据问答、来源引用、资料不足拒答、Query Rewrite 和反重复样本。 |
| 检索增强与长文本上下文 | 结合 BM25+Dense、RRF、Rerank 和 Small-to-Big，让模型回答时可以使用可追溯的教材片段和父段落上下文。 |
| Workflow 可信回答控制 | 将问题分类、会话记忆、Query Rewrite、检索策略选择、上下文充分性判断、回答生成和拒答控制拆成可观测节点。 |
| 自动评测与失败分析 | 使用固定题集、hard-refusal 专项集和端到端 smoke test 跟踪关键词覆盖、引用完整率、结构稳定率、拒答正确率和来源错配风险。 |

## 系统流程

用户问题 -> 问题分类 -> Query Rewrite -> BM25+Dense 检索 -> RRF/Rerank -> Small-to-Big -> `context_guard` -> QLoRA/后端模型生成 -> 来源引用或拒答 -> 日志评测

## 技术栈

核心主线：

```text
Python、Qwen2.5-3B-Instruct、LLaMA Factory、LoRA/QLoRA、SFT、
bge-m3、BM25+Dense、RRF、Rerank、Small-to-Big、Workflow、FastAPI、Streamlit
```

工程支撑：

```text
ChromaDB、Redis、Docker Compose、LangGraph、本地状态机 fallback、CSV/JSONL/Markdown 评测报告
```

## 当前进展

- 已构建初中科学教材知识库导入、文档切分、Embedding、ChromaDB 索引和 RAG 问答链路。
- 已支持 `vector`、`dense_rerank`、`bm25_hybrid` 三种检索模式，以及 Small-to-Big 长文本上下文扩展。
- 已实现 LangGraph Workflow 和本地状态机 fallback，支持会话记忆、Query Rewrite、上下文判断、来源引用和无资料拒答。
- 已提供 FastAPI 服务、Streamlit 页面、Redis 缓存限流和 Docker Compose 部署方式。
- 已生成 v2.1 SFT 数据集：总计 3000 条，train/dev/test 为 2400/300/300，其中包含 hard-refusal 与 anti-repetition 数据补丁。
- 已完成 Qwen2.5-3B-Instruct v2.1 QLoRA 训练与生成质量评测，并接入 OpenAI-compatible `local_sft` 后端进行端到端验证。

## 评测结果

SFT 生成质量评测使用固定 50 题质量集和 100 题 hard-refusal 专项集。核心结果如下：

| run | 固定题平均关键词 | 引用完整率 | 结构完整率 | 重复段落率 | 拒答正确率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| v2 QLoRA fixed50 | 0.7600 | 1.00 | 1.00 | n/a | 0.0000 |
| v2.1 QLoRA fixed50 | 0.8317 | 1.00 | 1.00 | 0.00 | 0.0909 |
| v2.1 QLoRA hard-refusal100 | 0.6225 | 1.00 | 1.00 | 0.00 | 0.9300 |

v2.1 在保持引用完整率和结构稳定率的同时，提高了固定题关键词覆盖，并在 hard-refusal 专项集上明显改善资料不足拒答能力。固定题中的部分泛化型无依据问题仍可能诱发强答，因此系统在 RAG/Workflow 层加入 `context_guard.py`，当检索内容不能直接支持问题时返回无来源拒答，降低无依据回答风险。

端到端 Workflow guard smoke test 覆盖 `/rag_chat`、LangGraph Workflow 和本地 Workflow 三条路径，共 24 条样例，当前通过率为 1.0，错误数为 0。详细报告见 `eval_results/rag_workflow_guard/`。

## API 接口

| 接口 | 用途 |
| --- | --- |
| `GET /health` | 查看服务状态、当前模型后端、运行时控制状态和可用模式。 |
| `POST /chat` | 基础问答接口，支持 mock、API 或 local_sft 生成后端。 |
| `POST /rag_chat` | 检索增强问答接口，返回回答、来源、检索片段和上下文充分性判断。 |
| `POST /agent/chat` | Workflow/Agent 问答接口，支持会话记忆、Query Rewrite、检索策略选择和可观测执行轨迹。 |
| `GET /agent/session/{session_id}` | 查看指定会话的短期记忆快照。 |
| `DELETE /agent/session/{session_id}` | 重置指定会话的短期记忆。 |

详细请求字段和复现命令见 [docs/reproduce.md](docs/reproduce.md)。

## 运行方式

安装依赖：

```bash
pip install -r requirements.txt
```

复制环境变量示例：

```bash
cp .env.example .env
```

构建知识库：

```bash
USE_MOCK=true python3 build_kb.py
```

启动 FastAPI：

```bash
uvicorn app:app --reload
```

启动 Streamlit：

```bash
streamlit run web_demo.py
```

也可以通过 Docker Compose 启动 FastAPI、Streamlit 和 Redis：

```bash
docker compose up --build
```

## 能力覆盖

- 应用开发：FastAPI 服务、Streamlit 展示、Docker Compose 部署、Redis 缓存限流。
- RAG 与可信回答：教材知识库、BM25+Dense、RRF、Rerank、Small-to-Big、来源引用和上下文充分性判断。
- 模型微调：SFT 数据构造、LLaMA Factory 配置、Qwen2.5-3B-Instruct LoRA/QLoRA 训练和 local_sft 接入。
- 评测复盘：固定题集、hard-refusal 专项集、端到端 Workflow guard、日志记录和失败样例分析。

## 目录结构

```text
.
├── app.py                         # FastAPI 服务入口
├── web_demo.py                    # Streamlit 演示页面
├── rag_pipeline.py                # RAG 主流程
├── rag_agent.py                   # 本地 Workflow fallback
├── langgraph_agent.py             # LangGraph Workflow 编排
├── context_guard.py               # 上下文充分性判断与无资料拒答控制
├── build_kb.py                    # 知识库构建入口
├── data/                          # 原始资料、chunks、SFT 数据和评测题
├── eval_results/                  # 评测输出和对比报告
├── training/llamafactory/         # LLaMA Factory 训练与推理配置
└── docs/                          # 复现、项目复盘和截图清单
```

## 运行与复现文档

- [docs/reproduce.md](docs/reproduce.md)：环境安装、知识库构建、API/页面启动、SFT 数据生成、训练配置、评测命令和 local_sft 接入。
- [docs/training_reproduction.md](docs/training_reproduction.md)：QLoRA 训练配置、数据规模、服务接入和核心评测指标。
- [docs/final_project_summary.md](docs/final_project_summary.md)：项目背景、技术路线、数据构造、模型微调、检索增强、Workflow、评测结果和局限。
- [docs/screenshots_checklist.md](docs/screenshots_checklist.md)：建议保留的页面、接口、评测和训练截图清单。
