# 架构说明：教育领域大模型可信问答系统

本文档面向公开仓库阅读者，说明系统目标、核心流程、模块职责和当前边界。项目主线是“教育可信问答闭环”：通过 SFT 数据构造、Qwen2.5-3B-Instruct LoRA/QLoRA 微调、教材证据检索、上下文充分性判断、来源引用和自动评测，让回答尽量做到有依据、可追溯、可评测、可复盘。

## 1. 项目目标

项目面向初中科学教材问答场景，重点处理以下问题：

- 通用模型在教材问答中容易脱离资料依据。
- 资料不足时，模型可能复述低相关片段或强行生成。
- 学生问题常带有口语化表达、追问和指代。
- 问答效果需要通过固定评测、失败分析和数据补丁持续验证。

系统采用“数据构造 -> QLoRA 微调 -> 检索增强 -> Workflow 控制 -> 固定评测 -> 失败分析 -> 数据补丁迭代”的路线。RAG 提供证据，SFT 适配教育问答和拒答格式，Workflow 负责把检索、判断、生成和拒答拆分为可观测步骤。

## 2. 总体流程

### Workflow 问答流程

```text
用户问题
-> 会话记忆补全
-> 问题类型识别
-> 检索策略选择
-> Query Rewrite（可选）
-> BM25+Dense / Vector 检索
-> RRF 融合与 Rerank
-> Small-to-Big 上下文扩展
-> context_guard 上下文充分性判断
-> 模型生成或无资料拒答
-> 来源引用
-> Trace 日志与评测记录
```

### SFT 数据与训练流程

```text
教材 chunks / 评测题 / 拒答样例
-> 构造 grounded QA、citation QA、compare/reasoning、query rewrite、refusal 样本
-> Alpaca JSONL train/dev/test 切分
-> LLaMA Factory dataset_info 注册
-> Qwen2.5-3B-Instruct LoRA/QLoRA 训练
-> OpenAI-compatible local_sft endpoint 接入 RAG / Workflow
-> 固定题和 hard-refusal 专项评测
```

### RAG 检索流程

```text
文档读取
-> small/big chunks 切分
-> Embedding 向量化
-> ChromaDB 索引构建
-> Vector / BM25 Hybrid 候选召回
-> RRF 融合
-> Rerank 精排
-> Small-to-Big 父段落扩展
-> Prompt 构造
-> 回答生成、引用和日志记录
```

## 3. 核心模块

### 知识库与检索

- `document_loader.py`：递归读取 `data/raw_docs/` 下的教育资料和教材片段。
- `domain_kb_importer.py`：从教材 Markdown 中抽样导入初中科学语料，并保留年级、册次、章节等元数据。
- `text_splitter.py`：生成 small chunks 和 big chunks，建立父子映射。
- `embedding_client.py`：调用 embedding 模型生成文本向量。
- `vector_store.py`：构建和查询 ChromaDB 向量索引。
- `hybrid_retriever.py`：实现 dense 检索、BM25 稀疏召回和 RRF 融合。
- `reranker.py`：对候选片段进行重排序。
- `long_text_context.py`：将命中的 small chunks 扩展到父级 big chunks。

### RAG 与 Workflow

- `rag_pipeline.py`：检索、上下文构造、模型调用、来源引用和检索日志的主流程。
- `context_guard.py`：判断问题是否能被检索上下文支持；不能支持时返回无来源拒答。
- `agent_memory.py`：维护会话级短期记忆，用于多轮追问中的主题延续和指代补全。
- `agent_skills.py`：封装分类、策略选择、改写、检索、上下文判断、生成和记忆更新等能力。
- `rag_agent.py`：本地状态机 Workflow fallback。
- `langgraph_agent.py`：LangGraph Workflow 编排版本，返回 graph trace 和 skill trace。
- `runtime_controls.py`：Redis 缓存和固定窗口限流；Redis 不可用时 fail-open。

### 服务与展示

- `app.py`：FastAPI 服务入口，提供 `/chat`、`/rag_chat`、`/agent/chat`、`/health` 等接口。
- `web_demo.py`：Streamlit 页面，展示 Agent 问答、RAG 问答、评测结果和日志。
- `docker-compose.yml`：一键启动 API、Web 和 Redis。

### 数据构造与评测

- `sft_dataset_builder.py`：首版小规模 Alpaca SFT 数据构造。
- `sft_dataset_builder_v2.py`：v2 规模化数据构造。
- `sft_dataset_builder_v21.py`：v2.1 hard-refusal 和 anti-repetition 数据补丁。
- `sft_generation_eval_runner.py`：OpenAI-compatible 生成质量评测。
- `rag_workflow_guard_eval_runner.py`：RAG/Workflow guard 端到端评测。
- `science_long_text_eval_runner.py`：Small-to-Big 与 small-only 上下文对比评测。
- `science_failure_analyzer.py`、`log_analyzer.py`：失败样例归因分析。

## 4. 数据与训练状态

当前数据集版本：

- `data/sft/`：首版小规模数据。
- `data/sft_v2/`：2400 条 SFT 数据，train/dev/test 为 1920/240/240。
- `data/sft_v21/`：3000 条 SFT 数据，train/dev/test 为 2400/300/300，包含 480 条 hard-refusal 与 120 条 anti-repetition 补丁样本。

训练配置位于 `training/llamafactory/`，覆盖 QLoRA/LoRA SFT 和 API serving。大模型权重、adapter、checkpoint 和临时训练运行目录不作为公开仓库内容。

## 5. 接口概览

| 接口 | 说明 |
| --- | --- |
| `GET /health` | 服务状态、当前模型后端、运行时控制状态和可用模式。 |
| `POST /chat` | 基础问答，支持 mock、API、local_sft 后端。 |
| `POST /rag_chat` | 检索增强问答，返回回答、来源、检索片段和上下文充分性判断。 |
| `POST /agent/chat` | Workflow 问答，支持会话记忆、Query Rewrite、策略选择和执行轨迹。 |
| `GET /agent/session/{session_id}` | 查看会话记忆。 |
| `DELETE /agent/session/{session_id}` | 重置会话记忆。 |

详细运行方式见 `docs/reproduce.md`。

## 6. 评测口径

项目使用多类评测验证效果：

- SFT fixed50：评估关键词覆盖、引用完整率、结构完整率、重复段落率和拒答正确率。
- hard-refusal100：评估资料不足、实时信息、隐私、未记录事实等问题的拒答稳定性。
- RAG/Workflow guard：验证 `/rag_chat`、LangGraph Workflow、本地 Workflow 三条路径在支持性问题和不支持性问题上的行为。
- Small-to-Big 对比：评估长文本父段落扩展是否改善教材问题回答。

当前公开结果应理解为“降低无依据回答风险、提升拒答稳定性”，不是对所有开放问题的绝对保证。

## 7. 当前边界

- 当前知识库以抽样教材和项目资料为主，尚未覆盖全量教材。
- 评测集规模有限，后续可以扩展更多跨章节、跨年级、多轮追问和实验现象问题。
- 当前服务化重点是本地演示和工程闭环，生产环境还需要补充认证、权限、监控、数据治理和更完整的部署方案。
- ChromaDB、Redis 和 Docker Compose 适合当前规模；更大规模场景可替换为专门的向量检索和服务治理组件。
