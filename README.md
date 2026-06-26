# 教育领域 RAG Agent 问答与检索优化系统

## 一、项目背景

本项目是一个教育领域知识库问答场景的大模型 RAG Agent 系统，基于 Python、FastAPI、ChromaDB、LangGraph、Redis 和 Streamlit 构建。

系统可面向企业制度文档、产品手册、培训资料、客服 FAQ 等场景，当前使用教育资料和初中科学教材作为实验语料，模拟企业内部文档问答流程。目标不是普通聊天，而是让大模型回答做到“有依据、可追溯、可评测、可复盘”。

系统支持 LangGraph Skills Agent、本地状态机 Agent fallback、会话记忆、垂直领域知识库导入、Long-text RAG、Small-to-Big 上下文扩展、本地资料读取、文本切分、Embedding 向量化、向量检索、Dense Rerank、BM25 Hybrid、Rerank 重排序、Query Rewrite、大模型回答生成、来源引用、Redis 缓存限流、Docker Compose 部署、日志记录、自动评测和可视化展示。

项目目标不是单纯实现一个问答 Demo，而是构建一个具备 Agent 工作流、垂直知识库构建、检索优化、结果溯源、效果评测和失败样例分析能力的 RAG 工程项目。

当前知识库包含两类资料：

* 大模型应用开发资料：RAG、Agent、Prompt Engineering、Embedding、Rerank、BM25 Hybrid 等。
* 初中科学教材资料：从 `ch-3` 项目的沪教版初中科学 Markdown 教材中均衡抽样导入，覆盖 7-9 年级上下册的科学知识点。

---

## 二、技术栈

* Python
* FastAPI
* Streamlit
* LangGraph
* LangChain Core
* ChromaDB
* Redis
* Docker Compose
* sentence-transformers
* SiliconFlow / DeepSeek API
* Prompt Engineering
* Agent Skills / Tool Use
* Conversation Memory
* Vertical Domain Knowledge Base
* Long-text RAG
* Small-to-Big Retrieval
* Query Rewrite
* JSON / JSONL / CSV
* BM25 Hybrid Search
* Rerank
* RAG Evaluation

---

## 三、系统功能

### 1. LangGraph Skills Agent 问答

系统新增 LangGraph Skills Agent，并保留本地状态机 Agent 作为 fallback。Agent 会先读取会话记忆并补全多轮追问，再识别用户问题类型、选择检索策略，必要时执行 Query Rewrite 和二次检索，最后判断上下文是否足够生成回答。

Agent Skills 包括：

* `resolve_memory`：读取 session 级短期记忆，补全“它、这个技术、前者、后者”等追问；
* `classify_query`：识别概念解释、对比分析、学习建议、资料缺失、模糊问题和普通问题；
* `select_strategy`：根据问题类型选择 `dense_rerank` 或 `bm25_hybrid`；
* `maybe_rewrite` / `rewrite_query`：对模糊问题或上下文不足问题进行 Query Rewrite；
* `retrieve_context`：调用现有 RAG 检索能力；
* `judge_context`：判断上下文是否足够；
* `generate_answer`：基于资料生成回答或执行资料不足拒答；
* `update_memory`：写入会话记忆；
* `finalize_response`：统一整理 Agent Trace、Graph Trace 和回答结果。

Query Rewrite 采用 LLM 优先、规则兜底的可控设计：真实 API 模式下由大模型把模糊问题改写成更适合检索的一句话；Mock 模式、API 失败或 LLM 输出不可用时自动回退到规则改写，保证本地演示稳定。

Agent 执行过程包括：

```text
resolve_memory（可选）
↓
classify_query
↓
select_strategy
↓
retrieve_context
↓
judge_context
↓
rewrite_query（可选）
↓
generate_answer
↓
update_memory（可选）
↓
finalize_response
↓
log_trace
```

页面会展示 Agent 的任务类型、检索策略、上下文充分性、Query Rewrite 结果和工具调用轨迹，便于面试和调试时解释系统行为。

### 2. 会话记忆与多轮追问

系统新增轻量会话记忆，用于记录当前学习主题、最近问题、上一轮回答摘要、引用来源、检索策略和历史 Query Rewrite。

当用户进行多轮追问时，Agent 会使用记忆完成指代消解，例如：

```text
第一轮：什么是 RAG？
第二轮：它为什么可以减少幻觉？
记忆补全：RAG 为什么可以减少幻觉？
```

该能力主要用于提升多轮学习助手体验和 Query Rewrite 质量，不作为单轮检索指标提升来包装。

### 3. Agent API 化、缓存与限流

FastAPI 提供 `POST /agent/chat`，支持通过 `session_id` 复用会话记忆，并可选择 `agent_engine=langgraph` 或 `agent_engine=local`。

Redis 用于运行时治理：

* 热点单轮问题结果缓存，降低重复大模型调用成本；
* Agent Chat API 固定窗口限流，避免接口被高频调用；
* Redis 不可用时自动 fail-open，不影响本地脚本和页面演示。

项目提供 Docker Compose，可一键启动 FastAPI、Streamlit 和 Redis，便于把本地原型升级成可交付服务。

### 4. 基础 RAG 问答

系统支持读取本地教育资料文档，将文档切分为多个 chunk，并使用 Embedding 模型将文本片段转换为向量后写入 ChromaDB。

用户输入问题后，系统会检索相关知识片段，构造 RAG Prompt，并调用大模型生成回答。

### 5. 垂直领域知识库构建

系统新增 `domain_kb_importer.py`，用于从 `../ch-3/knowledge_base_builder/data/textbooks/初中科学/沪教版初中科学` 导入教材语料。

导入策略：

* 从 1288 篇候选 Markdown 教材文档中抽样；
* 默认导入 80 篇，按 7-9 年级上下册均衡覆盖；
* 保留教材、年级、册次、出版社、章节、内容类型和原始文件路径；
* 写入 `data/raw_docs/science_textbooks/`；
* 生成 `data/science_textbook_manifest.json` 作为导入清单。

这样项目可以从“泛学习资料 Demo”升级为“初中科学教材垂直领域问答系统”，同时仍保留原来的大模型技术资料问答能力。

### 6. Long-text RAG：Small-to-Big

系统新增 Small-to-Big 长文本 RAG 能力：

* 小 chunk：默认 500 字左右，用于向量检索、BM25 召回和 Rerank 排序；
* 大 chunk：默认 1600 字左右，按 Markdown 段落优先切分，用于回答阶段提供更完整上下文；
* 父子映射：每个小 chunk 保存 `parent_chunk_id`，命中后可扩展到对应父级大段落；
* 可观测性：返回 `small_retrieved_chunks`、`long_context`、`trigger_chunk_ids`，展示哪些小 chunk 触发了父段落；
* 可切换：API 和 Streamlit 均支持 `context_mode=small` 或 `context_mode=small_to_big`。

该能力解决“召回片段太碎、答案缺少上下文”的问题，适合教材、政策、公文制度等长文本问答场景。

### 7. 来源引用返回

系统会返回回答所依据的来源文档和 chunk_id，便于追溯答案依据，提高问答结果的可信度。

### 8. BM25 Hybrid 检索优化

系统在基础向量检索基础上新增 BM25 稀疏召回能力，将向量召回结果和 BM25 召回结果通过 RRF 进行融合，形成 BM25 Hybrid 候选召回结果。

### 9. Hybrid-First BM25 Hybrid + Rerank

系统提供 `dense_rerank` 和 `bm25_hybrid` 两种优化实验模式，用于对比无稀疏召回和 BM25 稀疏召回的效果。

针对垂直领域知识库，系统采用 Hybrid-First BM25 Hybrid + Rerank 策略：

```text
向量召回 + BM25 稀疏召回
↓
RRF 融合候选
↓
轻量级 Rerank 重排序
↓
融合 Rerank 分与 BM25/Hybrid 检索分
↓
优先选择 Hybrid/Rerank 结果，dense 仅作兜底补充
```

该策略既保留了向量检索的语义兜底能力，也让教材术语、章节名和专有名词的 BM25 精确命中在最终排序中有更高权重。

### 10. 自动评测

系统构建了 60 条 RAG 测试问题，覆盖概念解释、原理机制、对比辨析、教育应用和资料缺失等类型。

系统同时新增 `data/science_rag_test_questions.json`，用于验证初中科学教材知识库的来源命中、关键词覆盖和资料缺失拒答能力。科学测试集当前包含 30 条问题，覆盖概念解释、机制说明、实验现象、对比分析、生活应用和资料缺失。

评测指标包括：

* 来源命中率
* 关键词命中率
* 引用完整率
* 无资料拒答率
* 平均回答长度
* 平均检索片段数

### 11. 日志记录与失败样例分析

系统会记录 RAG 问答日志、检索日志和 Agent 执行日志，包括检索模式、候选片段、最终上下文、来源引用、回答长度、工具调用轨迹和上下文判断结果等信息。

同时，项目提供失败样例分析脚本，用于定位检索失败、回答覆盖不足、Prompt 约束不足、知识库缺失等问题。科学教材评测会额外生成 `eval_results/science_failure_cases.md`，区分硬性失败和“严格关键词未命中但同义关键词命中”的软性观察。

### 12. Streamlit 可视化展示

项目提供 Streamlit 页面，支持：

* Agent 问答演示
* 启用或清空 Agent 会话记忆
* 展示 Agent 问题类型、检索策略和工具调用轨迹
* 输入问题
* 选择检索模式：vector / dense_rerank / bm25_hybrid
* 设置 top_k
* 设置 candidate_k
* 启用或关闭 Rerank
* 展示模型回答
* 展示引用来源
* 展示检索片段
* 展示 dense_score、bm25_score、hybrid_score、rerank_score
* 展示评测结果
* 查看检索日志

---

## 四、项目结构

```text
edu-rag-assistant/
├── app.py
├── web_demo.py
├── agent_memory.py
├── agent_session_store.py
├── agent_state.py
├── agent_skills.py
├── rag_agent.py
├── langgraph_agent.py
├── runtime_controls.py
├── llm_client.py
├── config.py
├── prompt_templates.py
├── document_loader.py
├── text_splitter.py
├── domain_kb_importer.py
├── long_text_context.py
├── embedding_client.py
├── vector_store.py
├── hybrid_retriever.py
├── reranker.py
├── rag_pipeline.py
├── rag_logger.py
├── rag_evaluator.py
├── experiment_runner.py
├── agent_experiment_runner.py
├── memory_experiment_runner.py
├── log_analyzer.py
├── DEV_SPEC.md
├── data/
│   ├── raw_docs/
│   │   └── science_textbooks/
│   ├── chunks/
│   │   ├── chunks.json
│   │   └── big_chunks.json
│   ├── rag_test_questions.json
│   ├── science_rag_test_questions.json
│   ├── science_textbook_manifest.json
│   └── memory_eval_questions.json
├── vector_db/
├── logs/
│   ├── rag_log.json
│   ├── retrieval_log.json
│   ├── agent_trace_log.json
│   └── agent_sessions.json
├── eval_results/
│   ├── baseline_eval.csv
│   ├── no_keyword_dense_rerank_eval.csv
│   ├── bm25_hybrid_rerank_eval.csv
│   ├── bm25_comparison_summary.json
│   ├── bm25_comparison_report.md
│   ├── bm25_failure_cases.md
│   └── science_failure_cases.md
├── assets/
│   └── rag_demo.png
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 五、核心流程

Agent 主流程：

```text
用户问题
↓
读取会话记忆并补全多轮追问
↓
LangGraph Skills 编排 / 本地状态机 fallback
↓
问题类型识别
↓
检索策略选择
↓
Query Rewrite（可选）
↓
向量召回 / BM25 Hybrid / Rerank
↓
上下文充分性判断
↓
回答生成或资料不足拒答
↓
来源引用
↓
Agent Trace 日志
```

RAG 检索流程：

```text
文档读取
↓
文本切分
↓
生成 small chunks 和 big chunks
↓
Embedding 向量化
↓
向量库构建
↓
向量召回 / BM25 召回
↓
BM25 Hybrid Search
↓
Rerank 重排序
↓
Small-to-Big 父段落扩展
↓
Prompt 构造
↓
大模型生成回答
↓
来源引用
↓
日志记录
↓
自动评测
↓
Dashboard 展示
```

垂直知识库构建流程：

```text
ch-3 初中科学教材 Markdown
↓
均衡抽样与元数据提取
↓
写入 data/raw_docs/science_textbooks/
↓
document_loader 递归读取
↓
text_splitter 生成 chunks.json
↓
vector_store 写入 ChromaDB
↓
Agent / RAG 检索问答
```

---

## 六、模型说明

本项目使用两类模型：

### 1. Embedding 模型

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

用于将教育资料文本和用户问题转换为向量，并基于 ChromaDB 进行相似度检索。

### 2. 生成式大模型

项目通过 SiliconFlow API 接入 DeepSeek 模型，用于根据检索到的上下文生成最终回答。

项目支持 Mock 模式：

```text
USE_MOCK=true
```

也支持真实 API 模式：

```text
USE_MOCK=false
```

---

## 七、评测设计

测试集文件：

```text
data/rag_test_questions.json
data/science_rag_test_questions.json
```

主测试集共 60 条问题，科学教材测试集共 30 条问题，覆盖以下类型：

| 问题类型        | 说明    |
| ----------- | ----- |
| concept     | 概念解释类 |
| mechanism   | 原理机制类 |
| compare     | 对比辨析类 |
| application | 教育应用类 |
| missing     | 资料缺失类 |

每条测试样例包含：

```json
{
  "id": 1,
  "question": "什么是 RAG？",
  "expected_source": "rag_intro.md",
  "expected_keywords": ["检索", "生成", "知识库"],
  "question_type": "concept"
}
```

---

## 八、实验结果

本项目当前默认对 `dense_rerank` 和 `bm25_hybrid` 两组方案进行对比实验。

`dense_rerank` 作为无稀疏召回对照组，只使用向量召回候选片段并进行 Rerank。

`bm25_hybrid` 在向量召回基础上加入 BM25 稀疏召回，并通过 RRF 融合后进行 Rerank。

运行 `experiment_runner.py` 后会生成最新指标表和差值分析。

详细实验报告见：

```text
eval_results/bm25_comparison_report.md
```

失败样例分析见：

```text
eval_results/bm25_failure_cases.md
eval_results/science_failure_cases.md
```

## 简历表达建议

**企业知识库 RAG Agent 问答与检索优化系统**

* 构建面向企业内部文档问答场景的 RAG Agent 系统，支持文档切分、Embedding 向量化、ChromaDB 检索、BM25 Hybrid、Rerank、来源引用和资料缺失拒答。
* 使用 LangGraph 将问题分类、Query Rewrite、检索、上下文判断、回答生成和记忆更新封装为 Skills，并保留本地状态机 Agent 作为 fallback。
* 设计 session 级短期记忆机制，支持多轮追问中的指代消解和主题延续，并通过 `/agent/chat` API 对外提供会话复用能力。
* 实现 Small-to-Big 长文本 RAG，小 chunk 用于召回，父级 big chunk 用于回答，提升企业制度、产品手册、培训资料等长文档问答的上下文完整性。
* 引入 Redis 实现热点问答缓存和接口限流，并通过 Docker Compose 支持 FastAPI、Streamlit、Redis 服务一键启动。
* 构建自动评测体系，统计来源命中率、严格/同义关键词命中率、引用完整率、资料缺失拒答率和上下文长度，用失败样例分析定位检索、改写和生成链路问题。

---

## 九、运行方式

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example`：

```bash
cp .env.example .env
```

示例配置：

```text
USE_MOCK=true
MODEL=deepseek-ai/DeepSeek-V3
API_KEY=your_api_key_here
BASE_URL=https://api.siliconflow.cn/v1/chat/completions
```

### 3. 一键构建知识库（推荐）

Mock 模式下跑通完整构建链路：

```bash
USE_MOCK=true python build_kb.py
```

同时运行 Small-to-Big 对比评测：

```bash
USE_MOCK=true python build_kb.py --run-eval
```

脚本会依次完成：导入 ch-3 教材语料、生成 small/big chunks、构建 Chroma 向量索引、运行 Small-to-Big 冒烟测试，并可选运行科学教材 small vs small_to_big 对比评测。

真实 API 模式下运行：

```bash
USE_MOCK=false python build_kb.py --run-eval
```

真实模式需要配置 `SILICONFLOW_API_KEY`，并会调用 Embedding、Rerank 和 Chat 模型。

### 4. 手动构建 chunks

```bash
python text_splitter.py
```

### 5. 手动构建向量库

```bash
python vector_store.py
```

### 6. 启动 FastAPI

```bash
uvicorn app:app --reload
```

访问接口文档：

```text
http://127.0.0.1:8000/docs
```

### 7. 调用 RAG 接口

```json
{
  "question": "什么是 RAG？",
  "top_k": 3,
  "retriever_mode": "bm25_hybrid",
  "context_mode": "small_to_big",
  "candidate_k": 10,
  "use_rerank": true
}
```

### 8. 调用 Agent 接口

`POST /agent/chat`

```json
{
  "question": "它为什么可以减少幻觉？",
  "session_id": "demo-session-001",
  "agent_engine": "langgraph",
  "top_k": 3,
  "candidate_k": 10,
  "max_rewrites": 1,
  "use_rerank": true,
  "context_mode": "small_to_big",
  "reset_memory": false,
  "enable_cache": false,
  "enable_rate_limit": true
}
```

Agent 接口会通过 `session_id` 复用会话记忆，并返回 `resolved_question`、`memory_used`、`memory_snapshot`、`agent_trace`、`graph_trace`、`skill_trace`、`cache_status` 和 `rate_limit`。

会话管理接口：

```text
GET /agent/session/{session_id}
DELETE /agent/session/{session_id}
```

### 9. 启动 Streamlit 页面

```bash
streamlit run web_demo.py
```

页面默认提供 Agent 问答演示，支持输入或复用 `session_id`，同时保留原始 RAG 问答、评测结果和日志查看。

### 10. Docker Compose 一键启动

```bash
docker compose up --build
```

服务地址：

```text
FastAPI: http://127.0.0.1:8000/docs
Streamlit: http://127.0.0.1:8501
Redis: redis://127.0.0.1:6379/0
```

Compose 会同时启动 API、Web 页面和 Redis，并挂载 `data/`、`vector_db/`、`logs/`、`eval_results/` 目录。

### 11. 直接运行 Agent

```bash
python rag_agent.py
```

---

## 十、自动评测

运行对比实验：

```bash
python experiment_runner.py
```

运行 Agent 版本评测：

```bash
python agent_experiment_runner.py
```

运行多轮记忆评测：

```bash
python memory_experiment_runner.py
```

运行科学教材 Small-to-Big 对比评测：

```bash
python science_long_text_eval_runner.py
```

运行工程化升级实验：

```bash
python engineering_experiment_runner.py
```

该实验用于验证 LangGraph Agent 编排、session 级记忆、Redis 缓存/限流运行状态和原 `/rag_chat` API 回归稳定性。

科学教材评测会同时输出 `exact_keyword_hit` 和 `semantic_keyword_hit`：

* `exact_keyword_hit`：严格字面关键词命中；
* `semantic_keyword_hit`：支持少量领域同义表达，例如“固态 ≈ 固体形态”、“气态 ≈ 气体形态”。
* `no_context_reject`：资料缺失题是否明确拒答，不参与来源命中率和关键词命中率统计。

输出文件：

```text
eval_results/no_keyword_dense_rerank_eval.csv
eval_results/bm25_hybrid_rerank_eval.csv
eval_results/bm25_comparison_summary.json
eval_results/bm25_comparison_report.md
eval_results/agent_rag_eval.csv
eval_results/agent_rag_summary.json
eval_results/agent_memory_eval.csv
eval_results/agent_memory_summary.json
eval_results/science_small_context_eval.csv
eval_results/science_small_to_big_eval.csv
eval_results/science_small_to_big_summary.json
eval_results/science_small_to_big_report.md
eval_results/engineering_experiment_summary.json
eval_results/engineering_experiment_report.md
eval_results/engineering_experiment_details.csv
eval_results/build_kb_summary.json
eval_results/science_failure_cases.md
```

生成失败样例分析：

```bash
python log_analyzer.py
```

输出文件：

```text
eval_results/bm25_failure_cases.md
```

---

## 十一、页面展示

Streamlit 页面支持展示 Agent 问答、会话记忆、工具调用轨迹、问答结果、引用来源、检索片段、检索分数、评测结果和检索日志。

示例截图：
### RAG 问答与检索片段展示

![RAG Answer](assets/rag_answer.png)

### 自动评测结果展示

![Evaluation Dashboard](assets/eval_dashboard.png)

---

## 十二、后续优化方向

1. 调整 BM25 与向量召回的 RRF 权重，观察召回稳定性变化；
2. 使用 Cross-Encoder Rerank 模型替代规则打分，提高候选片段排序效果；
3. 继续扩充 LLM Query Rewrite 评测集，对比规则改写和 LLM 改写在模糊问题上的检索效果；
4. 扩充教育资料知识库，引入更多课程资料、论文笔记和教学案例；
5. 接入 Ragas 等更完整的 RAG 评测框架，补充上下文相关性、忠实度等指标；
6. 后续扩展 Multi-Agent 或 MCP Server，增强复杂任务处理能力。

---

## 十三、项目亮点

1. 实现了教育资料 RAG Agent 问答完整链路；
2. 新增本地状态机 Agent，支持问题分类、策略路由、Query Rewrite、上下文判断和工具轨迹记录；
3. 新增轻量会话记忆，支持多轮追问中的指代消解和主题延续；
4. 支持 vector、dense_rerank 和 bm25_hybrid 三种检索模式；
5. 设计了 Hybrid-First BM25 Hybrid + Rerank 策略；
6. 新增 Long-text RAG / Small-to-Big，上下文从小 chunk 扩展到父级大段落；
7. 支持答案来源引用、检索日志和 Agent Trace 日志；
8. 构建 60 条单轮测试问题集、30 条科学教材测试集和多轮记忆评测集；
9. 输出实验报告、科学失败样例分析和边界说明；
10. 使用 Streamlit 搭建可视化演示页面；
11. 项目具备可展示、可评测、可复盘和可写入简历的完整工程闭环。
