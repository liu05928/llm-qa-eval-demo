# DEV_SPEC

## 一、项目目标

本项目是一个面向教育资料与初中科学教材的大模型 RAG Agent 问答与评测优化系统，目标是在基础 RAG 问答链路上，引入 Single-Agent 工作流、垂直领域知识库构建、Dense Rerank、BM25 Hybrid、自动评测、日志分析和 Streamlit 可视化展示能力，提升系统在知识库问答场景下的检索稳定性、回答可信度和结果可分析性。

系统支持本地状态机 Agent、会话记忆、问题类型识别、检索策略路由、Query Rewrite、初中科学教材导入、Long-text RAG、Small-to-Big 上下文扩展、本地教育资料读取、递归文档加载、文本切分、Embedding 向量化、向量库构建、向量检索、BM25 稀疏召回、RRF 融合检索、Rerank 重排序、Prompt 构造、大模型回答生成、来源引用、日志记录和自动评测。

## 二、核心流程

Agent 主流程：

用户问题 → 会话记忆补全 → 问题类型识别 → 检索策略路由 → Query Rewrite（可选）→ 向量检索 / BM25 Hybrid → Rerank → 上下文充分性判断 → 大模型生成或资料不足拒答 → 来源引用 → Agent Trace 日志 → Streamlit 展示

RAG 检索流程：

文档读取 → 生成 small chunks / big chunks → Embedding 向量化 small chunks → 向量库构建 → 向量检索 / BM25 召回 → RRF 融合 → Rerank → Small-to-Big 父段落扩展 → Prompt 构造 → 大模型生成 → 来源引用 → 日志记录 → 自动评测 → Streamlit 展示

垂直知识库构建流程：

ch-3 初中科学教材 Markdown → 均衡抽样 → 元数据提取 → 写入 data/raw_docs/science_textbooks → 递归加载 → 文本切分 → 向量库构建 → RAG / Agent 问答

## 三、项目功能设计

### 1. Single-Agent RAG 问答

系统新增本地状态机 Agent，不依赖 LangGraph 或 LangChain Agent。Agent 根据用户问题和会话记忆动态完成以下步骤：

1. resolve_memory：读取当前学习主题和最近对话，补全“它、这个技术、前者、后者”等多轮追问；
2. classify_query：识别概念解释、对比分析、学习建议、资料缺失候选、模糊问题和普通问题；
3. select_strategy：根据问题类型选择 dense_rerank 或 bm25_hybrid；
4. retrieve_context：调用现有 RAG 检索能力召回上下文；
5. judge_context：根据关键词覆盖度和来源数量判断上下文是否足够；
6. rewrite_query：当问题模糊或上下文不足时，将问题改写为更适合检索的查询；
7. generate_answer：上下文充足时复用 RAG Prompt 和大模型调用能力生成回答；
8. update_memory：记录本轮主题、回答摘要、引用来源和检索策略；
9. finalize_response：统一返回 answer、sources、retrieved_chunks、agent_trace、memory_snapshot 等字段；
10. log_trace：记录 Agent 执行过程，便于页面展示和失败复盘。

Query Rewrite 采用 LLM 优先、规则兜底的设计。真实 API 模式下调用专用 `query_rewrite` Prompt 生成检索查询；Mock 模式、API 失败、输出为空或输出与原问题重复时回退到规则改写。Agent Trace 会记录 `rewrite_strategy`、`fallback_used` 和改写后的查询，便于解释每次改写是否真正由 LLM 完成。

Agent 只展示可观测执行轨迹，例如节点名、工具名、输入摘要、输出摘要、状态和耗时，不暴露长篇推理内容。

### 2. 会话记忆

系统新增轻量会话记忆模块，用于支持多轮学习助手场景。记忆内容包括当前主题、最近问题、记忆补全问题、上一轮回答摘要、引用来源、检索策略和历史 Query Rewrite。

当前记忆是会话级短期记忆，优先解决多轮追问中的指代消解和主题延续，不引入长期向量记忆库，避免增加不必要的复杂度。

### 3. 基础 RAG 问答

系统支持读取本地教育资料文档，将文档切分为多个知识片段，并通过 Embedding 模型将文本片段转换为向量后写入本地向量数据库。用户输入问题后，系统可以进行相似度检索，召回相关知识片段，并结合大模型生成回答。

### 4. 垂直领域知识库

系统新增初中科学教材垂直知识库，语料来自 `../ch-3/knowledge_base_builder/data/textbooks/初中科学/沪教版初中科学`。

导入模块 `domain_kb_importer.py` 负责：

1. 扫描 ch-3 中的沪教版初中科学 Markdown 文档；
2. 解析 YAML 风格 front matter 中的 textbook_id、grade、semester、publisher、school_level、chapter、content_type 等字段；
3. 按 7-9 年级上下册进行均衡抽样，默认导入 80 篇；
4. 将元数据和教材正文合并写入 `data/raw_docs/science_textbooks/`；
5. 生成 `data/science_textbook_manifest.json`，记录候选文档数、导入数量、教材分布、章节分布和目标文件列表。

该模块将项目定位从“通用学习资料 RAG Demo”扩展为“初中科学教材垂直领域 RAG Agent 问答系统”。当前实现优先强调可复现导入、来源可追溯和演示稳定性，不一次性导入 ch-3 的全量 99573 篇 Markdown。

### 5. Long-text RAG：Small-to-Big

系统新增 Small-to-Big 长文本 RAG 能力，解决长文档问答中“召回片段太碎、回答缺少前后文”的问题。

设计方式：

1. `text_splitter.py` 同时生成两级 chunk：
   - `data/chunks/chunks.json` 保存 small chunks，用于召回；
   - `data/chunks/big_chunks.json` 保存 big chunks，用于回答上下文；
2. small chunk 默认 500 字左右，保留 `parent_chunk_id`、`parent_index`、`small_index`；
3. big chunk 默认 1600 字左右，按 Markdown 段落优先切分，保留 `child_chunk_ids`；
4. `long_text_context.py` 根据检索命中的 small chunks 扩展到父级 big chunks，并对同一父段落去重；
5. `rag_pipeline.py` 支持 `context_mode=small` 和 `context_mode=small_to_big` 两种上下文模式；
6. 返回 `small_retrieved_chunks`、`long_context`、`trigger_chunk_ids`，用于页面展示和日志复盘。

该设计保持“小 chunk 适合召回、大 chunk 适合回答”的职责分离，适合教材、政策、公文制度等长文本知识库。

### 6. 来源引用返回

系统在生成回答时返回对应的来源文档和检索片段，便于用户追溯回答依据，提高问答结果的可信度和可解释性。

### 7. BM25 Hybrid 检索优化

在原有向量检索基础上，系统新增 BM25 稀疏召回能力，并将向量召回结果与 BM25 召回结果进行融合。BM25 Hybrid 可以同时利用语义相似度和术语精确匹配信号，改善单一向量检索可能出现的漏召回问题。

### 8. Rerank 重排序

系统支持对向量候选或 BM25 Hybrid 候选知识片段进行轻量级重排序。流程上先扩大候选召回范围，再根据问题与 chunk 的相关性对候选片段进行排序，最终选择更相关的片段作为 Prompt 上下文。

### 9. 自动评测

系统构建了 60 条 RAG 测试问题，覆盖概念解释、原理机制、对比辨析、教育应用和资料缺失等类型，并从来源命中率、关键词命中率、引用完整率和无资料拒答率等指标评估系统效果。

系统新增 `data/science_rag_test_questions.json`，包含 30 条初中科学教材问题，用于验证教材知识库的来源命中、关键词覆盖、实验现象回答、生活应用回答、对比分析和资料缺失拒答能力。

### 10. 日志记录与失败样例分析

系统记录 RAG 问答日志、检索日志和 Agent Trace 日志，用于分析检索结果、来源引用、回答长度、工具调用路径、上下文充分性和失败样例。评测结果可以进一步用于定位检索失败、切分问题、Prompt 约束不足、知识库缺失和问题表达模糊等问题。科学教材评测额外输出 `eval_results/science_failure_cases.md`，区分硬性失败和关键词表达变体造成的软性观察。

### 11. Streamlit 可视化展示

系统提供 Streamlit 演示页面，支持 Agent 问答、会话记忆、原始 RAG 问答、评测展示和日志查看。Agent 页面展示问题类型、检索策略、Query Rewrite、上下文充分性、记忆补全问题、模型回答、引用来源、检索片段和工具调用轨迹。

## 四、主要模块说明

### document_loader.py

负责递归读取本地教育资料文档，为后续文本切分提供原始文本。递归读取支持将不同垂直领域资料放在 `data/raw_docs/` 下的独立子目录中。

### domain_kb_importer.py

负责从 ch-3 初中科学教材中导入垂直领域语料，完成抽样、元数据提取、正文规范化写入和导入清单生成。

### build_kb.py

负责一键构建知识库，串联语料导入、small/big chunks 生成、Chroma 向量索引构建、Small-to-Big 冒烟测试和可选科学教材对比评测。

### long_text_context.py

负责 Small-to-Big 上下文扩展，将检索命中的 small chunks 映射到父级 big chunks，并输出触发关系、扩展数量和上下文长度等可观测信息。

### text_splitter.py

负责将原始文档切分为 small chunks 和 big chunks，并保存 chunk_id、source、content、parent_chunk_id、child_chunk_ids 等元信息。

### embedding_client.py

负责将文本 chunk 转换为 embedding 向量。

### vector_store.py

负责构建本地向量数据库，并提供基础向量检索能力。

### hybrid_retriever.py

负责向量检索 dense_search、BM25 稀疏召回 bm25_search，以及 BM25 Hybrid 融合检索。

### reranker.py

负责对向量候选或 BM25 Hybrid 候选 chunk 进行轻量级重排序。

### agent_memory.py

负责轻量会话记忆，包括当前主题、最近对话、指代消解、主题延续和记忆快照序列化。

### agent_session_store.py

负责 Agent session 级记忆持久化，将 session_id 与 ConversationMemory 快照保存到 logs/agent_sessions.json，提供创建、读取、保存和重置能力。

### agent_state.py

负责定义 AgentState 和 AgentTraceStep，用于维护 Agent 运行状态、记忆状态和可展示的工具调用轨迹。

### rag_agent.py

负责 Single-Agent RAG 工作流，包括会话记忆、问题分类、策略选择、LLM Query Rewrite、规则兜底改写、上下文判断、回答生成、拒答控制和 Agent Trace 日志记录。

### rag_pipeline.py

负责 RAG 主流程，包括检索、Prompt 构造、大模型调用、来源返回和检索日志记录。

### rag_evaluator.py

负责自动评测 RAG 系统效果，评测指标包括来源命中率、关键词命中率、引用完整率和无资料拒答率。

### experiment_runner.py

负责自动运行 dense_rerank_no_keyword 和 bm25_hybrid_rerank 两组实验，并保存 CSV、JSON 汇总和 Markdown 对比报告。

### agent_experiment_runner.py

负责自动运行 Single-Agent RAG 版本评测，并保存 Agent 评测明细和指标汇总。

### memory_experiment_runner.py

负责自动运行多轮 Agent 记忆评测，并保存指代消解、主题延续、来源命中和资料缺失拒答等指标。

### science_long_text_eval_runner.py

负责自动运行科学教材 `small` 和 `small_to_big` 两组上下文实验，并输出 CSV、JSON 汇总和 Markdown 报告。评测同时保留严格关键词命中和同义关键词命中，避免“固态/固体形态”等表达变体造成假阴性。

### science_failure_analyzer.py

负责分析科学教材评测 CSV，输出 `eval_results/science_failure_cases.md`。报告会标记检索未命中、回答覆盖不足、引用缺失、资料缺失拒答不足，以及严格关键词未命中但同义关键词命中的软性观察。

### log_analyzer.py

负责分析评测结果和日志，输出失败样例分析。

### web_demo.py

负责 Streamlit 页面展示，包括 Agent 问答、原始 RAG 问答、检索模式选择、答案展示、引用来源展示、检索片段展示、工具调用轨迹和评测结果展示。

## 五、API 设计

### Agent Chat API

```text
POST /agent/chat
```

请求字段：

1. question：用户问题；
2. session_id：会话 ID，为空时自动生成；
3. top_k：最终上下文片段数；
4. candidate_k：候选召回片段数；
5. max_rewrites：最大 Query Rewrite 次数；
6. use_rerank：是否启用 Rerank；
7. reset_memory：是否在本轮问答前清空当前 session 记忆。

响应字段包括 session_id、question、resolved_question、answer、sources、retrieved_chunks、query_type、retriever_mode、memory_used、memory_snapshot 和 agent_trace。

### Agent Session API

```text
GET /agent/session/{session_id}
DELETE /agent/session/{session_id}
```

GET 用于查看当前 session 的会话记忆，DELETE 用于清空指定 session 的短期记忆。

## 六、评测设计

测试集包含 60 条大模型技术资料问题和 30 条初中科学教材问题，覆盖以下类型：

1. 概念解释类；
2. 原理机制类；
3. 对比辨析类；
4. 教育应用类；
5. 资料缺失类；
6. 实验现象类；
7. 生活应用类。

主要评测指标包括：

1. source_hit：检索来源是否命中预期文档；
2. keyword_hit：回答是否包含预期关键词；
3. has_citation：回答是否返回来源引用；
4. no_context_reject：资料缺失问题是否拒答。

科学教材评测额外保留：

1. exact_keyword_hit：严格字面关键词命中；
2. semantic_keyword_hit：支持少量领域同义表达的关键词命中；
3. no_context_reject：资料缺失题是否明确拒答，且不参与来源命中率和关键词命中率统计。

## 七、实验设计

### Agent：Single-Agent RAG

```text
run_rag_agent(
    question="RAG 和普通问答有什么区别？",
    top_k=3,
    candidate_k=10,
    max_rewrites=1,
    use_rerank=true
)
```

默认策略：

1. 概念解释、学习建议、对比分析、模糊问题和资料缺失候选优先使用 bm25_hybrid；
2. 普通问题默认使用 dense_rerank；
3. 模糊问题先触发 Query Rewrite；
4. 上下文不足且仍有 rewrite 次数时再次改写并二次检索；
5. 最终上下文仍不足时执行资料不足拒答，不引用低相关来源。

### Memory：多轮会话记忆评测

```text
memory = ConversationMemory()
run_rag_agent("什么是 RAG？", memory=memory)
run_rag_agent("它为什么可以减少幻觉？", memory=memory)
```

多轮记忆评测独立于 60 条单轮评测，主要验证：

1. memory_used_accuracy：是否在需要时使用记忆；
2. memory_rewrite_hit_rate：补全后的问题是否包含预期主题；
3. topic_follow_hit_rate：多轮追问是否延续正确主题；
4. source_hit_rate：追问后是否仍命中预期来源；
5. no_context_reject_rate：资料缺失追问是否拒答。

### Vector：基础向量检索

```text
retriever_mode = vector
top_k = 3
```

该版本用于基础 RAG 问答，不参与当前默认 BM25 对比实验。

### No Keyword：Dense Rerank

```text
retriever_mode = dense_rerank
candidate_k = 10
final_top_k = 3
use_rerank = true
```

该版本只使用向量召回 candidate_k 个候选片段，再通过 Rerank 选择最终上下文，用于作为无稀疏召回对照组。

### BM25 Hybrid + Rerank

```text
retriever_mode = bm25_hybrid
candidate_k = 10
final_top_k = 3
use_rerank = true
```

该版本使用向量召回 + BM25 稀疏召回 + RRF 融合扩大候选集合，再通过 Rerank 分与 BM25/Hybrid 检索分融合选择最终 small chunks。若启用 `context_mode=small_to_big`，最终 small chunks 会继续扩展为父级 big chunks 参与回答。垂直领域教材问答中，BM25 术语命中优先级更高，dense 结果作为兜底补充。

## 八、实验输出

实验结果保存到：

```text
eval_results/baseline_eval.csv
eval_results/no_keyword_dense_rerank_eval.csv
eval_results/bm25_hybrid_rerank_eval.csv
eval_results/bm25_comparison_summary.json
eval_results/bm25_comparison_report.md
eval_results/bm25_failure_cases.md
eval_results/agent_rag_eval.csv
eval_results/agent_rag_summary.json
eval_results/agent_memory_eval.csv
eval_results/agent_memory_summary.json
eval_results/science_small_context_eval.csv
eval_results/science_small_to_big_eval.csv
eval_results/science_small_to_big_summary.json
eval_results/science_small_to_big_report.md
eval_results/science_failure_cases.md
eval_results/build_kb_summary.json
```

其中：

1. baseline_eval.csv 保存历史基础向量检索版本的评测结果；
2. no_keyword_dense_rerank_eval.csv 保存 Dense Rerank 无稀疏召回版本的评测结果；
3. bm25_hybrid_rerank_eval.csv 保存 BM25 Hybrid + Rerank 版本的评测结果；
4. bm25_comparison_summary.json 保存两组实验的指标汇总和差值；
5. bm25_comparison_report.md 保存 BM25 对比实验报告；
6. bm25_failure_cases.md 保存失败样例和原因分析；
7. agent_rag_eval.csv 保存 Agent 版本逐题评测结果；
8. agent_rag_summary.json 保存 Agent 版本指标汇总；
9. agent_memory_eval.csv 保存多轮记忆评测逐轮结果；
10. agent_memory_summary.json 保存多轮记忆评测指标汇总；
11. science_small_context_eval.csv 保存科学教材 small 上下文对照组结果；
12. science_small_to_big_eval.csv 保存科学教材 Small-to-Big 实验组结果；
13. science_small_to_big_summary.json 保存 Small-to-Big 对比指标汇总；
14. science_small_to_big_report.md 保存 Small-to-Big 对比实验报告；
15. science_failure_cases.md 保存科学教材失败样例和边界分析；
16. build_kb_summary.json 保存一键构建摘要。

## 九、后续优化方向

1. 调整 BM25 与向量召回的 RRF 权重，观察召回稳定性变化；
2. 使用 Cross-Encoder 模型进行更精细的 Rerank；
3. 扩充 Query Rewrite 对照评测，对比规则改写和 LLM 改写在模糊问题上的检索命中率；
4. 扩充教育资料知识库，提升知识覆盖范围；
5. 接入 Ragas 等更完整的 RAG 评测框架；
6. 后续扩展 Multi-Agent 或 MCP Server，增强复杂任务处理能力。
