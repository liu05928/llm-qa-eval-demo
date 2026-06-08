# DEV_SPEC

## 一、项目目标

本项目是一个面向教育资料的大模型 RAG 知识库问答与评测优化系统，目标是在基础 RAG 问答链路上，引入 Hybrid Search、Rerank、自动评测、日志分析和 Streamlit 可视化展示能力，提升系统在知识库问答场景下的检索稳定性、回答可信度和结果可分析性。

系统支持本地教育资料读取、文本切分、Embedding 向量化、向量库构建、关键词检索、向量检索、Hybrid Search 融合检索、Rerank 重排序、Prompt 构造、大模型回答生成、来源引用、日志记录和自动评测。

## 二、核心流程

文档读取 → 文本切分 → Embedding 向量化 → 向量库构建 → 向量检索 / 关键词检索 → Hybrid Search → Rerank → Prompt 构造 → 大模型生成 → 来源引用 → 日志记录 → 自动评测 → Streamlit 展示

## 三、项目功能设计

### 1. 基础 RAG 问答

系统支持读取本地教育资料文档，将文档切分为多个知识片段，并通过 Embedding 模型将文本片段转换为向量后写入本地向量数据库。用户输入问题后，系统可以进行相似度检索，召回相关知识片段，并结合大模型生成回答。

### 2. 来源引用返回

系统在生成回答时返回对应的来源文档和检索片段，便于用户追溯回答依据，提高问答结果的可信度和可解释性。

### 3. Hybrid Search 检索优化

在原有向量检索基础上，系统新增关键词检索能力，并将向量检索结果与关键词检索结果进行融合。Hybrid Search 可以同时利用语义相似度和关键词匹配结果，改善单一向量检索可能出现的漏召回问题。

### 4. Rerank 重排序

系统支持对 Hybrid Search 召回的候选知识片段进行轻量级重排序。流程上先扩大候选召回范围，再根据问题与 chunk 的相关性对候选片段进行排序，最终选择更相关的片段作为 Prompt 上下文。

### 5. 自动评测

系统构建了 50 条 RAG 测试问题，覆盖概念解释、原理机制、对比辨析、教育应用和资料缺失等类型，并从来源命中率、关键词命中率、引用完整率和无资料拒答率等指标评估系统效果。

### 6. 日志记录与失败样例分析

系统记录 RAG 问答日志和检索日志，用于分析检索结果、来源引用、回答长度和失败样例。评测结果可以进一步用于定位检索失败、切分问题、Prompt 约束不足、知识库缺失和问题表达模糊等问题。

### 7. Streamlit 可视化展示

系统提供 Streamlit 演示页面，支持用户输入问题、选择检索模式、设置 top_k 参数，并展示模型回答、引用来源、检索片段、检索分数和评测结果。

## 四、主要模块说明

### document_loader.py

负责读取本地教育资料文档，为后续文本切分提供原始文本。

### text_splitter.py

负责将原始文档切分为多个 chunk，并保存 chunk_id、source、content 等元信息。

### embedding_client.py

负责将文本 chunk 转换为 embedding 向量。

### vector_store.py

负责构建本地向量数据库，并提供基础向量检索能力。

### hybrid_retriever.py

负责关键词检索 sparse_search、向量检索 dense_search，以及 Hybrid Search 融合检索。

### reranker.py

负责对 Hybrid Search 召回的候选 chunk 进行轻量级重排序。

### rag_pipeline.py

负责 RAG 主流程，包括检索、Prompt 构造、大模型调用、来源返回和检索日志记录。

### rag_evaluator.py

负责自动评测 RAG 系统效果，评测指标包括来源命中率、关键词命中率、引用完整率和无资料拒答率。

### experiment_runner.py

负责自动运行 baseline 和 hybrid_rerank 两组实验，并保存 CSV 评测结果。

### log_analyzer.py

负责分析评测结果和日志，输出失败样例分析。

### web_demo.py

负责 Streamlit 页面展示，包括问答输入、检索模式选择、答案展示、引用来源展示、检索片段展示和评测结果展示。

## 五、评测设计

测试集包含 50 条问题，覆盖以下五类：

1. 概念解释类；
2. 原理机制类；
3. 对比辨析类；
4. 教育应用类；
5. 资料缺失类。

主要评测指标包括：

1. source_hit：检索来源是否命中预期文档；
2. keyword_hit：回答是否包含预期关键词；
3. has_citation：回答是否返回来源引用；
4. no_context_reject：资料缺失问题是否拒答。

## 六、实验设计

### Baseline：基础向量检索

```text
retriever_mode = vector
top_k = 3
```

该版本作为基础对照组，用于评估原始向量检索 RAG 流程的效果。

### Optimized：Hybrid Search + Rerank

```text
retriever_mode = hybrid
candidate_k = 10
final_top_k = 3
use_rerank = true
```

该版本先通过 Hybrid Search 扩大候选召回范围，再通过 Rerank 选择最终上下文，用于评估检索优化策略对系统效果的影响。

## 七、实验输出

实验结果保存到：

```text
eval_results/baseline_eval.csv
eval_results/hybrid_rerank_eval.csv
eval_results/experiment_report.md
eval_results/failure_cases.md
```

其中：

1. baseline_eval.csv 保存基础向量检索版本的评测结果；
2. hybrid_rerank_eval.csv 保存 Hybrid Search + Rerank 版本的评测结果；
3. experiment_report.md 保存实验目的、测试集设计、评测指标、实验结果和后续优化方向；
4. failure_cases.md 保存失败样例和原因分析。

## 八、后续优化方向

1. 使用 BM25 替代简单关键词匹配，提高关键词检索效果；
2. 使用 Cross-Encoder 模型进行更精细的 Rerank；
3. 增加 Query Rewrite，提高问题表达不清时的检索效果；
4. 扩充教育资料知识库，提升知识覆盖范围；
5. 接入 Ragas 等更完整的 RAG 评测框架；
6. 后续扩展 Agent 或 MCP Server，增强复杂任务处理能力。
