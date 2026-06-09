# 教育资料 RAG 知识库问答与评测优化系统

## 一、项目背景

本项目是一个面向教育资料的大模型 RAG 知识库问答与评测优化系统，基于 Python、FastAPI、ChromaDB 和 Streamlit 构建。

系统支持本地教育资料读取、文本切分、Embedding 向量化、向量检索、Dense Rerank、BM25 Hybrid、Rerank 重排序、大模型回答生成、来源引用、日志记录、自动评测和可视化展示。

项目目标不是单纯实现一个问答 Demo，而是构建一个具备检索优化、结果溯源、效果评测和失败样例分析能力的 RAG 工程项目。

---

## 二、技术栈

* Python
* FastAPI
* Streamlit
* ChromaDB
* sentence-transformers
* SiliconFlow / DeepSeek API
* Prompt Engineering
* JSON / JSONL / CSV
* BM25 Hybrid Search
* Rerank
* RAG Evaluation

---

## 三、系统功能

### 1. 基础 RAG 问答

系统支持读取本地教育资料文档，将文档切分为多个 chunk，并使用 Embedding 模型将文本片段转换为向量后写入 ChromaDB。

用户输入问题后，系统会检索相关知识片段，构造 RAG Prompt，并调用大模型生成回答。

### 2. 来源引用返回

系统会返回回答所依据的来源文档和 chunk_id，便于追溯答案依据，提高问答结果的可信度。

### 3. BM25 Hybrid 检索优化

系统在基础向量检索基础上新增 BM25 稀疏召回能力，将向量召回结果和 BM25 召回结果通过 RRF 进行融合，形成 BM25 Hybrid 候选召回结果。

### 4. Dense-Preserving BM25 Hybrid + Rerank

系统提供 `dense_rerank` 和 `bm25_hybrid` 两种优化实验模式，用于对比无稀疏召回和 BM25 稀疏召回的效果。

因此系统采用 Dense-Preserving BM25 Hybrid + Rerank 策略：

```text
基础向量检索兜底
↓
BM25 Hybrid 扩大候选召回
↓
轻量级 Rerank 重排序
↓
保留 dense top2
↓
使用 rerank 后的候选结果补充最终上下文
```

该策略既保留了向量检索的语义稳定性，也利用 BM25 和 Rerank 提升候选片段的覆盖度和排序效果。

### 5. 自动评测

系统构建了 50 条 RAG 测试问题，覆盖概念解释、原理机制、对比辨析、教育应用和资料缺失等类型。

评测指标包括：

* 来源命中率
* 关键词命中率
* 引用完整率
* 无资料拒答率
* 平均回答长度
* 平均检索片段数

### 6. 日志记录与失败样例分析

系统会记录 RAG 问答日志和检索日志，包括检索模式、候选片段、最终上下文、来源引用和回答长度等信息。

同时，项目提供失败样例分析脚本，用于定位检索失败、回答覆盖不足、Prompt 约束不足、知识库缺失等问题。

### 7. Streamlit 可视化展示

项目提供 Streamlit 页面，支持：

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
├── llm_client.py
├── config.py
├── prompt_templates.py
├── document_loader.py
├── text_splitter.py
├── embedding_client.py
├── vector_store.py
├── hybrid_retriever.py
├── reranker.py
├── rag_pipeline.py
├── rag_logger.py
├── rag_evaluator.py
├── experiment_runner.py
├── log_analyzer.py
├── DEV_SPEC.md
├── data/
│   ├── raw_docs/
│   ├── chunks/
│   └── rag_test_questions.json
├── vector_db/
├── logs/
│   ├── rag_log.json
│   └── retrieval_log.json
├── eval_results/
│   ├── baseline_eval.csv
│   ├── no_keyword_dense_rerank_eval.csv
│   ├── bm25_hybrid_rerank_eval.csv
│   ├── bm25_comparison_summary.json
│   ├── bm25_comparison_report.md
│   └── bm25_failure_cases.md
├── assets/
│   └── rag_demo.png
├── requirements.txt
├── .env.example
└── README.md
```

---

## 五、核心流程

```text
文档读取
↓
文本切分
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
```

测试集共 50 条问题，覆盖以下类型：

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
```

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

### 3. 构建 chunks

```bash
python text_splitter.py
```

### 4. 构建向量库

```bash
python vector_store.py
```

### 5. 启动 FastAPI

```bash
uvicorn app:app --reload
```

访问接口文档：

```text
http://127.0.0.1:8000/docs
```

### 6. 调用 RAG 接口

```json
{
  "question": "什么是 RAG？",
  "top_k": 3,
  "retriever_mode": "bm25_hybrid",
  "candidate_k": 10,
  "use_rerank": true
}
```

### 7. 启动 Streamlit 页面

```bash
streamlit run web_demo.py
```

---

## 十、自动评测

运行对比实验：

```bash
python experiment_runner.py
```

输出文件：

```text
eval_results/no_keyword_dense_rerank_eval.csv
eval_results/bm25_hybrid_rerank_eval.csv
eval_results/bm25_comparison_summary.json
eval_results/bm25_comparison_report.md
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

Streamlit 页面支持展示问答结果、引用来源、检索片段、检索分数、评测结果和检索日志。

示例截图：
### RAG 问答与检索片段展示

![RAG Answer](assets/rag_answer.png)

### 自动评测结果展示

![Evaluation Dashboard](assets/eval_dashboard.png)

---

## 十二、后续优化方向

1. 调整 BM25 与向量召回的 RRF 权重，观察召回稳定性变化；
2. 使用 Cross-Encoder Rerank 模型替代规则打分，提高候选片段排序效果；
3. 增加 Query Rewrite，提高复杂问题和模糊问题的检索效果；
4. 扩充教育资料知识库，引入更多课程资料、论文笔记和教学案例；
5. 接入 Ragas 等更完整的 RAG 评测框架，补充上下文相关性、忠实度等指标；
6. 后续扩展 Agent 或 MCP Server，增强复杂任务处理能力。

---

## 十三、项目亮点

1. 实现了教育资料 RAG 问答完整链路；
2. 支持 vector、dense_rerank 和 bm25_hybrid 三种检索模式；
3. 设计了 Dense-Preserving BM25 Hybrid + Rerank 策略；
4. 支持答案来源引用和检索日志记录；
5. 构建 50 条测试问题集并完成自动评测；
6. 输出实验报告和失败样例分析；
7. 使用 Streamlit 搭建可视化演示页面；
8. 项目具备可展示、可评测、可复盘和可写入简历的完整工程闭环。
