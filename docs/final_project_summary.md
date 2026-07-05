# 项目复盘：教育领域大模型可信问答系统

## 背景与目标

本项目面向初中科学教材问答场景，目标是让模型回答尽量基于教材证据，并在资料不足时稳定拒答。项目不是单独的 RAG demo，也不是单独的微调实验，而是围绕“教育可信问答闭环”把数据构造、模型适配、检索增强、Workflow 控制和自动评测串起来。

核心问题包括：

- 通用模型回答容易脱离教材依据。
- 学生问题常带有口语化表达、追问和指代。
- 检索资料不足时，模型容易复述低相关内容或强行生成。
- 单次人工观察难以判断系统是否真的变好，需要固定评测和失败分析。

## 技术路线

系统主线为：

```text
数据构造 -> QLoRA 微调 -> 检索增强 -> Workflow 控制 -> 固定评测 -> 失败分析 -> 数据补丁迭代
```

RAG 负责提供可追溯教材证据，QLoRA 负责适配教育问答、引用格式和拒答模式，Workflow 负责将检索、判断、生成和拒答拆成可观测步骤。评测结果再反向驱动数据补丁和控制逻辑迭代。

## 数据与微调

SFT 数据覆盖：

- `grounded_qa`：基于教材资料回答。
- `citation_qa`：答案末尾提供参考来源。
- `compare_or_reasoning`：概念对比和原理解释。
- `query_rewrite`：将口语化问题改写为可检索问题。
- `refusal`：资料不足、超出知识库范围或无法由资料支持时拒答。
- `anti_repetition`：降低重复段落输出。

当前 v2.1 数据集共 3000 条，train/dev/test 为 2400/300/300。v2.1 在 v2 基础上加入 600 条补丁样本，其中包含 hard-refusal 和 anti-repetition 数据。

微调采用 Qwen2.5-3B-Instruct + LLaMA Factory + QLoRA。训练产物不直接提交到仓库，仓库保留训练配置、轻量日志、评测结果和运行摘要。

## 检索增强与 Workflow

检索侧包含：

- ChromaDB 向量索引。
- `vector`、`dense_rerank`、`bm25_hybrid`、`contextual_hybrid` 四种模式。
- BM25+Dense 候选召回。
- RRF 融合与 Rerank 精排。
- Small-to-Big 长文本上下文扩展。
- 来源引用和检索日志。

Workflow 侧包含：

- 会话记忆和多轮追问补全。
- 问题分类与检索策略选择。
- Query Rewrite。
- v2 上下文证据判断，输出 `support_level`、`evidence_score`、命中/缺失词和轻量 claim verification。
- 回答生成或无资料拒答。
- Agent trace、Graph trace 和 Skills trace。

`context_guard.py` 是当前可信控制中的关键补充：当检索上下文不能直接支持问题时，系统不再把低相关片段交给模型硬答，而是返回无来源拒答。v2 Guard 在原有 `context_sufficient` 之外继续返回 `support_level`、`evidence_score`、`guard_details` 和 `claim_verification`，便于后续阈值调优和回归分析。

## 评测结果

SFT 生成质量核心结果：

| run | 固定题平均关键词 | 引用完整率 | 结构完整率 | 重复段落率 | 拒答正确率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| v2 QLoRA fixed50 | 0.7600 | 1.00 | 1.00 | n/a | 0.0000 |
| v2.1 QLoRA fixed50 | 0.8317 | 1.00 | 1.00 | 0.00 | 0.0909 |
| v2.1 QLoRA hard-refusal100 | 0.6225 | 1.00 | 1.00 | 0.00 | 0.9300 |

端到端 Workflow guard smoke test 覆盖 `/rag_chat`、LangGraph Workflow 和本地 Workflow 三条路径，共 24 条样例，当前通过率为 100%，错误数为 0。扩展回归集覆盖 76 个唯一样例、228 条路径结果，mock v2 当前整体通过率为 100%，guard 拒答通过率为 100%，支持题来源命中率为 100%，错误数为 0。扩展评测保留 `primary_expected_source` 用于诊断，主来源命中率为 96.55%，其余 6 条为合理来源别名命中。

真实模型端到端链路已用 Vast RTX 5060 Ti 验证：v2.1 QLoRA adapter 通过 `local_sft` 服务完成 smoke 24/24 和 extended 228/228 回归，overall、guard 拒答和支持题来源命中率均为 100%，错误数为 0。最终 extended 报告目录为 `eval_results/rag_workflow_guard/local_sft_extended_v21_vast_64tok_no_rerank/`。

汇总版实验表格见 `docs/experiment_results.md`。

结果说明：

- v2.1 提升了固定题关键词覆盖，并保持引用完整率和结构完整率稳定。
- hard-refusal 专项集证明数据补丁能明显改善资料不足拒答。
- 固定题中的开放型无依据问题仍需要 Workflow 和 `context_guard` 共同控制。
- 当前结果应理解为“降低无依据回答风险”，不是对所有问题的绝对保证。

## 工程交付形态

项目提供以下可运行形态：

- FastAPI 服务：基础问答、RAG 问答、Workflow 问答、健康检查和会话记忆接口。
- Streamlit 页面：Agent 问答、RAG 问答、评测展示和日志查看。
- Docker Compose：一键启动 API、页面和 Redis。
- 评测脚本：SFT 生成质量、hard-refusal、RAG/Workflow guard、科学教材 long-text 对比。
- 训练配置：LLaMA Factory LoRA/QLoRA 训练和 API 服务配置。

## 局限与后续方向

- 固定题中的开放型无依据问题仍存在强答风险，需要继续扩充更贴近真实学生表达的拒答样本。
- 评测集规模有限，后续可以增加跨章节、跨年级、实验现象和多轮追问样例。
- 当前主要使用本地 ChromaDB 和轻量 Redis 控制，生产部署还需要更完整的监控、鉴权和数据治理。
- Workflow guard 已完成 mock 与 `local_sft` 真模型 smoke/extended 回归闭环；后续重点是扩充评测规模、替换本地或可控 rerank 服务，并补充线上监控。
