# 实验结果总览

本文档汇总当前项目已经完成的核心实验结果，便于答辩、复盘和后续复现。完整运行命令见 `docs/reproduce.md`，原始报告保留在 `eval_results/`。

## 1. SFT 生成质量

| run | 固定题平均关键词 | 引用完整率 | 结构完整率 | 重复段落率 | 拒答正确率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| v2 QLoRA fixed50 | 0.7600 | 1.00 | 1.00 | n/a | 0.0000 |
| v2.1 QLoRA fixed50 | 0.8317 | 1.00 | 1.00 | 0.00 | 0.0909 |
| v2.1 QLoRA hard-refusal100 | 0.6225 | 1.00 | 1.00 | 0.00 | 0.9300 |

结论：v2.1 在保持引用和结构稳定的同时，提高了固定题关键词覆盖，并显著改善资料不足类问题的拒答能力。固定题中的泛化型无依据问题仍需要 RAG/Workflow 层的 `context_guard` 继续约束。

报告位置：

- `eval_results/sft_generation_v21/comparison_report.md`

## 2. Mock RAG/Workflow Guard 回归

扩展回归集覆盖 76 个唯一样例，分别经过 `/rag_chat`、LangGraph Workflow 和本地 Workflow 三条路径，共 228 条路径结果。

| 指标 | 结果 |
| --- | ---: |
| overall pass rate | 100.00% |
| guard refusal pass rate | 100.00% |
| support source match rate | 100.00% |
| error count | 0 |
| primary source match rate | 96.55% |
| reasonable source alias rows | 6 |

三条路径结果：

| path | total | pass rate | guard refusal | support source match | primary source match | errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rag_chat_api` | 76 | 100.00% | 100.00% | 100.00% | 96.55% | 0 |
| `workflow_langgraph` | 76 | 100.00% | 100.00% | 100.00% | 96.55% | 0 |
| `workflow_local` | 76 | 100.00% | 100.00% | 100.00% | 96.55% | 0 |

报告位置：

- `eval_results/rag_workflow_guard/smoke_mock_guard_v2_source_mismatch_final/report.md`
- `eval_results/rag_workflow_guard/extended_mock_guard_v2_source_mismatch_final/report.md`
- `eval_results/rag_workflow_guard/smoke_mock_guard_v2_final_local/report.md`
- `eval_results/rag_workflow_guard/extended_mock_guard_v2_final_local/report.md`

## 3. 真实 local_sft 回归

真实模型实验在 Vast RTX 5060 Ti 上完成，使用 `qwen25-3b-edu-qlora-v21` adapter 和 `local_sft_server.py` 提供的 OpenAI-compatible 接口。

最终 extended 回归使用 `--no-rerank`，用于隔离外部重排 API 的网络不稳定性，避免 SiliconFlow rerank 超时污染真实模型链路判断。

| 指标 | 结果 |
| --- | ---: |
| smoke rows | 24/24 |
| extended rows | 228/228 |
| unique cases | 76 |
| overall pass rate | 100.00% |
| guard refusal pass rate | 100.00% |
| support source match rate | 100.00% |
| error count | 0 |
| reasonable source alias rows | 4 |

三条路径结果：

| path | total | pass rate | guard refusal | support source match | primary source match | avg latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rag_chat_api` | 76 | 100.00% | 100.00% | 100.00% | 96.55% | 4946.68 ms |
| `workflow_langgraph` | 76 | 100.00% | 100.00% | 100.00% | 98.28% | 5493.21 ms |
| `workflow_local` | 76 | 100.00% | 100.00% | 100.00% | 98.28% | 5377.69 ms |

报告位置：

- `eval_results/rag_workflow_guard/local_sft_smoke_v21_vast/report.md`
- `eval_results/rag_workflow_guard/local_sft_extended_v21_vast_64tok_no_rerank/report.md`

## 4. Source Match 口径

扩展评测同时保留两个来源判断口径：

- `primary_expected_source`：原始主期望来源，用于诊断检索是否偏离最初设计。
- `expected_sources`：允许多个合理来源命中，用于避免把等价资料误判为系统失败。

当支持题命中 `expected_sources` 但没有命中 `primary_expected_source` 时，报告标记为 `likely_eval_alias_needed`。这类样例不计为失败，但会保留在报告中，便于后续人工审查评测题口径。

## 5. 当前结论

当前项目已经完成从 mock 到真实微调模型的 RAG/Workflow guard 闭环。扩展回归达到：

- overall pass rate 100.00%
- guard refusal pass rate 100.00%
- support source match rate 100.00%
- error count 0

这些结果说明当前系统在已有知识库、评测集和控制策略范围内，可以稳定做到“有依据则回答并引用来源、无依据则拒答”。对外表述仍建议使用“降低无依据回答风险、提升拒答稳定性”，不要扩大为对任意开放问题的绝对保证。

云端实例销毁后，仍可使用 mock 后端在本地复核系统控制逻辑。最新本地基线为 `smoke_mock_guard_v2_final_local` 和 `extended_mock_guard_v2_final_local`，均已通过 `--fail-on-error`。
