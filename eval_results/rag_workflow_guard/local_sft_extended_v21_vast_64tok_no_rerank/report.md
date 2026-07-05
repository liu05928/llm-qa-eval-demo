# RAG/Workflow Guard Regression Report

- Generated at: 2026-07-05 12:27:17
- Backend: `local_sft`
- Case set: `extended`
- Guard mode: `v2`
- Unique cases: 76
- Overall pass rate: 100.00%
- Error count: 0

## Local SFT Endpoint

- Available: `True`
- Models URL: `http://127.0.0.1:8001/v1/models`
- Model IDs: `qwen25-3b-edu-qlora-v21`

## Path Summary

| path | total | pass_rate | guard_refusal_pass_rate | support_pass_rate | support_source_match_rate | primary_source_match_rate | errors | avg_latency_ms | avg_evidence_score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rag_chat_api | 76 | 100.00% | 100.00% | 100.00% | 100.00% | 96.55% | 0 | 4946.68 | 0.7263 |
| workflow_langgraph | 76 | 100.00% | 100.00% | 100.00% | 100.00% | 98.28% | 0 | 5493.21 | 0.7264 |
| workflow_local | 76 | 100.00% | 100.00% | 100.00% | 100.00% | 98.28% | 0 | 5377.69 | 0.7264 |

## Case Type Summary

| case_type | total_rows | pass_rate | source_match_rate |
| --- | ---: | ---: | ---: |
| guard | 54 | 100.00% | 0.00% |
| support | 174 | 100.00% | 100.00% |

## Source Match Categories

| category | rows |
| --- | ---: |
| likely_eval_alias_needed | 4 |
| primary_expected_source | 170 |

## Alias-Matched Source Rows

- `rag_chat_api` / `project_006`: primary_expected_source=rag_intro.md, matched_sources=vector_database.md|retrieval_optimization.md|embedding_intro.md, allowed_sources=rag_intro.md|vector_database.md|embedding_intro.md|retrieval_optimization.md
- `rag_chat_api` / `project_010`: primary_expected_source=rag_intro.md, matched_sources=retrieval_optimization.md|vector_database.md|embedding_intro.md, allowed_sources=rag_intro.md|retrieval_optimization.md|hybrid_search.md
- `workflow_local` / `project_010`: primary_expected_source=rag_intro.md, matched_sources=retrieval_optimization.md|vector_database.md|embedding_intro.md, allowed_sources=rag_intro.md|retrieval_optimization.md|hybrid_search.md
- `workflow_langgraph` / `project_010`: primary_expected_source=rag_intro.md, matched_sources=retrieval_optimization.md|vector_database.md|embedding_intro.md, allowed_sources=rag_intro.md|retrieval_optimization.md|hybrid_search.md
