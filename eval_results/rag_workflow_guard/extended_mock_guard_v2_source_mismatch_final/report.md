# RAG/Workflow Guard Regression Report

- Generated at: 2026-07-05 18:22:19
- Backend: `mock`
- Case set: `extended`
- Guard mode: `v2`
- Unique cases: 76
- Overall pass rate: 100.00%
- Error count: 0

## Local SFT Endpoint

- Available: `False`
- Models URL: `http://127.0.0.1:8001/v1/models`
- Error: `<urlopen error [Errno 61] Connection refused>`

## Path Summary

| path | total | pass_rate | guard_refusal_pass_rate | support_pass_rate | support_source_match_rate | primary_source_match_rate | errors | avg_latency_ms | avg_evidence_score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rag_chat_api | 76 | 100.00% | 100.00% | 100.00% | 100.00% | 96.55% | 0 | 475.08 | 0.7405 |
| workflow_langgraph | 76 | 100.00% | 100.00% | 100.00% | 100.00% | 96.55% | 0 | 867.13 | 0.7405 |
| workflow_local | 76 | 100.00% | 100.00% | 100.00% | 100.00% | 96.55% | 0 | 863.96 | 0.7405 |

## Case Type Summary

| case_type | total_rows | pass_rate | source_match_rate |
| --- | ---: | ---: | ---: |
| guard | 54 | 100.00% | 0.00% |
| support | 174 | 100.00% | 100.00% |

## Source Match Categories

| category | rows |
| --- | ---: |
| likely_eval_alias_needed | 6 |
| primary_expected_source | 168 |

## Alias-Matched Source Rows

- `rag_chat_api` / `project_009`: primary_expected_source=rag_intro.md, matched_sources=vector_knowledge_base.md|rerank_intro.md, allowed_sources=rag_intro.md|education_ai.md|retrieval_optimization.md|hybrid_search.md|rerank_intro.md|vector_knowledge_base.md
- `workflow_local` / `project_009`: primary_expected_source=rag_intro.md, matched_sources=vector_knowledge_base.md|rerank_intro.md|education_ai.md, allowed_sources=rag_intro.md|education_ai.md|retrieval_optimization.md|hybrid_search.md|rerank_intro.md|vector_knowledge_base.md
- `workflow_langgraph` / `project_009`: primary_expected_source=rag_intro.md, matched_sources=vector_knowledge_base.md|rerank_intro.md|education_ai.md, allowed_sources=rag_intro.md|education_ai.md|retrieval_optimization.md|hybrid_search.md|rerank_intro.md|vector_knowledge_base.md
- `rag_chat_api` / `project_010`: primary_expected_source=rag_intro.md, matched_sources=retrieval_optimization.md|hybrid_search.md, allowed_sources=rag_intro.md|retrieval_optimization.md|hybrid_search.md
- `workflow_local` / `project_010`: primary_expected_source=rag_intro.md, matched_sources=retrieval_optimization.md|hybrid_search.md, allowed_sources=rag_intro.md|retrieval_optimization.md|hybrid_search.md
- `workflow_langgraph` / `project_010`: primary_expected_source=rag_intro.md, matched_sources=retrieval_optimization.md|hybrid_search.md, allowed_sources=rag_intro.md|retrieval_optimization.md|hybrid_search.md
