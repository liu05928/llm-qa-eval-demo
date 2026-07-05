# RAG/Workflow Guard Regression Report

- Generated at: 2026-07-05 18:19:26
- Backend: `mock`
- Case set: `smoke`
- Guard mode: `v2`
- Unique cases: 8
- Overall pass rate: 100.00%
- Error count: 0

## Local SFT Endpoint

- Available: `False`
- Models URL: `http://127.0.0.1:8001/v1/models`
- Error: `<urlopen error [Errno 61] Connection refused>`

## Path Summary

| path | total | pass_rate | guard_refusal_pass_rate | support_pass_rate | support_source_match_rate | primary_source_match_rate | errors | avg_latency_ms | avg_evidence_score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rag_chat_api | 8 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0 | 451.85 | 0.2500 |
| workflow_langgraph | 8 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0 | 796.56 | 0.2500 |
| workflow_local | 8 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0 | 798.28 | 0.2500 |

## Case Type Summary

| case_type | total_rows | pass_rate | source_match_rate |
| --- | ---: | ---: | ---: |
| guard | 18 | 100.00% | 0.00% |
| support | 6 | 100.00% | 100.00% |

## Source Match Categories

| category | rows |
| --- | ---: |
| primary_expected_source | 6 |
