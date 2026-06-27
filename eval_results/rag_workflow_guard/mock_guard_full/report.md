# RAG/Workflow Guard Regression Report

- Generated at: 2026-06-27 22:27:25
- Backend: `mock`
- Overall pass rate: 100.00%
- Error count: 0

## Local SFT Endpoint

- Available: `False`
- Models URL: `http://127.0.0.1:8001/v1/models`
- Error: `<urlopen error [Errno 1] Operation not permitted>`

## Path Summary

| path | total | pass_rate | guard_refusal_pass_rate | support_pass_rate | errors | avg_latency_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rag_chat_api | 8 | 100.00% | 100.00% | 100.00% | 0 | 268.41 |
| workflow_langgraph | 8 | 100.00% | 100.00% | 100.00% | 0 | 231.92 |
| workflow_local | 8 | 100.00% | 100.00% | 100.00% | 0 | 232.22 |
