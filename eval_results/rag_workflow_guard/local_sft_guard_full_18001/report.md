# RAG/Workflow Guard Regression Report

- Generated at: 2026-06-27 22:53:29
- Backend: `local_sft`
- Overall pass rate: 100.00%
- Error count: 0

## Local SFT Endpoint

- Available: `True`
- Models URL: `http://127.0.0.1:18001/v1/models`
- Model IDs: `gpt-3.5-turbo`

## Path Summary

| path | total | pass_rate | guard_refusal_pass_rate | support_pass_rate | errors | avg_latency_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rag_chat_api | 8 | 100.00% | 100.00% | 100.00% | 0 | 3222.47 |
| workflow_langgraph | 8 | 100.00% | 100.00% | 100.00% | 0 | 3010.14 |
| workflow_local | 8 | 100.00% | 100.00% | 100.00% | 0 | 2793.20 |
