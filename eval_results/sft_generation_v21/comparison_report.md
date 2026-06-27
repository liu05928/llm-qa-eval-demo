# SFT Generation Evaluation v2.1 Comparison

Fixed question set: `data/sft_v2/quality_eval_questions.json` / `data/sft_v21/quality_eval_questions.json` (same 50 held-out questions).

Hard-refusal set: `data/sft_v21/hard_refusal_eval_questions.json` (100 held-out questions).

| run | questions | avg_keyword_score | keyword_all_hit_rate | citation_complete_rate | structure_complete_rate | repeated_section_rate | refusal_correct_rate | possible_fabrication_rate | avg_latency_s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base_qwen25_3b fixed50 | 50 | 0.5950 | 0.34 | 0.76 | 0.58 | n/a | 0.3636 | 0.14 | 5.6770 |
| v1_qlora fixed50 | 50 | 0.5667 | 0.24 | 0.98 | 0.72 | n/a | 0.0000 | 0.22 | 5.5560 |
| v2_qlora fixed50 | 50 | 0.7600 | 0.44 | 1.00 | 1.00 | n/a | 0.0000 | 0.22 | 7.5255 |
| v2.1_qlora fixed50 | 50 | 0.8317 | 0.46 | 1.00 | 1.00 | 0.00 | 0.0909 | 0.20 | 7.5792 |
| v2.1_qlora hard_refusal100 | 100 | 0.6225 | 0.04 | 1.00 | 1.00 | 0.00 | 0.9300 | 0.07 | 6.1354 |

## Main Findings

- v2.1 improved fixed-set keyword coverage over v2 while keeping citation and answer structure at 1.00.
- The anti-repetition patch plus decoding controls removed repeated answer sections in the two v2.1 evaluations.
- The hard-refusal patch worked on the dedicated 100-question hard-refusal set: refusal correctness reached 0.93.
- Fixed-set refusal remains weak: only 1 of 11 refusal cases was correctly refused. The remaining failures usually answered unsupported questions by repackaging unrelated retrieved context.

## Follow-up Implemented

An explicit context-sufficiency gate has been added in `context_guard.py`. `/rag_chat`, the local Workflow, and the LangGraph Workflow now share the same unsupported-question detection and no-context refusal behavior. The next validation step is to rerun the RAG/Workflow regression suite in a full dependency environment.
