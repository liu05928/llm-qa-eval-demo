# Training Reproduction Notes

This document summarizes the public, reproducible parts of the QLoRA training setup. It intentionally avoids local machine paths, temporary cloud operations, SSH commands, and model weight artifacts.

## Base Model And Method

- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Fine-tuning method: LoRA / QLoRA through LLaMA Factory
- Primary public config: `training/llamafactory/qwen25_3b_qlora_sft_v21.yaml`
- Serving config: `training/llamafactory/qwen25_3b_qlora_api_v21.yaml`

## Data

The v2.1 dataset is generated from project chunks, science textbook chunks, fixed evaluation questions, hard-refusal templates, and anti-repetition patches.

| split | count |
| --- | ---: |
| train | 2400 |
| dev | 300 |
| test | 300 |

Type distribution:

| type | count |
| --- | ---: |
| grounded_qa | 1080 |
| citation_qa | 480 |
| compare_or_reasoning | 240 |
| query_rewrite | 240 |
| refusal | 840 |
| anti_repetition | 120 |

Regenerate data:

```bash
python3 sft_dataset_builder_v21.py
```

## Training

Run QLoRA training:

```bash
llamafactory-cli train training/llamafactory/qwen25_3b_qlora_sft_v21.yaml
```

The public repository keeps configs and metric summaries. Adapter weights and checkpoints should remain outside GitHub.

## Serving

Serve the trained adapter through an OpenAI-compatible endpoint:

```bash
API_PORT=8001 llamafactory-cli api training/llamafactory/qwen25_3b_qlora_api_v21.yaml
```

Local app environment:

```text
USE_MOCK=false
GENERATION_BACKEND=local_sft
LOCAL_SFT_BASE_URL=http://127.0.0.1:8001/v1/chat/completions
LOCAL_SFT_API_KEY=EMPTY
LOCAL_SFT_TEMPERATURE=0.2
LOCAL_SFT_TOP_P=0.8
LOCAL_SFT_MAX_TOKENS=512
LOCAL_SFT_REPETITION_PENALTY=1.1
```

## Evaluation

Fixed 50-question evaluation:

```bash
python3 sft_generation_eval_runner.py --questions data/sft_v21/quality_eval_questions.json --output-dir eval_results/sft_generation_v21 --run-name v21_qlora_fixed50
```

Hard-refusal evaluation:

```bash
python3 sft_generation_eval_runner.py --questions data/sft_v21/hard_refusal_eval_questions.json --output-dir eval_results/sft_generation_v21 --run-name v21_qlora_hard_refusal100
```

Core results:

| run | avg keyword | citation complete | structure complete | repeated sections | refusal correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| v2.1 QLoRA fixed50 | 0.8317 | 1.00 | 1.00 | 0.00 | 0.0909 |
| v2.1 QLoRA hard-refusal100 | 0.6225 | 1.00 | 1.00 | 0.00 | 0.9300 |

The result should be read as improved refusal stability on the dedicated hard-refusal set. For broader unsupported questions, the deployed system also relies on RAG/Workflow context sufficiency control.
