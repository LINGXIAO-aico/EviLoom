# Reproduction guide

## Lightweight selector check

```bash
python -m pip install -e ".[dev]"
pytest
python examples/synthetic_demo.py
```

The public repository intentionally omits datasets, videos, model weights,
decoded frames, CLIP/BLIP caches and per-question model outputs.

## End-to-end benchmark protocol

The validated experiment used:

- Qwen2.5-VL-7B-Instruct as answerer;
- LMMS-Eval task adapters;
- 32 selected frames per question;
- cached 1-fps candidate frames and visual embeddings;
- option-blind selection;
- dataset-native scoring.

To integrate EviLoom into another evaluation stack:

1. decode or cache candidate frames in timestamp order;
2. compute one normalized visual embedding per candidate;
3. compute question-to-frame retrieval scores;
4. optionally produce validated planner target/reference queries and their
   retrieval scores;
5. call `select_eviloom` with `budget=32`;
6. load selected frames in timestamp/packet order into the answer model;
7. score predictions with the benchmark's official adapter.

The portable package covers steps 4-6 at the selection/presentation level. It
does not force a particular embedding model or video decoder, which keeps the
algorithm reusable and prevents hidden multi-gigabyte dependencies.
