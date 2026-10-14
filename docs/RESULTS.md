# Results and evaluation scope

This repository reports two evaluation tracks. Results are compared only within
the same track because their frame-loading, prompting and scoring paths differ.

## 1. LongVideoBench in the original research pipeline

**Setup:** complete 1,337-question validation split, Qwen2.5-VL-7B-Instruct,
32 selected frames, 1-fps candidate pool, subtitles enabled, and the same answer
prompt/parser for every row.

| Selector | Accuracy | 95% CI |
|---|---:|---:|
| Top-K | 55.50 | [52.80, 58.19] |
| FOCUS (same local evaluation stack) | 55.57 | [52.88, 58.26] |
| Top-K + evidence packets | 56.99 | [54.30, 59.61] |
| **EviLoom** | **58.26** | **[55.57, 60.88]** |

The direct comparison isolates two effects:

- Top-K to Top-K + packets: **+1.49 points** from presentation;
- Top-K + packets to EviLoom: **+1.27 points** from the distribution-coverage
  selector under the same packet presentation;
- Top-K to EviLoom: **+2.76 points** end to end.

The paired EviLoom-versus-Top-K-Packet comparison contained 60 rescues and 43
regressions (17 net rescues). Its paired 95% interval for the accuracy difference
was [-0.22, 2.77] percentage points, so this result is evidence of a useful effect,
not a claim of statistical significance by itself.

Machine-readable values are in
[`results/longvideobench_original_protocol.csv`](../results/longvideobench_original_protocol.csv).

## 2. Common LMMS-Eval path

All EviLoom runs in this track used Qwen2.5-VL-7B-Instruct, a 32-frame budget
and dataset-native LMMS-Eval scoring.

| Dataset | N | Metric | Score |
|---|---:|---|---:|
| Video-MME | 2700 | perception score | 63.3333 |
| LongVideoBench | 1337 | LVB accuracy | 61.5560 |
| MLVU | 2174 | seven-task macro accuracy | 69.4110 |
| LVBench | 1549 | LVBench score | 45.5132 |

### Full MLVU selector comparison

| Selector | Score |
|---|---:|
| Uniform | 62.4445 |
| FOCUS | 66.9391 |
| AKS | 67.4096 |
| BOLT | 68.0256 |
| Top-K | 68.1950 |
| **EviLoom** | **69.4110** |

These are project-side common-protocol evaluations. They compare selectors under
the same local answerer and evaluation path; they are not presented as verbatim
reproductions of every baseline paper's complete preprocessing and model stack.

Machine-readable values are in [`results/`](../results/).

## Interpretation boundary

Supported by the released results:

- distribution-aware coverage can outperform independent Top-K ranking under a
  fixed frame budget;
- the LongVideoBench gain is not explained by packet presentation alone;
- EviLoom integrates with a standard LMMS-Eval answer/scoring path;
- selection is deterministic once retrieval scores, embeddings and configuration
  are fixed.

Not claimed:

- universal superiority over every official implementation or model stack;
- statistical significance on every benchmark;
- direct comparability between the original research pipeline and LMMS-Eval
  absolute scores.
