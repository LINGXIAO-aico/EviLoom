# Results and claim scope

## Full benchmark runs

All four EviLoom runs used Qwen2.5-VL-7B-Instruct, a 32-frame budget and the
same LMMS-Eval-based answering path. Video-MME, LongVideoBench and LVBench are
reported as accuracy-like official task scores; MLVU is the official macro
average across its seven task groups.

| Dataset | N | Metric | Score |
|---|---:|---|---:|
| Video-MME | 2700 | perception score | 63.3333 |
| LongVideoBench | 1337 | LVB accuracy | 61.5560 |
| MLVU | 2174 | 7-task macro accuracy | 69.4110 |
| LVBench | 1549 | LVBench score | 45.5132 |

## Full MLVU selector comparison

| Method | Score |
|---|---:|
| Uniform | 62.4445 |
| FOCUS | 66.9391 |
| AKS | 67.4096 |
| BOLT | 68.0256 |
| Top-K | 68.1950 |
| **EviLoom** | **69.4110** |

These are project-side unified evaluations. AKS, BOLT and FOCUS use their
selector logic in the common local candidate/evaluation stack; this table must
not be read as a verbatim reproduction of each paper's complete environment.

## Later development-set variants

The latest common-protocol 4×300 development comparison was:

| Dataset | EviLoom | EviLoom-C | RCD |
|---|---:|---:|---:|
| Video-MME | **60.00** | **60.00** | 58.33 |
| LongVideoBench | **59.33** | 57.67 | 58.00 |
| MLVU | 67.63 | **69.35** | 68.62 |
| LVBench | 46.00 | 48.00 | **48.67** |

This table is development evidence, not a blind confirmation. It explains why
the public release keeps EviLoom as the main method: later variants offered
dataset-specific gains but no stable cross-dataset dominance, and they were not
run on all complete splits.

## What the results do and do not show

Supported:

- EviLoom is the strongest fully evaluated method produced by this project.
- It improves over the completed MLVU baselines under one unified protocol.
- The method is deterministic once retrieval scores/features are fixed.

Not supported:

- universal superiority over every official implementation;
- statistical significance on all four datasets;
- a claim that packetization alone improves frame selection;
- a claim that the experimental EviLoom-C refinement is generally better.

Machine-readable tables are in [`results/`](../results/).
