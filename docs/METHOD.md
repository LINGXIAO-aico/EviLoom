# Method

## 1. Inputs and information boundary

For each video, EviLoom receives a time-sorted candidate set with timestamps,
cached visual embeddings and question-to-frame retrieval scores. Selection is
question-only: answer options, labels, ground-truth intervals and previous
answers are excluded.

An optional planner sees the question and six query-relevant anchor frames. It
may describe a visible target, an explicit reference event and a temporal
direction. Its output is validated against question/anchor text; unsupported
entities or free-form answer hypotheses are rejected.

## 2. Robust evidence distribution

Raw retrieval scores are median/MAD standardized, clipped and passed through a
temperature-controlled softmax. This yields a non-negative distribution over
the whole candidate timeline and is less sensitive to score scale or a few
outliers than raw cosine ranking.

## 3. Local submodular coverage

The base objective combines direct probability mass with facility-location
coverage. A selected frame covers nearby candidates in proportion to both
visual similarity and exponentially decaying temporal proximity. Once one
region is covered, redundant neighbors have smaller marginal gain, allowing
other evidence modes to compete for the fixed budget.

The proxy is non-negative, monotone and submodular. Greedy maximization under a
cardinality budget therefore inherits the standard `(1-1/e)` approximation
guarantee for this proxy objective.

## 4. Calibrated temporal operators

Temporal channels are activated only when licensed by explicit language such as
`first`, `last`, `before`, `after` or recurrence cues. Each channel keeps its
own coverage state instead of being prematurely mixed into one score vector.

Planner proposal is not acceptance. Target/reference retrieval peaks receive a
robust prominence score; the channel weight is proportional to squared
reliability, is discarded below 0.01, is capped per relation, and all operator
weights together are capped at 0.20. If no channel is reliable, EviLoom remains
the distribution-and-coverage selector rather than falling back to Top-K.

## 5. Evidence packets

Selected frames are sorted by time and grouped when they are temporally close
and share the same evidence role. Packetization changes presentation only: it
never adds, removes or replaces a selected frame. This makes local event
continuity visible to the answer model while preserving a fixed 32-frame
selection budget.
