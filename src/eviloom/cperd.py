from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from .selector import normalize_features, robust_distribution
from .types import EviLoomCResult, PairCandidate


class LocalFacilityObjective:
    """Exact sparse implementation of PERD's direct + local coverage proxy."""

    def __init__(
        self,
        demand: np.ndarray,
        features: np.ndarray,
        timestamps: np.ndarray,
        direct_weight: float = 0.35,
        time_scale_sec: float = 4.0,
        time_window_sec: float = 12.0,
    ):
        self.demand = np.asarray(demand, dtype=np.float64)
        self.features = normalize_features(features)
        self.times = np.asarray(timestamps, dtype=np.float64)
        self.alpha = float(np.clip(direct_weight, 0.0, 1.0))
        self.tau = max(float(time_scale_sec), 1e-8)
        self.window = max(float(time_window_sec), 0.0)
        if len(self.features) != len(self.demand) or len(self.times) != len(self.demand):
            raise ValueError("feature, demand, and timestamp shapes disagree")
        self._kernel_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def kernel(self, candidate: int) -> tuple[np.ndarray, np.ndarray]:
        candidate = int(candidate)
        cached = self._kernel_cache.get(candidate)
        if cached is not None:
            return cached
        left = int(np.searchsorted(self.times, self.times[candidate] - self.window, side="left"))
        right = int(np.searchsorted(self.times, self.times[candidate] + self.window, side="right"))
        indices = np.arange(left, right, dtype=np.int64)
        visual = np.maximum(self.features[indices] @ self.features[candidate], 0.0)
        temporal = np.exp(-np.abs(self.times[indices] - self.times[candidate]) / self.tau)
        values = np.clip(visual * temporal, 0.0, 1.0).astype(np.float64)
        self._kernel_cache[candidate] = (indices, values)
        return indices, values

    def coverage(self, selected: Iterable[int]) -> np.ndarray:
        covered = np.zeros(len(self.demand), dtype=np.float64)
        for candidate in selected:
            indices, values = self.kernel(int(candidate))
            covered[indices] = np.maximum(covered[indices], values)
        return covered

    def value(self, selected: Iterable[int]) -> float:
        chosen = list(map(int, selected))
        direct = float(np.sum(self.demand[chosen])) if chosen else 0.0
        return self.alpha * direct + (1.0 - self.alpha) * float(
            np.sum(self.demand * self.coverage(chosen))
        )

    def add_marginals(self, selected: Iterable[int], candidates: Iterable[int]) -> dict[int, float]:
        chosen = set(map(int, selected))
        covered = self.coverage(chosen)
        out: dict[int, float] = {}
        for candidate in candidates:
            candidate = int(candidate)
            if candidate in chosen:
                continue
            indices, values = self.kernel(candidate)
            gain = float(np.sum(self.demand[indices] * np.maximum(values - covered[indices], 0.0)))
            out[candidate] = self.alpha * float(self.demand[candidate]) + (1.0 - self.alpha) * gain
        return out

    def removal_losses(self, selected: Iterable[int]) -> dict[int, float]:
        chosen = list(map(int, selected))
        full = self.value(chosen)
        return {candidate: full - self.value(x for x in chosen if x != candidate) for candidate in chosen}


def clip_span_surplus(query: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    q = np.asarray(query, dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    x, y = normalize_features(np.vstack([left, right]))
    a, b, rho = float(q @ x), float(q @ y), float(x @ y)
    joint = (a * a - 2.0 * rho * a * b + b * b) / max(1.0 - rho * rho, 1e-6)
    return float(max(0.0, min(joint, 1.0) - max(a * a, b * b)))


def propose_pairs(
    objective: LocalFacilityObjective,
    base_indices: list[int],
    query_vector: np.ndarray,
    *,
    runner_count: int = 64,
    critic_pair_count: int = 16,
) -> tuple[list[PairCandidate], dict]:
    base = sorted(set(map(int, base_indices)))
    base_set = set(base)
    excluded = [i for i in range(len(objective.demand)) if i not in base_set]
    marginals = objective.add_marginals(base, excluded)
    runners = sorted(marginals, key=lambda i: (-marginals[i], float(objective.times[i])))[:runner_count]
    proposals: list[PairCandidate] = []
    for runner in runners:
        scored = [
            (clip_span_surplus(query_vector, objective.features[anchor], objective.features[runner]), anchor)
            for anchor in base
        ]
        for score, anchor in sorted(scored, key=lambda x: (-x[0], float(objective.times[x[1]])))[:2]:
            proposals.append(PairCandidate(anchor=int(anchor), runner=int(runner), prefilter_score=float(score)))
    proposals.sort(
        key=lambda p: (-p.prefilter_score, float(objective.times[p.runner]), float(objective.times[p.anchor]))
    )
    return proposals[:critic_pair_count], {
        "runner_pool_size": len(runners),
        "critic_pair_count": min(len(proposals), critic_pair_count),
    }


def calibrate_and_bound(
    candidates: list[PairCandidate],
    objective: LocalFacilityObjective,
    base_indices: list[int],
) -> dict:
    if not candidates:
        return {"positive_edges": 0, "total_pair_weight": 0.0}
    margins = np.asarray([p.critic_margin for p in candidates], dtype=np.float64)
    median = float(np.median(margins))
    mad = float(np.median(np.abs(margins - median)))
    scale = max(1.4826 * mad, float(np.std(margins)), 1e-6)
    for pair in candidates:
        positive = max(0.0, min(float(pair.critic_margin), float(pair.critic_margin) - median) / scale)
        pair.calibrated_score = float(np.tanh(positive))
    losses = objective.removal_losses(base_indices)
    by_anchor: dict[int, list[PairCandidate]] = {}
    for pair in candidates:
        by_anchor.setdefault(pair.anchor, []).append(pair)
    for anchor, edges in by_anchor.items():
        total = sum(edge.calibrated_score for edge in edges)
        if total <= 0.0:
            continue
        exclusive = max(0.0, float(losses.get(anchor, 0.0)))
        for edge in edges:
            edge.weight = exclusive * edge.calibrated_score / (1.0 + total)
    return {
        "median": median,
        "mad": mad,
        "positive_edges": sum(p.weight > 0.0 for p in candidates),
        "total_pair_weight": float(sum(p.weight for p in candidates)),
    }


def _pair_value(selected: Iterable[int], candidates: list[PairCandidate]) -> float:
    chosen = set(map(int, selected))
    return float(
        sum(
            edge.weight
            * ((int(edge.anchor in chosen) + int(edge.runner in chosen)) / 2.0) ** 2
            for edge in candidates
        )
    )


def refine_exact_swaps(
    objective: LocalFacilityObjective,
    base_indices: list[int],
    candidates: list[PairCandidate],
    *,
    max_swaps: int | None = None,
) -> tuple[list[int], list[dict]]:
    selected = set(map(int, base_indices))
    trace: list[dict] = []
    universe = sorted({edge.runner for edge in candidates if edge.weight > 0.0})
    limit = len(selected) if max_swaps is None else int(max_swaps)
    for step in range(limit):
        before_f = objective.value(selected)
        before_g = _pair_value(selected, candidates)
        best = None
        for remove in sorted(selected):
            reduced = selected - {remove}
            reduced_f = objective.value(reduced)
            for add in universe:
                if add in selected:
                    continue
                trial = reduced | {add}
                after_f = reduced_f + objective.add_marginals(reduced, [add])[add]
                after_g = _pair_value(trial, candidates)
                if after_g <= before_g + 1e-15:
                    continue
                gain = (after_f + after_g) - (before_f + before_g)
                key = (float(gain), -float(objective.times[add]), -float(objective.times[remove]))
                if best is None or key > best[0]:
                    best = (key, remove, add, after_f, after_g)
        if best is None or best[0][0] <= 1e-12:
            break
        _, remove, add, after_f, after_g = best
        selected.remove(remove)
        selected.add(add)
        trace.append(
            {
                "step": step,
                "removed": int(remove),
                "added": int(add),
                "gain": float((after_f + after_g) - (before_f + before_g)),
            }
        )
    return sorted(selected, key=lambda i: float(objective.times[i])), trace


Critic = Callable[[str, list[PairCandidate]], list[float]]


def select_eviloom_c(
    question: str,
    base_indices: list[int],
    query_scores: np.ndarray,
    frame_features: np.ndarray,
    timestamps: np.ndarray,
    critic: Critic,
    *,
    runner_count: int = 64,
    critic_pair_count: int = 16,
    max_swaps: int | None = None,
) -> EviLoomCResult:
    """Refine a frozen EviLoom set using bounded conditional complementarity."""
    demand = robust_distribution(query_scores)
    objective = LocalFacilityObjective(demand, frame_features, timestamps)
    query_vector = demand @ objective.features
    candidates, proposal_debug = propose_pairs(
        objective,
        base_indices,
        query_vector,
        runner_count=runner_count,
        critic_pair_count=critic_pair_count,
    )
    margins = critic(question, candidates)
    if len(margins) != len(candidates):
        raise ValueError("critic must return one margin per candidate pair")
    for pair, margin in zip(candidates, margins):
        pair.critic_margin = float(margin)
    calibration = calibrate_and_bound(candidates, objective, base_indices)
    indices, swaps = refine_exact_swaps(
        objective, base_indices, candidates, max_swaps=max_swaps
    )
    return EviLoomCResult(
        indices=indices,
        candidates=candidates,
        accepted_swaps=swaps,
        diagnostics={
            **proposal_debug,
            **calibration,
            "fallback_exact": indices == sorted(base_indices, key=lambda i: float(objective.times[i])),
            "strict_budget": len(indices) == len(base_indices),
        },
    )


# Compatibility alias for the internal research name.
select_cperd = select_eviloom_c

