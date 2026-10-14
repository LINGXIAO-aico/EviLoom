from __future__ import annotations

import numpy as np

from .packet import build_packets
from .types import EvidenceChannel, SelectionResult, SelectorConfig


def robust_distribution(
    scores: np.ndarray,
    temperature: float = 1.0,
    z_clip: float = 5.0,
) -> np.ndarray:
    """Convert arbitrary frame scores into a robust probability distribution."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("scores must be a vector")
    if len(values) == 0:
        return values.copy()
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(len(values), 1.0 / len(values), dtype=np.float64)
    fill = float(np.median(values[finite]))
    values = np.where(finite, values, fill)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if scale <= 1e-8:
        scale = float(np.std(values))
    if scale <= 1e-8:
        return np.full(len(values), 1.0 / len(values), dtype=np.float64)
    z = np.clip((values - median) / scale, -float(z_clip), float(z_clip))
    logits = z / max(float(temperature), 1e-8)
    logits -= float(np.max(logits))
    weights = np.exp(logits)
    return weights / max(float(np.sum(weights)), 1e-12)


def normalize_features(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("frame_features must have shape [N, D]")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.where(np.isfinite(values), np.maximum(values, 0.0), 0.0)
    total = float(values.sum())
    return values / total if total > 1e-12 else np.zeros_like(values)


def _kernel_lists(
    features: np.ndarray,
    times: np.ndarray,
    time_scale_sec: float,
    time_window_sec: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    neighbors: list[np.ndarray] = []
    kernels: list[np.ndarray] = []
    tau = max(float(time_scale_sec), 1e-8)
    window = max(float(time_window_sec), 0.0)
    for candidate in range(len(times)):
        left = int(np.searchsorted(times, times[candidate] - window, side="left"))
        right = int(np.searchsorted(times, times[candidate] + window, side="right"))
        local = np.arange(left, right, dtype=np.int64)
        visual = np.maximum(features[local] @ features[candidate], 0.0)
        temporal = np.exp(-np.abs(times[local] - times[candidate]) / tau)
        neighbors.append(local)
        kernels.append(np.clip(visual * temporal, 0.0, 1.0))
    return neighbors, kernels


def select_eviloom(
    query_scores: np.ndarray,
    frame_features: np.ndarray,
    timestamps: np.ndarray,
    *,
    channels: list[EvidenceChannel] | None = None,
    config: SelectorConfig | None = None,
) -> SelectionResult:
    """Select a fixed-size evidence set with EviLoom's multi-channel objective.

    The base query and every auxiliary channel keep separate facility-location
    coverage states.  This avoids collapsing distinct evidence needs into one
    score vector before optimization.
    """
    cfg = config or SelectorConfig()
    p_query = robust_distribution(query_scores, cfg.temperature)
    features = normalize_features(frame_features)
    times = np.asarray(timestamps, dtype=np.float64)
    if len(p_query) != len(features) or len(times) != len(features):
        raise ValueError("query_scores, frame_features, and timestamps must agree on N")
    if len(times) and np.any(np.diff(times) < 0):
        raise ValueError("timestamps must be sorted")

    valid_channels: list[EvidenceChannel] = []
    for channel in channels or []:
        distribution = _normalize(channel.distribution)
        if len(distribution) != len(times):
            raise ValueError(f"channel {channel.name!r} has the wrong length")
        if channel.weight > 0.0 and distribution.sum() > 0.0:
            valid_channels.append(
                EvidenceChannel(
                    name=channel.name,
                    distribution=distribution,
                    weight=float(channel.weight),
                    time_scale_sec=float(channel.time_scale_sec),
                    time_window_sec=float(channel.time_window_sec),
                )
            )

    n = len(times)
    target = min(max(int(cfg.budget), 0), n)
    alpha = float(np.clip(cfg.direct_weight, 0.0, 1.0))
    base_neighbors, base_kernels = _kernel_lists(
        features, times, cfg.time_scale_sec, cfg.time_window_sec
    )
    channel_kernels = [
        _kernel_lists(features, times, channel.time_scale_sec, channel.time_window_sec)
        for channel in valid_channels
    ]
    base_covered = np.zeros(n, dtype=np.float64)
    channel_covered = [np.zeros(n, dtype=np.float64) for _ in valid_channels]
    available = np.ones(n, dtype=bool)
    selected: list[int] = []
    roles: dict[int, str] = {}
    trace: list[dict] = []

    for step in range(target):
        gains = np.full(n, -np.inf, dtype=np.float64)
        details: dict[int, tuple[float, list[float]]] = {}
        for candidate in np.flatnonzero(available):
            local = base_neighbors[candidate]
            marginal = np.maximum(base_kernels[candidate] - base_covered[local], 0.0)
            cover_gain = float(np.sum(p_query[local] * marginal))
            base_gain = alpha * float(p_query[candidate]) + (1.0 - alpha) * cover_gain
            auxiliary: list[float] = []
            for channel_i, channel in enumerate(valid_channels):
                neighbors, kernels = channel_kernels[channel_i]
                c_local = neighbors[candidate]
                c_marginal = np.maximum(
                    kernels[candidate] - channel_covered[channel_i][c_local], 0.0
                )
                auxiliary.append(
                    channel.weight
                    * float(np.sum(channel.distribution[c_local] * c_marginal))
                )
            gains[candidate] = base_gain + sum(auxiliary)
            details[int(candidate)] = (base_gain, auxiliary)

        best = int(np.argmax(gains))
        if not np.isfinite(gains[best]):
            break
        base_gain, auxiliary = details[best]
        role = "direct"
        if auxiliary and max(auxiliary) > base_gain:
            role = valid_channels[int(np.argmax(auxiliary))].name
        selected.append(best)
        roles[best] = role
        available[best] = False
        local = base_neighbors[best]
        base_covered[local] = np.maximum(base_covered[local], base_kernels[best])
        for channel_i in range(len(valid_channels)):
            neighbors, kernels = channel_kernels[channel_i]
            c_local = neighbors[best]
            channel_covered[channel_i][c_local] = np.maximum(
                channel_covered[channel_i][c_local], kernels[best]
            )
        trace.append(
            {
                "step": step,
                "index": best,
                "timestamp": float(times[best]),
                "gain": float(gains[best]),
                "role": role,
            }
        )

    if len(selected) < target:
        for index in np.argsort(-p_query, kind="mergesort"):
            if int(index) not in selected:
                selected.append(int(index))
                roles[int(index)] = "direct"
            if len(selected) >= target:
                break

    indices = sorted(selected[:target], key=lambda i: float(times[i]))
    packets = build_packets(indices, times, roles=roles, max_gap_sec=cfg.packet_gap_sec)
    combined = p_query.copy()
    for channel in valid_channels:
        combined += channel.weight * channel.distribution
    combined = _normalize(combined)
    return SelectionResult(
        indices=indices,
        demand=combined,
        roles={int(i): roles.get(int(i), "direct") for i in indices},
        packets=packets,
        diagnostics={
            "objective": "direct_mass_plus_multi_channel_local_facility_location",
            "greedy_cardinality_guarantee": "1-1/e for the monotone submodular proxy",
            "budget": int(cfg.budget),
            "strict_budget": len(indices) == target,
            "channel_weights": {c.name: c.weight for c in valid_channels},
            "covered_query_demand": float(np.sum(p_query * base_covered)),
            "trace": trace,
        },
    )

