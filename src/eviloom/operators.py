from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from .selector import robust_distribution
from .types import EvidenceChannel, OperatorConfig

RELATION_CAPS = {
    "first": 0.12,
    "last": 0.12,
    "before": 0.10,
    "after": 0.08,
    "recurrence": 0.08,
    # The server audit found no gain from duplicating ordinary when/while
    # questions, so same-time remains diagnostic-only by default.
    "same_time": 0.0,
}


@dataclass(frozen=True)
class FlowEvidence:
    """One option-blind target/reference retrieval pair from the planner."""

    target: str
    reference: str
    target_scores: np.ndarray
    reference_scores: np.ndarray
    flow_id: str = "flow"
    metadata: dict = field(default_factory=dict)


def infer_temporal_relation(question: str) -> str:
    """Return only relations explicitly licensed by the question wording."""
    q = re.sub(r"\s+", " ", str(question).lower())
    if re.search(r"\bafter\b", q) or re.search(
        r"\bwhat (?:happens|happened|appears|appeared|comes|came) next\b", q
    ) or re.search(r"\b(?:then|subsequently|later)\b", q):
        return "after"
    if re.search(r"\b(before|prior to|previously|earlier)\b", q):
        return "before"
    if re.search(r"\b(first|initially|earliest)\b", q):
        return "first"
    if re.search(r"\b(last|finally|latest)\b", q):
        return "last"
    if "at the same time" in q or "simultaneously" in q or re.search(r"\b(when|while)\b", q):
        return "same_time"
    if re.search(r"\b(elsewhere|again|another|other time|both|each time)\b", q):
        return "recurrence"
    return "none"


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.where(np.isfinite(values), np.maximum(values, 0.0), 0.0)
    total = float(values.sum())
    return values / total if total > 1e-12 else np.zeros_like(values)


def _robust_z(scores: np.ndarray) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    x = np.where(finite, x, float(np.median(x[finite])))
    median = float(np.median(x))
    scale = 1.4826 * float(np.median(np.abs(x - median)))
    if scale <= 1e-8:
        scale = float(np.std(x))
    return np.clip((x - median) / max(scale, 1e-8), -8.0, 8.0)


def retrieval_prominence(scores: np.ndarray) -> float:
    """Robust confidence of a retrieval peak, independent of video length."""
    z = _robust_z(scores)
    if not len(z):
        return 0.0
    return float(np.clip((float(np.percentile(z, 97.0)) - 0.5) / 3.5, 0.0, 1.0))


def _local_peaks(
    scores: np.ndarray,
    timestamps: np.ndarray,
    *,
    max_peaks: int = 12,
    nms_sec: float = 2.0,
    min_z: float = 0.5,
) -> list[int]:
    values = np.asarray(scores, dtype=np.float64)
    times = np.asarray(timestamps, dtype=np.float64)
    if not len(values):
        return []
    z = _robust_z(values)
    candidates = [
        i
        for i in range(len(values))
        if z[i] >= min_z
        and (i == 0 or values[i] >= values[i - 1])
        and (i == len(values) - 1 or values[i] >= values[i + 1])
    ]
    if not candidates:
        candidates = [int(np.argmax(values))]
    chosen: list[int] = []
    for index in sorted(candidates, key=lambda i: (-float(z[i]), float(times[i]))):
        if all(abs(float(times[index] - times[other])) >= nms_sec for other in chosen):
            chosen.append(int(index))
        if len(chosen) >= max_peaks:
            break
    return chosen


def _weighted_local_peaks(
    scores: np.ndarray,
    timestamps: np.ndarray,
    *,
    max_peaks: int,
    nms_sec: float,
    min_z: float,
) -> tuple[list[int], np.ndarray]:
    peaks = _local_peaks(
        scores,
        timestamps,
        max_peaks=max_peaks,
        nms_sec=nms_sec,
        min_z=min_z,
    )
    if not peaks:
        return [], np.zeros(0, dtype=np.float64)
    z = _robust_z(scores)
    logits = np.asarray([z[index] for index in peaks], dtype=np.float64)
    weights = np.exp(logits - float(np.max(logits)))
    return peaks, _normalize(weights)


def _spikes(
    indices: list[int],
    timestamps: np.ndarray,
    sigma_sec: float,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    times = np.asarray(timestamps, dtype=np.float64)
    out = np.zeros(len(times), dtype=np.float64)
    if not indices:
        return out
    sigma = max(float(sigma_sec), 1e-6)
    mass = _normalize(np.ones(len(indices)) if weights is None else weights)
    for index, weight in zip(indices, mass):
        out += float(weight) * np.exp(
            -0.5 * np.square((times - float(times[index])) / sigma)
        )
    return _normalize(out)


def build_temporal_operator_channel(
    question: str,
    target_scores: np.ndarray,
    reference_scores: np.ndarray,
    timestamps: np.ndarray,
    *,
    weight: float = 0.15,
    sigma_sec: float = 4.0,
    min_nms_sec: float = 2.0,
    duration_fraction: float = 0.005,
    max_peaks: int = 12,
    max_reference_peaks: int = 6,
    min_peak_z: float = 1.0,
    direction_tau_sec: float = 32.0,
    direction_duration_fraction: float = 0.05,
    direction_margin_sec: float = 1.0,
) -> tuple[EvidenceChannel | None, dict]:
    """Create a bounded evidence channel for an explicit temporal relation.

    ``target_scores`` and ``reference_scores`` should come from option-blind
    text/frame retrieval. The operator does not invent a free-form hypothesis.
    """
    relation = infer_temporal_relation(question)
    if relation == "none" or weight <= 0:
        return None, {"relation": relation, "valid_side_mass": 0.0}
    times = np.asarray(timestamps, dtype=np.float64)
    duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    nms = max(float(min_nms_sec), float(duration_fraction) * duration)
    target_peaks, target_weights = _weighted_local_peaks(
        target_scores,
        times,
        max_peaks=max_peaks,
        nms_sec=nms,
        min_z=min_peak_z,
    )
    reference_peaks, _ = _weighted_local_peaks(
        reference_scores,
        times,
        max_peaks=max_reference_peaks,
        nms_sec=nms,
        min_z=min_peak_z,
    )
    debug = {
        "relation": relation,
        "target_peaks": target_peaks,
        "reference_peaks": reference_peaks,
        "target_prominence": retrieval_prominence(target_scores),
        "reference_prominence": retrieval_prominence(reference_scores),
        "valid_side_mass": 0.0,
    }
    if not target_peaks:
        return None, debug

    if relation == "first":
        distribution = _spikes([min(target_peaks, key=lambda i: times[i])], times, sigma_sec)
    elif relation == "last":
        distribution = _spikes([max(target_peaks, key=lambda i: times[i])], times, sigma_sec)
    elif relation == "recurrence":
        if len(target_peaks) < 2:
            return None, debug
        distribution = _spikes(target_peaks, times, sigma_sec)
    elif relation == "same_time":
        joint = np.sqrt(
            robust_distribution(target_scores) * robust_distribution(reference_scores)
        )
        distribution = _normalize(joint)
    else:
        if not reference_peaks:
            return None, debug
        tau = max(float(direction_tau_sec), float(direction_duration_fraction) * duration)
        accepted: list[int] = []
        accepted_weights: list[float] = []
        for target in target_peaks:
            valid_distances = []
            for reference in reference_peaks:
                delta = float(times[target] - times[reference])
                if relation == "after" and delta >= float(direction_margin_sec):
                    valid_distances.append(delta)
                if relation == "before" and delta <= -float(direction_margin_sec):
                    valid_distances.append(-delta)
            if valid_distances:
                accepted.append(target)
                peak_weight = float(target_weights[target_peaks.index(target)])
                accepted_weights.append(peak_weight * math.exp(-min(valid_distances) / tau))
        if not accepted:
            return None, debug
        debug["valid_side_mass"] = float(
            sum(target_weights[target_peaks.index(index)] for index in accepted)
        ) / max(float(np.sum(target_weights)), 1e-12)
        distribution = _spikes(
            accepted,
            times,
            sigma_sec,
            _normalize(np.asarray(accepted_weights)),
        )

    return EvidenceChannel(
        name=f"operator:{relation}",
        distribution=distribution,
        weight=float(np.clip(weight, 0.0, 0.2)),
        time_scale_sec=4.0,
        time_window_sec=12.0,
    ), debug


def build_query_mode_channel(
    query_scores: np.ndarray,
    timestamps: np.ndarray,
    *,
    weight: float = 0.08,
    sigma_sec: float = 4.0,
) -> EvidenceChannel | None:
    times = np.asarray(timestamps, dtype=np.float64)
    duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    peaks = _local_peaks(
        query_scores,
        times,
        nms_sec=max(2.0, 0.005 * duration),
    )
    if not peaks or weight <= 0:
        return None
    return EvidenceChannel(
        name="query_modes",
        distribution=_spikes(peaks, times, sigma_sec),
        weight=float(np.clip(weight, 0.0, 0.2)),
        time_scale_sec=4.0,
        time_window_sec=12.0,
    )


def build_calibrated_operator_channels(
    question: str,
    flows: list[FlowEvidence],
    timestamps: np.ndarray,
    *,
    config: OperatorConfig | None = None,
) -> tuple[list[EvidenceChannel], dict]:
    """Build the bounded temporal channels used by the validated method.

    A planner may propose many flows, but a proposal changes selection only if
    both its target and temporal landmark produce prominent retrieval peaks.
    This deliberately separates *proposal rate* from *accepted evidence*.
    """
    cfg = config or OperatorConfig()
    relation = infer_temporal_relation(question)
    channels: list[EvidenceChannel] = []
    audit: list[dict] = []
    cap = min(float(cfg.single_weight_cap), float(RELATION_CAPS.get(relation, 0.0)))
    for flow in flows:
        if relation == "none" or cap <= 0.0:
            audit.append({"flow_id": flow.flow_id, "accepted": False, "reason": "no_licensed_operator"})
            continue
        target_prominence = retrieval_prominence(flow.target_scores)
        reference_prominence = retrieval_prominence(flow.reference_scores)
        if relation in {"first", "last", "recurrence"}:
            reliability = target_prominence
        else:
            reliability = math.sqrt(target_prominence * reference_prominence)
        weight = cap * float(np.clip(reliability, 0.0, 1.0)) ** float(cfg.reliability_power)
        if weight < float(cfg.min_effective_weight):
            audit.append(
                {
                    "flow_id": flow.flow_id,
                    "accepted": False,
                    "reason": "low_calibrated_weight",
                    "reliability": reliability,
                    "weight": weight,
                }
            )
            continue
        channel, operator_debug = build_temporal_operator_channel(
            question,
            flow.target_scores,
            flow.reference_scores,
            timestamps,
            weight=weight,
            sigma_sec=cfg.sigma_sec,
            min_nms_sec=cfg.min_nms_sec,
            duration_fraction=cfg.duration_fraction,
            max_peaks=cfg.max_peaks,
            max_reference_peaks=cfg.max_reference_peaks,
            min_peak_z=cfg.min_peak_z,
            direction_tau_sec=cfg.direction_tau_sec,
            direction_duration_fraction=cfg.direction_duration_fraction,
            direction_margin_sec=cfg.direction_margin_sec,
        )
        if channel is None:
            audit.append({"flow_id": flow.flow_id, "accepted": False, "reason": "empty_operator"})
            continue
        reliability *= float(operator_debug.get("valid_side_mass", 1.0))
        weight = cap * float(np.clip(reliability, 0.0, 1.0)) ** float(cfg.reliability_power)
        if weight < float(cfg.min_effective_weight):
            audit.append(
                {
                    "flow_id": flow.flow_id,
                    "accepted": False,
                    "reason": "low_directional_mass",
                    "reliability": reliability,
                    "weight": weight,
                }
            )
            continue
        channels.append(
            EvidenceChannel(
                name=f"operator:{relation}:{flow.flow_id}",
                distribution=channel.distribution,
                weight=weight,
                time_scale_sec=channel.time_scale_sec,
                time_window_sec=channel.time_window_sec,
            )
        )
        audit.append(
            {
                "flow_id": flow.flow_id,
                "accepted": True,
                "relation": relation,
                "target": flow.target,
                "reference": flow.reference,
                "reliability": reliability,
                "weight": weight,
                "valid_side_mass": operator_debug.get("valid_side_mass", 1.0),
            }
        )

    raw_total = float(sum(channel.weight for channel in channels))
    if raw_total > float(cfg.total_weight_cap) > 0.0:
        scale = float(cfg.total_weight_cap) / raw_total
        channels = [
            EvidenceChannel(
                name=channel.name,
                distribution=channel.distribution,
                weight=channel.weight * scale,
                time_scale_sec=channel.time_scale_sec,
                time_window_sec=channel.time_window_sec,
            )
            for channel in channels
        ]
    return channels, {
        "relation": relation,
        "proposed": len(flows),
        "accepted": len(channels),
        "raw_weight_sum": raw_total,
        "final_weight_sum": float(sum(channel.weight for channel in channels)),
        "flows": audit,
    }

