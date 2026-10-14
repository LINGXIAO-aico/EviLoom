from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class EvidenceChannel:
    """One independently covered evidence demand over the candidate timeline."""

    name: str
    distribution: np.ndarray
    weight: float
    time_scale_sec: float = 4.0
    time_window_sec: float = 12.0


@dataclass(frozen=True)
class SelectorConfig:
    budget: int = 32
    temperature: float = 1.0
    direct_weight: float = 0.35
    time_scale_sec: float = 4.0
    time_window_sec: float = 12.0
    packet_gap_sec: float = 4.0


@dataclass(frozen=True)
class OperatorConfig:
    """Calibration used by the validated B=32 EviLoom configuration."""

    total_weight_cap: float = 0.20
    single_weight_cap: float = 0.12
    reliability_power: float = 2.0
    min_effective_weight: float = 0.01
    max_peaks: int = 12
    max_reference_peaks: int = 6
    min_nms_sec: float = 4.0
    duration_fraction: float = 0.005
    min_peak_z: float = 1.0
    sigma_sec: float = 2.0
    direction_tau_sec: float = 32.0
    direction_duration_fraction: float = 0.05
    direction_margin_sec: float = 1.0


@dataclass
class SelectionResult:
    indices: list[int]
    demand: np.ndarray
    roles: dict[int, str]
    packets: list[dict]
    diagnostics: dict = field(default_factory=dict)


@dataclass
class PairCandidate:
    anchor: int
    runner: int
    prefilter_score: float
    critic_margin: float = 0.0
    calibrated_score: float = 0.0
    weight: float = 0.0


@dataclass
class EviLoomCResult:
    indices: list[int]
    candidates: list[PairCandidate] = field(default_factory=list)
    accepted_swaps: list[dict] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


# Backward-compatible research name used by the server experiments.
CPERDResult = EviLoomCResult

