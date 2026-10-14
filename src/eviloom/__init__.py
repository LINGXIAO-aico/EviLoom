"""EviLoom: training-free evidence selection for long-video QA."""

from .cperd import select_cperd, select_eviloom_c
from .operators import (
    FlowEvidence,
    build_calibrated_operator_channels,
    build_query_mode_channel,
    build_temporal_operator_channel,
    infer_temporal_relation,
)
from .planner import FlowSpec, PLANNER_PROMPT, StructuredPlan, parse_structured_plan
from .packet import build_packets
from .selector import robust_distribution, select_eviloom, select_perd_v2
from .types import (
    CPERDResult,
    EvidenceChannel,
    EviLoomCResult,
    OperatorConfig,
    SelectionResult,
    SelectorConfig,
)

__all__ = [
    "CPERDResult",
    "EvidenceChannel",
    "FlowEvidence",
    "FlowSpec",
    "EviLoomCResult",
    "OperatorConfig",
    "PLANNER_PROMPT",
    "SelectionResult",
    "SelectorConfig",
    "StructuredPlan",
    "build_calibrated_operator_channels",
    "build_packets",
    "build_query_mode_channel",
    "build_temporal_operator_channel",
    "infer_temporal_relation",
    "parse_structured_plan",
    "robust_distribution",
    "select_cperd",
    "select_eviloom",
    "select_eviloom_c",
    "select_perd_v2",
]

