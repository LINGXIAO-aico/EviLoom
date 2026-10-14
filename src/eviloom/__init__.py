"""EviLoom: training-free evidence selection for long-video QA."""

from .operators import (
    FlowEvidence,
    build_calibrated_operator_channels,
    build_query_mode_channel,
    build_temporal_operator_channel,
    infer_temporal_relation,
)
from .packet import build_packets
from .planner import PLANNER_PROMPT, FlowSpec, StructuredPlan, parse_structured_plan
from .selector import robust_distribution, select_eviloom
from .types import (
    EvidenceChannel,
    OperatorConfig,
    SelectionResult,
    SelectorConfig,
)

__all__ = [
    "PLANNER_PROMPT",
    "EvidenceChannel",
    "FlowEvidence",
    "FlowSpec",
    "OperatorConfig",
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
    "select_eviloom",
]

