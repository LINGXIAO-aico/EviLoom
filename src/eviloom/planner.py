"""Option-blind structured planning utilities.

The library intentionally does not depend on a particular VLM. Call a model
with :data:`PLANNER_PROMPT`, then validate its JSON with
``parse_structured_plan`` before computing retrieval scores.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

PLANNER_PROMPT = """You are an option-blind evidence planner for long-video QA.
You receive only a QUESTION and a few query-relevant ANCHOR frames. Never answer
the question and never inspect or enumerate answer choices.

Return only structures supported by the question or visible anchors:
- target: a visible relation, state, or action whose evidence is needed;
- reference: an event or state explicitly named in the question or anchors;
- direction: -1 for earlier, 0 for same/either time, +1 for later;
- multiplicity: whether separated occurrences may both matter.

Do not invent a cause, identity, entity, action, color, shape, count, or answer.
If no reliable structure is supported, return an empty flows list.

STRICT JSON:
{{"anchor_facts":["visible fact"],"flows":[{{"target":"...",
"reference":"...","direction":0.0,"multiplicity":0.0}}],"reason":"..."}}

QUESTION: {question}
"""


@dataclass(frozen=True)
class FlowSpec:
    target: str
    reference: str
    direction: float = 0.0
    multiplicity: float = 0.0
    flow_id: str = "flow"


@dataclass
class StructuredPlan:
    anchor_facts: list[str] = field(default_factory=list)
    flows: list[FlowSpec] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    reason: str = ""
    parsed_ok: bool = False


_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of",
    "in", "on", "at", "for", "and", "or", "what", "why", "how",
    "which", "who", "this", "that", "it", "with", "from", "as",
}


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", str(text).lower())
    return {word for word in words if word not in _STOP and len(word) > 1}


def _supported(text: str, support: str) -> bool:
    requested = _tokens(text)
    available = _tokens(support)
    return bool(requested and requested.intersection(available))


def parse_structured_plan(raw: str, question: str, *, max_flows: int = 3) -> StructuredPlan:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return StructuredPlan(parsed_ok=False)
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return StructuredPlan(parsed_ok=False)
    facts = [re.sub(r"\s+", " ", str(x)).strip()[:240] for x in data.get("anchor_facts", [])]
    facts = [fact for fact in facts if fact][:8]
    support = " ".join([question, *facts])
    flows: list[FlowSpec] = []
    rejected: list[dict] = []
    for position, item in enumerate(data.get("flows", [])):
        if not isinstance(item, dict):
            continue
        target = re.sub(r"\s+", " ", str(item.get("target", ""))).strip()[:240]
        reference = re.sub(r"\s+", " ", str(item.get("reference", ""))).strip()[:240]
        if not target or not reference or not _supported(target, support) or not _supported(reference, support):
            rejected.append({"target": target, "reference": reference, "reason": "unsupported_text"})
            continue
        try:
            direction = max(-1.0, min(1.0, float(item.get("direction", 0.0))))
            multiplicity = max(0.0, min(1.0, float(item.get("multiplicity", 0.0))))
        except (TypeError, ValueError):
            direction, multiplicity = 0.0, 0.0
        flows.append(FlowSpec(target, reference, direction, multiplicity, f"F{position + 1}"))
        if len(flows) >= max_flows:
            break
    return StructuredPlan(
        anchor_facts=facts,
        flows=flows,
        rejected=rejected,
        reason=str(data.get("reason", ""))[:240],
        parsed_ok=True,
    )
