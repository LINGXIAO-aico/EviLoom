from __future__ import annotations

import json

import numpy as np

from eviloom import (
    FlowEvidence,
    SelectorConfig,
    build_calibrated_operator_channels,
    parse_structured_plan,
    robust_distribution,
    select_eviloom,
)


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    scores = rng.normal(size=80)
    features = rng.normal(size=(80, 16))
    timestamps = np.arange(80, dtype=np.float64)
    return scores, features, timestamps


def test_distribution_is_finite_probability() -> None:
    distribution = robust_distribution(np.asarray([1.0, np.nan, 3.0, np.inf]))
    assert np.isfinite(distribution).all()
    assert np.isclose(distribution.sum(), 1.0)


def test_selector_is_deterministic_and_respects_budget() -> None:
    scores, features, timestamps = fixture()
    first = select_eviloom(scores, features, timestamps, config=SelectorConfig(budget=32))
    second = select_eviloom(scores, features, timestamps, config=SelectorConfig(budget=32))
    assert first.indices == second.indices
    assert len(first.indices) == len(set(first.indices)) == 32
    assert first.indices == sorted(first.indices)
    packet_indices = [index for packet in first.packets for index in packet["indices"]]
    assert packet_indices == first.indices


def test_option_blind_plan_parser_rejects_unsupported_query() -> None:
    raw = json.dumps(
        {
            "anchor_facts": ["a girl is on the ground"],
            "flows": [
                {"target": "the girl stands up", "reference": "the girl is on the ground"},
                {"target": "a red helicopter arrives", "reference": "an explosion"},
            ],
        }
    )
    plan = parse_structured_plan(raw, "What does the girl do after she falls down?")
    assert plan.parsed_ok
    assert len(plan.flows) == 1
    assert len(plan.rejected) == 1


def test_operator_weights_are_bounded() -> None:
    _, _, timestamps = fixture()
    target = np.exp(-0.5 * ((timestamps - 60.0) / 2.0) ** 2)
    reference = np.exp(-0.5 * ((timestamps - 20.0) / 2.0) ** 2)
    channels, audit = build_calibrated_operator_channels(
        "What happens after the person opens the door?",
        [FlowEvidence("happens", "opens the door", target, reference, "F1")],
        timestamps,
    )
    assert audit["final_weight_sum"] <= 0.20 + 1e-12
    assert all(channel.weight <= 0.12 + 1e-12 for channel in channels)
