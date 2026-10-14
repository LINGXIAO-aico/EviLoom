from __future__ import annotations

import numpy as np

from eviloom import (
    FlowEvidence,
    SelectorConfig,
    build_calibrated_operator_channels,
    select_eviloom,
)


def main() -> None:
    rng = np.random.default_rng(7)
    n, dimension = 180, 32
    timestamps = np.arange(n, dtype=np.float64)
    # Smooth temporal features make neighboring candidates deliberately
    # redundant, like adjacent 1-fps video frames.
    frequencies = np.linspace(0.015, 0.12, dimension // 2)
    phases = timestamps[:, None] * frequencies[None, :]
    features = np.concatenate([np.sin(phases), np.cos(phases)], axis=1)
    features += 0.03 * rng.normal(size=(n, dimension))
    features /= np.linalg.norm(features, axis=1, keepdims=True)

    query_scores = rng.normal(0.0, 0.12, size=n)
    query_scores += 1.8 * np.exp(-0.5 * ((timestamps - 42.0) / 5.0) ** 2)
    query_scores += 1.5 * np.exp(-0.5 * ((timestamps - 126.0) / 7.0) ** 2)

    target_scores = rng.normal(0.0, 0.12, size=n)
    target_scores += 2.0 * np.exp(-0.5 * ((timestamps - 126.0) / 4.0) ** 2)
    reference_scores = rng.normal(0.0, 0.12, size=n)
    reference_scores += 2.0 * np.exp(-0.5 * ((timestamps - 42.0) / 4.0) ** 2)

    question = "What does the girl do after she falls down?"
    channels, operator_audit = build_calibrated_operator_channels(
        question,
        [FlowEvidence("the girl does", "she falls down", target_scores, reference_scores, "F1")],
        timestamps,
    )
    result = select_eviloom(
        query_scores,
        features,
        timestamps,
        channels=channels,
        config=SelectorConfig(budget=32),
    )
    print("selected timestamps:", [round(float(timestamps[i]), 1) for i in result.indices])
    print("evidence packets:", result.packets)
    print("operator audit:", operator_audit)


if __name__ == "__main__":
    main()
