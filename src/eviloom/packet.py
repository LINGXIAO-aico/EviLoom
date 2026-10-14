from __future__ import annotations

import numpy as np


def build_packets(
    indices: list[int],
    timestamps: np.ndarray,
    *,
    roles: dict[int, str] | None = None,
    max_gap_sec: float = 8.0,
) -> list[dict]:
    """Group selected frames into local temporal evidence packets.

    Packetization changes presentation only. It never adds, drops, or replaces
    a selected frame.
    """
    times = np.asarray(timestamps, dtype=np.float64)
    role_map = roles or {}
    ordered = sorted(set(map(int, indices)), key=lambda i: float(times[i]))
    groups: list[list[int]] = []
    current: list[int] = []
    for index in ordered:
        split = bool(
            current
            and (
                float(times[index] - times[current[-1]]) > float(max_gap_sec)
                or role_map.get(index, "direct") != role_map.get(current[-1], "direct")
            )
        )
        if split:
            groups.append(current)
            current = []
        current.append(index)
    if current:
        groups.append(current)
    return [
        {
            "packet_id": packet_id,
            "indices": group,
            "timestamps": [float(times[i]) for i in group],
            "start_sec": float(times[group[0]]),
            "end_sec": float(times[group[-1]]),
            "role": role_map.get(group[0], "direct"),
        }
        for packet_id, group in enumerate(groups)
    ]

