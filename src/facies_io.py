from __future__ import annotations
import numpy as np

FACIES_INFO = {
    0: "Floodplain",
    1: "Point bar",
    2: "Channel",
    3: "Boundary",
}

def read_facies_intervals_txt(path: str) -> list[tuple[float, float, int]]:
    """
    Parse VWxxxx.txt:
      lines after header are: top, bot, facies (tab-separated)
    Returns list of (top, bot, facies).
    """
    intervals = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # skip header-like lines
            if line.startswith("WELLNAME") or line.startswith("DEPTH") or line.startswith("NULL") or line.startswith("DATE"):
                continue
            if line.startswith("Top") or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            top = float(parts[0])
            bot = float(parts[1])
            fac = int(float(parts[2]))
            intervals.append((top, bot, fac))
    return intervals

def intervals_to_facies_series(
    intervals: list[tuple[float, float, int]],
    t_ms: np.ndarray,
    default_label: int = -1,
) -> np.ndarray:
    """
    Convert interval labels to per-sample series aligned to t_ms (length T).
    We treat top/bot values as ms (or equivalent index scale).
    Fill [top, bot) with facies.
    """
    T = len(t_ms)
    fac = np.full((T,), default_label, dtype=np.int64)
    dt = float(t_ms[1] - t_ms[0]) if T > 1 else 1.0

    for top, bot, y in intervals:
        if bot <= top:
            continue
        # convert to indices (inclusive start, exclusive end)
        i0 = int(np.floor((top - t_ms[0]) / dt + 1e-6))
        i1 = int(np.ceil((bot - t_ms[0]) / dt - 1e-6))
        i0 = max(0, min(T, i0))
        i1 = max(0, min(T, i1))
        if i1 > i0:
            fac[i0:i1] = y

    return fac