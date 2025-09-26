from typing import Dict
from dataclasses import dataclass
import math

# -----------------------------------------------------------------------------
# Metrics (fidelity proxy, TVD, IST) via simulation
# -----------------------------------------------------------------------------
@dataclass
class Metrics:
    combo: str
    circuit: str
    nqubits: int
    depth_p: int
    shots: int
    tvd: float
    ist: float
    fidelity_proxy: float


def _counts_to_prob(counts: Dict[str, int], shots: int) -> Dict[str, float]:
    return {k: v / shots for k, v in counts.items()} if shots else {}


def total_variation_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def inference_strength(p: Dict[str, float], q: Dict[str, float]) -> float:
    """IST: separation between most likely outcomes. Range ~[0,1].
    This is a simple proxy; refine if you have a formal definition variant.
    """
    def mode_prob(d: Dict[str, float]) -> float:
        return max(d.values()) if d else 0.0
    return abs(mode_prob(p) - mode_prob(q))


def fidelity_like(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Bhattacharyya coefficient as a fidelity-like score between distributions."""
    keys = set(p) | set(q)
    return sum(math.sqrt(p.get(k, 0.0) * q.get(k, 0.0)) for k in keys)