
import math
import random
from typing import Optional
from qiskit import QuantumCircuit

# -----------------------------------------------------------------------------
# Clifford Dummy Generator (RZ→Clifford heuristic)
# -----------------------------------------------------------------------------
class CliffordDummy:
    """Approximate non-Clifford RZ gates into Clifford angles to get a stabilizer circuit.

    Heuristic (paper-like):
      - If θ is close to odd multiples of π/4, snap to nearest k*π/2 with small randomness.
      - Otherwise quantize to multiples of π/2 deterministically.
    TODO: add alternative heuristics (state-preserving, device-adaptive, etc.).
    """

    @staticmethod
    def quantize_angle(theta: float, jitter: bool = True) -> float:
        # Normalize angle to (-π, π]
        t = (theta + math.pi) % (2 * math.pi) - math.pi
        # Distance to odd multiples of π/4
        k = round(t / (math.pi / 4))
        near_odd_quarter = (k % 2 != 0) and (abs(t - k * (math.pi / 4)) < (math.pi / 100))
        if near_odd_quarter and jitter:
            # Randomly map to a nearby π/2 multiple (paper-like random rounding)
            k2 = round(t / (math.pi / 2)) + random.choice([-1, 0, 1])
            return k2 * (math.pi / 2)
        # Default deterministic map to π/2 grid
        return round(t / (math.pi / 2)) * (math.pi / 2)

    @staticmethod
    def to_clifford_dummy(qc: QuantumCircuit, seed: Optional[int] = 42) -> QuantumCircuit:
        if seed is not None:
            random.seed(seed)
        dq = QuantumCircuit(qc.num_qubits, qc.num_clbits)
        for ci in qc.data:
            instr = ci.operation
            qargs = ci.qubits
            cargs = ci.clbits
            name = instr.name
            if name == "rz":
                theta = float(instr.params[0])
                dq.rz(CliffordDummy.quantize_angle(theta), qargs[0])
            elif name == "crz":
                theta = float(instr.params[0])
                dq.crz(CliffordDummy.quantize_angle(theta), qargs[0], qargs[1])
            else:
                dq.append(instr, qargs, cargs)
        # Preserve measurements
        return dq