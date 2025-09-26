"""
COMPASS Reproduction Scaffold (Qiskit/Python)
================================================
Author: Dong-Wan Kim (Howard)
Goal: Minimal, modular scaffold to reproduce the COMPASS paper's core pipeline
      using Qiskit + Clifford dummy circuits + Aer simulation, with hooks for
      E-COMPASS (Top-k), RZ-approx heuristics, and QAOA-specialized analysis.

How to use (quick start)
------------------------
1) Create a venv and install deps:
   python -m venv .venv && source .venv/bin/activate
   pip install qiskit qiskit-aer numpy pandas matplotlib tqdm

2) Run a small smoke test (BV + 2 pass-combos):
   python compass_repro.py --circuits bv --depths 1 --shots 2000 \
       --combos na-sa-al sa-sa-al --backend aer --save results_bv.csv

3) Plot the CSV later with your favorite tool, or extend `plot_results` below.

Notes
-----
- This is a scaffold: readable, hackable, and safe to extend. Each TODO marks
  a place that is intentionally simple (for clarity) but designed for upgrade.
- Stage naming: Stage 1/2 = pass selection (E-COMPASS ready). Stage 3 hooks
  are provided for ADAPT / Imitation Game integration in a later iteration.
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.providers.fake_provider import GenericBackendV2 as Backend
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (Optimize1qGates, CommutativeCancellation, ConsolidateBlocks, BasisTranslator,)
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel

# -----------------------------------------------------------------------------
# Circuit Factory
# -----------------------------------------------------------------------------
class CircuitFactory:
    """Factory methods for benchmark circuits used in COMPASS paper.

    Implements: BV, QAOA (toy MaxCut on ring graph), and PEA (toy).
    For exact parity with the paper, replace toy versions with exact specs.
    """

    @staticmethod
    def bv(n: int) -> QuantumCircuit:
        """Bernstein–Vazirani with hidden string of ones (simple case)."""
        qc = QuantumCircuit(n + 1, n)
        # Prepare oracle: f(x) = s·x (mod 2) with s = 111...1
        # Initialize last qubit to |1> and Hadamard on all qubits
        qc.x(n)
        qc.h(range(n + 1))
        # Oracle: n CNOTs from each input to ancilla
        for i in range(n):
            qc.cx(i, n)
        # Hadamard and measure inputs
        qc.h(range(n))
        qc.measure(range(n), range(n))
        return qc

    @staticmethod
    def qaoa_ring(n: int, p: int, gamma: float = 0.7, beta: float = 0.3) -> QuantumCircuit:
        """Toy QAOA for MaxCut on ring graph of n nodes, depth p.
        Parameters are fixed (not optimized) for structural benchmarking.
        """
        qc = QuantumCircuit(n, n)
        qc.h(range(n))
        for _ in range(p):
            # Cost unitary: ZZ on ring edges with RZ rotations
            for i in range(n):
                j = (i + 1) % n
                qc.cx(i, j)
                qc.rz(2 * gamma, j)
                qc.cx(i, j)
            # Mixer
            for i in range(n):
                qc.rx(2 * beta, i)
        qc.measure(range(n), range(n))
        return qc

    @staticmethod
    def pea_toy(n: int, k: int = 1) -> QuantumCircuit:
        """Toy Phase Estimation that estimates a phase e^{2πi*phi} with unitary RZ on a target.
        n: number of counting qubits; k: target rotation multiple.
        """
        qc = QuantumCircuit(n + 1, n)
        # Prepare |+> states on counting qubits
        qc.h(range(n))
        # Controlled-U^{2^j} with U = RZ(2π k / 2^n)
        for j in range(n):
            angle = 2 * math.pi * k / (2 ** (n - j))
            qc.crz(angle, j, n)
        # Inverse QFT on counting register
        for j in range(n // 2):
            qc.swap(j, n - 1 - j)
        for j in range(n):
            for m in range(j):
                qc.cp(-math.pi / (2 ** (j - m)), m, j)
            qc.h(j)
        qc.measure(range(n), range(n))
        return qc

# -----------------------------------------------------------------------------
# Pass Combos (Stage 1/2 candidates)
# -----------------------------------------------------------------------------
class PassCombos:
    """Reference pass-combo definitions. Names mirror the proposal (na/sa/al etc.).

    You can either:
      - Use Qiskit's preset pass managers (optimization_level=0..3), or
      - Hand-roll small PassManager pipelines for clarity.

    Below we show both styles. Modify as needed to match the paper more closely.
    """

    @staticmethod
    def preset(level: int) -> PassManager:
        return generate_preset_pass_manager(optimization_level=level)

    @staticmethod
    def na_sa_al() -> PassManager:
        # Simple illustrative pipeline: (naive mapping) + some 1q/2q cancels + layout
        
        pm = PassManager([
            BasisTranslator(sel, ["rz", "rx", "ry", "cx", "id"]),
            ConsolidateBlocks(force_consolidate=True),
            Optimize1qGates(),
            CommutativeCancellation(),
        ])
        return pm

    @staticmethod
    def sa_sa_al() -> PassManager:
        pm = PassManager([
            Optimize1qGates(),
            ConsolidateBlocks(force_consolidate=True),
            CommutativeCancellation(),
            Optimize1qGates(),
        ])
        return pm

    @staticmethod
    def no_no_al() -> PassManager:
        # Minimal transformation (near-identity transpile), mostly for baseline.
        pm = PassManager([])
        return pm

    @staticmethod
    def resolve(name: str) -> PassManager:
        name = name.strip().lower()
        if name in {"ol0", "opt0"}:
            return PassCombos.preset(0)
        if name in {"ol1", "opt1"}:
            return PassCombos.preset(1)
        if name in {"ol2", "opt2"}:
            return PassCombos.preset(2)
        if name in {"ol3", "opt3"}:
            return PassCombos.preset(3)
        if name == "na-sa-al":
            return PassCombos.na_sa_al()
        if name == "sa-sa-al":
            return PassCombos.sa_sa_al()
        if name == "no-no-al":
            return PassCombos.no_no_al()
        raise ValueError(f"Unknown combo: {name}")

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

# -----------------------------------------------------------------------------
# Experiment Runner
# -----------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    circuits: List[str]
    depths: List[int]
    combos: List[str]
    shots: int = 2000
    backend: str = "aer"  # future: ibmq backend name
    seed: Optional[int] = 1


class CompassRepro:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.backend = Aer.get_backend("qasm_simulator") if cfg.backend == "aer" else Aer.get_backend("qasm_simulator")

    # ------------------ circuit generation ------------------
    def _gen_circuit(self, tag: str, depth: int) -> QuantumCircuit:
        tag = tag.lower()
        if tag == "bv":
            n = max(3, depth + 2)  # tie depth arg to size for quick scans
            return CircuitFactory.bv(n)
        if tag == "qaoa":
            n = max(4, depth * 2)
            return CircuitFactory.qaoa_ring(n=n, p=depth)
        if tag == "pea":
            n = max(3, depth + 2)
            return CircuitFactory.pea_toy(n=n-1, k=1)
        raise ValueError(f"Unknown circuit tag: {tag}")

    # ------------------ run one combo ------------------
    def _run_combo(self, qc: QuantumCircuit, combo_name: str, shots: int) -> Tuple[Dict[str, int], Dict[str, int]]:
        pm = PassCombos.resolve(combo_name)
        tqc = pm.run(qc)
        # Original distribution
        job = self.backend.run(tqc, shots=shots)
        counts_orig = job.result().get_counts()
        # Dummy distribution
        dq = CliffordDummy.to_clifford_dummy(tqc, seed=42)
        job2 = self.backend.run(dq, shots=shots)
        counts_dummy = job2.result().get_counts()
        return counts_orig, counts_dummy

    # ------------------ full sweep ------------------
    def run(self) -> pd.DataFrame:
        rows: List[Dict] = []
        for circ in self.cfg.circuits:
            for depth in self.cfg.depths:
                qc = self._gen_circuit(circ, depth)
                for combo in tqdm(self.cfg.combos, desc=f"{circ}[p={depth}]", leave=False):
                    c_orig, c_dummy = self._run_combo(qc, combo, self.cfg.shots)
                    p = _counts_to_prob(c_orig, self.cfg.shots)
                    q = _counts_to_prob(c_dummy, self.cfg.shots)
                    tvd = total_variation_distance(p, q)
                    ist = inference_strength(p, q)
                    fid = fidelity_like(p, q)
                    rows.append(asdict(Metrics(combo=combo,
                                               circuit=circ,
                                               nqubits=qc.num_qubits,
                                               depth_p=depth,
                                               shots=self.cfg.shots,
                                               tvd=tvd,
                                               ist=ist,
                                               fidelity_proxy=fid)))
        return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# (Optional) Simple plotting helper (kept minimal)
# -----------------------------------------------------------------------------

def plot_results(df: pd.DataFrame, y: str = "fidelity_proxy") -> None:
    import matplotlib.pyplot as plt
    for circ in sorted(df.circuit.unique()):
        sub = df[df.circuit == circ]
        for combo in sorted(sub.combo.unique()):
            sub2 = sub[sub.combo == combo]
            xs = sub2.depth_p.values
            ys = sub2[y].values
            plt.plot(xs, ys, marker="o", label=f"{circ}-{combo}")
    plt.xlabel("depth p")
    plt.ylabel(y)
    plt.title(f"COMPASS reproduction metric: {y}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="COMPASS reproduction scaffold")
    p.add_argument("--circuits", nargs="+", default=["bv", "qaoa"], help="bv qaoa pea")
    p.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3], help="list of depths/p")
    p.add_argument("--combos", nargs="+", default=["na-sa-al", "sa-sa-al", "ol1", "ol3"], help="pass combos")
    p.add_argument("--shots", type=int, default=2000)
    p.add_argument("--backend", type=str, default="aer")
    p.add_argument("--save", type=str, default="")
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = ExperimentConfig(
        circuits=args.circuits,
        depths=args.depths,
        combos=args.combos,
        shots=args.shots,
        backend=args.backend,
    )
    runner = CompassRepro(cfg)
    df = runner.run()
    if args.save:
        df.to_csv(args.save, index=False)
        print(f"Saved: {args.save}")
    if args.plot:
        plot_results(df)
    else:
        print(df.head())


if __name__ == "__main__":
    main()
