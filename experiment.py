from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict
from qiskit import QuantumCircuit
from circuits import CircuitFactory
from metric import Metrics, _counts_to_prob, total_variation_distance, inference_strength, fidelity_like
from pass_combinations import PassCombos
from clifford_dummy import CliffordDummy
from qiskit_aer import Aer
from tqdm import tqdm
import pandas as pd

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
            return CircuitFactory.bv_phase(n)
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