from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.passes import (Optimize1qGates, CommutativeCancellation, ConsolidateBlocks, BasisTranslator,)

from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel


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