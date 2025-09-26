from qiskit import QuantumCircuit
from typing import Optional, Iterable


# -----------------------------------------------------------------------------
# Circuit Factory
# -----------------------------------------------------------------------------
class CircuitFactory:
    """Factory methods for benchmark circuits used in COMPASS paper.

    Implements: BV, QAOA (toy MaxCut on ring graph), and PEA (toy).
    For exact parity with the paper, replace toy versions with exact specs.
    """

    @staticmethod
    def bv_ancilla(n: int, s: Optional[Iterable[int]] = None, include_measure: bool = True) -> QuantumCircuit:
        """
        표준 교과서형 BV (ancilla 사용, n개의 CNOT).
        Depth ~= O(n)
        """
        if s is None:
            s = [1] * n
        s = list(s)
        assert len(s) == n

        qc = QuantumCircuit(n + 1, n if include_measure else 0)
        qc.x(n)                  # ancilla |1>
        qc.h(range(n + 1))
        for i, bit in enumerate(s):
            if bit:
                qc.cx(i, n)
        qc.h(range(n))
        if include_measure:
            qc.measure(range(n), range(n))
        return qc
    
    @staticmethod
    def bv_phase(n: int, s: Optional[Iterable[int]] = None, include_measure: bool = True) -> QuantumCircuit:
        """
        Ancilla-free BV via phase-kickback.
        Depth ~= 3 (H -> parallel Z -> H). Z는 서로 다른 큐빗에 병렬 적용 가능.
        s: 숨은 비트열(길이 n). None이면 모두 1로 가정.
        """
        if s is None:
            s = [1] * n
        s = list(s)
        assert len(s) == n

        qc = QuantumCircuit(n, n if include_measure else 0)
        qc.h(range(n))                             # layer 1
        for i, bit in enumerate(s):                # layer 2 (병렬 가능)
            if bit:
                qc.z(i)
        qc.h(range(n))                             # layer 3
        if include_measure:
            qc.measure(range(n), range(n))         # (선택) 마지막 measure 레이어
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
        import math
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

    @staticmethod
    def adder(n: int) -> QuantumCircuit:
        """Toy ripple-carry adder on 2n+1 qubits."""
        qc = QuantumCircuit(2 * n + 1, 2 * n)
        # Prepare inputs in superposition
        qc.h(range(2 * n))
        # Simple toy adder: just CNOT chains as placeholder
        for i in range(n):
            qc.cx(i, n + i)
        qc.measure(range(2 * n), range(2 * n))
        return qc

    @staticmethod
    def ghz(n: int) -> QuantumCircuit:
        """GHZ state on n qubits."""
        qc = QuantumCircuit(n, n)
        qc.h(0)
        for i in range(1, n):
            qc.cx(0, i)
        qc.measure(range(n), range(n))
        return qc

    @staticmethod
    def cnx(n: int) -> QuantumCircuit:
        """Toy CNX circuit with n qubits."""
        qc = QuantumCircuit(n, n)
        # Chain of CNOTs from qubit 0 to others
        for i in range(1, n):
            qc.cx(0, i)
        qc.measure(range(n), range(n))
        return qc

    @staticmethod
    def cnx_dirty(n: int) -> QuantumCircuit:
        """Toy CNX dirty ancilla circuit with n qubits."""
        qc = QuantumCircuit(n, n)
        # Use qubit 0 as control, others as targets with some X gates to simulate dirty ancilla
        qc.x(1)
        for i in range(2, n):
            qc.cx(0, i)
        qc.measure(range(n), range(n))
        return qc

    @staticmethod
    def cnx_h(n: int) -> QuantumCircuit:
        """Toy CNX circuit with Hadamard gates."""
        qc = QuantumCircuit(n, n)
        qc.h(0)
        for i in range(1, n):
            qc.cx(0, i)
        qc.h(0)
        qc.measure(range(n), range(n))
        return qc

    @staticmethod
    def cld(n: int) -> QuantumCircuit:
        """Toy CLD circuit with n qubits."""
        qc = QuantumCircuit(n, n)
        # Apply layers of controlled rotations
        for i in range(n - 1):
            qc.crz(0.5, i, i + 1)
        qc.measure(range(n), range(n))
        return qc

    @staticmethod
    def vqe_toy(n: int) -> QuantumCircuit:
        """Toy VQE ansatz on n qubits."""
        qc = QuantumCircuit(n, n)
        for i in range(n):
            qc.ry(0.5, i)
        for i in range(n - 1):
            qc.cz(i, i + 1)
        qc.measure(range(n), range(n))
        return qc

    @staticmethod
    def bc(n: int) -> QuantumCircuit:
        """Toy BC (Bernstein Circuit) with n qubits."""
        qc = QuantumCircuit(n, n)
        qc.h(range(n))
        for i in range(n - 1):
            qc.cx(i, i + 1)
        qc.measure(range(n), range(n))
        return qc

    @staticmethod
    def pc(n: int) -> QuantumCircuit:
        """Toy PC (Parity Circuit) with n qubits."""
        qc = QuantumCircuit(n, n)
        qc.h(range(n))
        for i in range(n - 1):
            qc.cx(i, i + 1)
        qc.measure(range(n), range(n))
        return qc

    @staticmethod
    def mb(n: int) -> QuantumCircuit:
        """Toy MB (Measurement-Based) circuit with n qubits."""
        qc = QuantumCircuit(n, n)
        qc.h(range(n))
        for i in range(n - 1):
            qc.cz(i, i + 1)
        qc.measure(range(n), range(n))
        return qc

    @staticmethod
    def hs(n: int) -> QuantumCircuit:
        """Toy HS (Hadamard-Swap) circuit with n qubits."""
        qc = QuantumCircuit(n, n)
        qc.h(range(n))
        for i in range(0, n - 1, 2):
            qc.swap(i, i + 1)
        qc.measure(range(n), range(n))
        return qc


if __name__ == "__main__":
    import math

    qc_phase = CircuitFactory.bv_phase(20, include_measure=False)
    print("bv20 phase depth (no measure):", qc_phase.depth())      # ~3
    print("bv20 phase gates (no measure):", qc_phase.count_ops())  # H:40, Z:20 
    print(qc_phase)

    qc_anc = CircuitFactory.bv_ancilla(20, include_measure=False)
    print("bv20 ancilla depth (no measure):", qc_anc.depth())      # ~22~23 근처
    print(qc_anc)
    # Example usage
    # bv_circuit = CircuitFactory.bv(20)
    # print("Bernstein-Vazirani Circuit (n=20):")
    # print(bv_circuit.depth())
    # print(bv_circuit)

    # qaoa_circuit = CircuitFactory.qaoa_ring(4, 2)
    # print("\nQAOA Circuit (n=4, p=2):")
    # print(qaoa_circuit.depth())
    # print(qaoa_circuit)

    # pea_circuit = CircuitFactory.pea_toy(3, 1)
    # print("\nPhase Estimation Circuit (n=3, k=1):")
    # print(pea_circuit)

    # adder_circuit = CircuitFactory.adder(2)
    # print("\nAdder Circuit (n=2):")
    # print(adder_circuit)

    # ghz_circuit = CircuitFactory.ghz(3)
    # print("\nGHZ Circuit (n=3):")
    # print(ghz_circuit)

    # cnx_circuit = CircuitFactory.cnx(3)
    # print("\nCNX Circuit (n=3):")
    # print(cnx_circuit)

    # cnx_dirty_circuit = CircuitFactory.cnx_dirty(4)
    # print("\nCNX Dirty Circuit (n=4):")
    # print(cnx_dirty_circuit)

    # cnx_h_circuit = CircuitFactory.cnx_h(3)
    # print("\nCNX with Hadamard Circuit (n=3):")
    # print(cnx_h_circuit)

    # cld_circuit = CircuitFactory.cld(3)
    # print("\nCLD Circuit (n=3):")
    # print(cld_circuit)

    # vqe_circuit = CircuitFactory.vqe_toy(3)
    # print("\nVQE Toy Circuit (n=3):")
    # print(vqe_circuit)

    # bc_circuit = CircuitFactory.bc(3)
    # print("\nBC Circuit (n=3):")
    # print(bc_circuit)

    # pc_circuit = CircuitFactory.pc(3)
    # print("\nPC Circuit (n=3):")
    # print(pc_circuit)

    # mb_circuit = CircuitFactory.mb(3)
    # print("\nMB Circuit (n=3):")
    # print(mb_circuit)

    # hs_circuit = CircuitFactory.hs(4)
    # print("\nHS Circuit (n=4):")
    # print(hs_circuit)