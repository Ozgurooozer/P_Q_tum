"""
qse/core/qse_engine.py
======================
Shared circuit runner and formula library for all QSE experiments.

All theorems implemented here:
  T1   P(VNE=v|h,N)  Multinomial closed form
  T6   VNE(θ) = N·H_binary(sin²(θ/2))
  T7   E[VNE_AB] = h_A·(N−h_B)/N  Hypergeometric
  T10  VNE = Σᵢ H_bin(sin²_Ai)·[Bᵢ ∉ X-eigenspace]
  VK   Vandermonde: T1 = T7 marginal
"""

from __future__ import annotations
import numpy as np
from math import comb, factorial, log2
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, entropy, DensityMatrix


# ─── Constants ────────────────────────────────────────────────────────────────

CLIFFORD_GATES = ('H', 'X', 'Y', 'Z', 'S', 'Sdg', 'T', 'Tdg')


# ─── Circuit runner ───────────────────────────────────────────────────────────

def run_clifford_bridge(
    gates_a: list[str],
    gates_b: list[str],
    bridge: list[tuple] | None = None,
) -> Statevector:
    """
    2N-qubit Clifford bridge circuit.

    Args
    ----
    gates_a  : N gate labels for Wave A (qubits 0..N-1).
    gates_b  : N gate labels for Wave B (qubits N..2N-1).
    bridge   : list of (ctrl, tgt) pairs. Default: N parallel CNOTs.

    Returns
    -------
    Statevector of the full 2N-qubit system.
    """
    N = len(gates_a)
    assert len(gates_b) == N

    qc = QuantumCircuit(2 * N)
    _apply = {
        'H': qc.h, 'X': qc.x, 'Y': qc.y, 'Z': qc.z,
        'T': qc.t, 'S': qc.s, 'Sdg': qc.sdg, 'Tdg': qc.tdg, 'I': qc.id,
    }
    for i, g in enumerate(gates_a):
        _apply[g.upper()](i)
    for i, g in enumerate(gates_b):
        _apply[g.upper()](i + N)

    if bridge is None:
        bridge = [(i, i + N) for i in range(N)]
    for ctrl, tgt in bridge:
        qc.cx(ctrl, tgt)

    return Statevector(qc)


def run_rx_bridge(N: int, theta: float) -> Statevector:
    """Rx(θ)⊗2N + N parallel CNOTs — the T6 circuit."""
    qc = QuantumCircuit(2 * N)
    for q in range(2 * N):
        qc.rx(theta, q)
    for i in range(N):
        qc.cx(i, i + N)
    return Statevector(qc)


# ─── Metrics ──────────────────────────────────────────────────────────────────

def vne(sv: Statevector, N: int) -> float:
    """Von Neumann entropy of Wave A (trace out Wave B)."""
    rho = partial_trace(sv, list(range(N, 2 * N)))
    return float(entropy(rho, base=2))


def integer_deviation(sv: Statevector, N: int) -> float:
    """|VNE − round(VNE)| — zero iff circuit is Clifford."""
    v = vne(sv, N)
    return abs(v - round(v))


def pair_independence_error(sv: Statevector, N: int) -> float:
    """
    |VNE_total − Σᵢ VNE_pair_i|.
    Zero for parallel CNOT Clifford circuits (Lemma 2.2).
    """
    total = vne(sv, N)
    pair_sum = 0.0
    for i in range(N):
        keep = [j for j in range(2 * N) if j != i and j != i + N]
        rho_p = partial_trace(sv, keep)
        rho_a = partial_trace(rho_p, [1])
        pair_sum += float(entropy(rho_a, base=2))
    return abs(total - pair_sum)


def magic_pr(sv: Statevector, N: int) -> float:
    """Participation ratio of Wave A spectrum (magic proxy)."""
    rho = partial_trace(sv, list(range(N, 2 * N)))
    evals = np.real(np.linalg.eigvalsh(DensityMatrix(rho).data))
    evals = evals[evals > 1e-12]
    return float(1.0 / np.sum(evals ** 2)) if len(evals) > 0 else 1.0


# ─── Theorem formulas ─────────────────────────────────────────────────────────

def h_binary(p: float) -> float:
    """Binary entropy function H(p) = −p·log₂p − (1−p)·log₂(1−p)."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * log2(p) - (1.0 - p) * log2(1.0 - p)


# T1 ── Multinomial closed form ─────────────────────────────────────────────

def T1_prob(N: int, h: int, v: int) -> float:
    """
    Theorem 1: P(VNE=v | H=h, N).

    Parameters
    ----------
    N : number of qubit pairs
    h : total Hadamard count (across 2N positions)
    v : target VNE value
    """
    total = 0
    for p in range(N + 1):
        r = h - v - 2 * p
        s = N - p - v - r
        if r < 0 or s < 0 or p + v + r + s != N:
            continue
        try:
            total += factorial(N) // (
                factorial(p) * factorial(v) * factorial(r) * factorial(s)
            )
        except ValueError:
            continue
    denom = comb(2 * N, h)
    return total / denom if denom > 0 else 0.0


def T1_mean(N: int, h: int) -> float:
    """E[VNE] under Theorem 1."""
    return sum(v * T1_prob(N, h, v) for v in range(N + 1))


def T1_matrix(N: int) -> np.ndarray:
    """Full P(VNE=v|h,N) matrix, shape (2N+1, N+1)."""
    mat = np.zeros((2 * N + 1, N + 1))
    for h in range(2 * N + 1):
        for v in range(N + 1):
            mat[h, v] = T1_prob(N, h, v)
    return mat


# T6 ── Rx(θ) binary entropy ──────────────────────────────────────────────────

def T6_vne(N: int, theta: float) -> float:
    """
    Theorem 6: VNE(θ) = N · H_binary(sin²(θ/2))
    for Rx(θ)⊗2N + N parallel CNOTs.
    """
    return N * h_binary(np.sin(theta / 2) ** 2)


# T7 ── Hypergeometric (3N independent) ────────────────────────────────────────

def T7_prob(N: int, h_A: int, h_B: int, v: int) -> float:
    """
    Theorem 7: P(VNE_AB=v | h_A, h_B, N).
    Hypergeometric distribution.
    """
    k = h_A - v  # overlap count
    if k < 0 or k > min(h_A, h_B):
        return 0.0
    if (h_B - k) < 0 or (h_B - k) > (N - h_A):
        return 0.0
    denom = comb(N, h_B)
    return comb(h_A, k) * comb(N - h_A, h_B - k) / denom if denom > 0 else 0.0


def T7_mean(N: int, h_A: int, h_B: int) -> float:
    """E[VNE_AB] = h_A · (N − h_B) / N."""
    return h_A * (N - h_B) / N


# T10 ── Unified formula ───────────────────────────────────────────────────────

def T10_vne(p_list: list[float], q_list: list[float]) -> float:
    """
    Theorem 10 (Unified VNE):
    VNE = Σᵢ H_binary(pᵢ · (1 − qᵢ))

    where pᵢ = P(A_i in superposition), qᵢ = P(B_i in X-eigenspace).

    Special cases
    -------------
    T1 : pᵢ ∈ {0,1}, qᵢ ∈ {0,1}   (Clifford, discrete)
    T6 : pᵢ = sin²(θ/2), qᵢ = 0   (Rx rotation, qᵢ=0 → no X-eigenstate)
    T7 : E[pᵢ]=hA/N, E[qᵢ]=hB/N   (independent H placement)
    """
    return sum(h_binary(p * (1.0 - q)) for p, q in zip(p_list, q_list))


# VK ── Vandermonde identity ───────────────────────────────────────────────────

def vandermonde_check(N: int) -> float:
    """
    Verify: T1(v|h,N) = Σ_{hA+hB=h} P(hA,hB)·T7(v|hA,hB,N).
    Returns maximum absolute error across all (h, v).
    """
    max_err = 0.0
    for h in range(2 * N + 1):
        denom = comb(2 * N, h)
        for v in range(N + 1):
            t1 = T1_prob(N, h, v)
            t7_marg = 0.0
            for hA in range(min(h, N) + 1):
                hB = h - hA
                if hB < 0 or hB > N:
                    continue
                weight = comb(N, hA) * comb(N, hB) / denom
                t7_marg += weight * T7_prob(N, hA, hB, v)
            max_err = max(max_err, abs(t1 - t7_marg))
    return max_err


# ─── Helpers ──────────────────────────────────────────────────────────────────

def random_gates(n: int, gate_set: tuple, rng) -> list[str]:
    return [rng.choice(gate_set) for _ in range(n)]


def h_mask_gates(N: int, h_positions: set, rng, non_h=('X', 'Y', 'Z')) -> tuple:
    """Return (gates_a, gates_b) with H at h_positions ∈ {0..2N-1}."""
    gates = [
        'H' if i in h_positions else rng.choice(list(non_h))
        for i in range(2 * N)
    ]
    return gates[:N], gates[N:]


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("QSE Engine — self-test\n")

    # T1
    print("T1: P(VNE=v | h=4, N=4)")
    for v in range(5):
        print(f"  v={v}: {T1_prob(4, 4, v):.6f}")

    # T6
    import numpy as np
    print(f"\nT6: VNE(π/2, N=4) = {T6_vne(4, np.pi/2):.6f}  (expected 4.0)")

    # T7
    print(f"\nT7: E[VNE_AB | hA=2, hB=2, N=4] = {T7_mean(4, 2, 2):.6f}  (expected 1.0)")

    # T10
    print(f"\nT10: unified_VNE([1,1,0,0],[0,0,0,0]) = {T10_vne([1,1,0,0],[0,0,0,0]):.6f}  (expected 2.0)")

    # VK
    for N in [2, 3, 4]:
        err = vandermonde_check(N)
        print(f"VK: N={N}  max_err={err:.2e}  {'✓' if err < 1e-12 else '✗'}")

    # Circuit
    rng = np.random.default_rng(42)
    ga, gb = h_mask_gates(4, {0, 1, 4, 5}, rng)
    sv = run_clifford_bridge(ga, gb)
    print(f"\nSample circuit VNE = {vne(sv, 4):.4f}")
    print(f"Integer deviation  = {integer_deviation(sv, 4):.2e}")
