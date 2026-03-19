# QSE — Quantum Simulation Engine

**Entanglement Distribution in Clifford Bridge Circuits**  
*Experimental-Theoretical Investigation | Independent Research, 2025*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

---

## Overview

This repository characterizes von Neumann entropy (VNE) distribution in Clifford bridge circuits. Starting from a local 8-qubit simulator, it develops a unified analytic framework across discrete (H gates), continuous (Rx rotations), and multi-wave (3N) regimes.

---

## Unified VNE Theorem (T10)

For a 2N-qubit Clifford bridge circuit (Wave A → CNOT bridge → Wave B):

```
VNE = Σᵢ H_binary(sin²(θ_Ai/2)) · [B_i ∉ X-eigenspace]
```

- `H_binary(p) = −p·log₂(p) − (1−p)·log₂(1−p)`
- `θ_Ai` = Bloch polar angle of qubit i in Wave A
- `[B_i ∉ X-eigenspace]` = 1 unless B_i ∈ {|+⟩, |−⟩}

**Physical origin:** `CNOT(|ψ_A⟩, |+⟩) = |ψ_A⟩ ⊗ |+⟩` — X-eigenstates absorb CNOT without entanglement.

---

## All Theorems

| # | Statement | Status |
|---|-----------|--------|
| T1 | `P(VNE=v\|h,N) = (1/C(2N,h))·Σ_p[N!/(p!v!(h-v-2p)!(N+p-h)!)]` | ✓ Proved |
| T2 | T-gate injection: linear for k<4, periodic at T⁴=i·I | ✓ Numerical |
| T3 | H→T→CNOT = magic; T→H→CNOT = zero magic | ✓ Proved |
| T4 | 3N: VNE_AB independent of h_C | ✓ Proved |
| T5 | T on Wave A → both AB and BC Clifford-protected | ✓ Proved |
| T6 | `VNE(θ) = N·H_binary(sin²(θ/2))` for Rx(θ)⊗2N + CNOT | ✓ Proved R²=1.0 |
| T7 | `E[VNE_AB] = h_A·(N−h_B)/N` — Hypergeometric | ✓ Proved |
| T8 | T on A → VNE_AB and VNE_BC both preserved | ✓ Proved |
| T9 | `S(ρ_BC) = S(ρ_A)` — Schmidt symmetry for pure states | ✓ Proved |
| T10 | Unified: `VNE=Σᵢ H_bin(sin²_Ai)·[Bᵢ∉X-eigenspace]` | ✓ err<1e-15 |
| VK | Vandermonde: T1 = T7 marginal (h_A+h_B=h fixed) | ✓ err<1e-16 |

---

## Quick Start

```bash
pip install qiskit qiskit-aer numpy matplotlib scipy

python proofs/algebraic_proof.py     # Verify T1
python proofs/b_asymmetry.py         # Verify T10 B-rule
python experiments/qse_v18_unified.py # Full unified test
```

---

## Repository Structure

```
qse/
├── core/qse_engine.py          — base runner, metrics, all formulas
├── experiments/                — v3 through v18, full trail
├── proofs/                     — algebraic proofs T1, T3, T10
└── papers/                     — qse_final_en.docx + qse_final_tr.docx
```

---

## Citation

```
QSE Project (2025). A Unified Mechanism for Entanglement Distribution
in Clifford Bridge Circuits. Independent Research Note v18.
```

MIT License.
