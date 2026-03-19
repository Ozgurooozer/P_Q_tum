"""
QSE v21 — T10: B Koordinatlarının VNE Büyüklüğüne Etkisi

Soru: VNE_i = H_binary(sin²_Ai) · [B_i ∉ X-eigenspace] doğru mu?
      Yoksa B'nin Bloch koordinatları (Bx) VNE büyüklüğünü sürekli etkiliyor mu?

Türetim (tek çift):
  A = Rx(θ_A)|0⟩, B = |ψ_B⟩ ile Bloch vektörü (Bx, By, Bz)
  CNOT sonrası ρ_A özdeğerleri:
    λ± = 1/2 ± (1/2)·√(cos²θ_A + sin²θ_A · Bx²)
  →  VNE_i = H_binary( (1 + √(cos²θ_A + sin²θ_A · Bx²)) / 2 )

Kurallar:
  - Önce veri, sonra teori
  - Her adım tek sorulu
  - Başarısızlık da veridir

Kurulum: pip install qiskit qiskit-aer numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from math import log2
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, entropy, SparsePauliOp

RNG = np.random.default_rng(42)

def h_bin(p):
    if p <= 0 or p >= 1: return 0.0
    return -p*log2(p) - (1-p)*log2(1-p)

# ─── Tek çift VNE ────────────────────────────────────────────

def vne_single(theta_A: float, B_circuit_fn) -> float:
    """
    Tek çift: A = Rx(theta_A)|0⟩, B = B_circuit_fn uygulanmış |0⟩
    CNOT(0→1), sonra qubit 0'ın VNE'si.
    """
    qc = QuantumCircuit(2)
    qc.rx(theta_A, 0)
    B_circuit_fn(qc)
    qc.cx(0, 1)
    sv = Statevector(qc)
    rho = partial_trace(sv, [1])
    return float(entropy(rho, base=2))

def bx_of(B_circuit_fn) -> float:
    """B durumunun Bloch x-bileşeni = ⟨X⟩"""
    qc = QuantumCircuit(2)  # 2 qubit oluştur (lambda'lar qubit 1'e erişiyor)
    B_circuit_fn(qc)
    sv = Statevector(qc)
    # Sadece qubit 1'in Bloch vektörü
    rho = partial_trace(sv, [0])
    return float(np.real(rho.expectation_value(SparsePauliOp('X'))))

def formula_vne(theta_A: float, Bx: float) -> float:
    """Türetilen formül: H_binary((1 + √(cos²θ + sin²θ·Bx²))/2)"""
    inner = np.cos(theta_A)**2 + np.sin(theta_A)**2 * Bx**2
    lam_plus = (1 + np.sqrt(inner)) / 2
    return h_bin(lam_plus)

# ══════════════════════════════════════════════════════════════
# TEST 1 — Bilinen durumlar: formül eskisiyle uyuşuyor mu?
# ══════════════════════════════════════════════════════════════

print("=" * 65)
print("TEST 1 — Bilinen durumlar")
print("=" * 65)
print(f"{'A (θ)':>12} {'B':>12} | {'Obs':>8} | {'Formül':>8} | {'Err':>8}")
print("-" * 55)

cases_known = [
    (np.pi/2, lambda qc: None,          "|0⟩"),
    (np.pi/2, lambda qc: qc.x(1),       "|1⟩"),
    (np.pi/2, lambda qc: qc.h(1),       "|+⟩"),
    (np.pi/2, lambda qc: (qc.x(1), qc.h(1)), "|-⟩"),
    (np.pi/4, lambda qc: None,          "|0⟩"),
    (np.pi/4, lambda qc: qc.h(1),       "|+⟩"),
    (0,       lambda qc: None,          "|0⟩"),
]

for theta_A, b_fn, b_label in cases_known:
    obs = vne_single(theta_A, b_fn)
    Bx  = bx_of(b_fn)
    pred = formula_vne(theta_A, Bx)
    err = abs(obs - pred)
    print(f"Rx({theta_A/np.pi:.2f}π) {b_label:>10} | {obs:>8.5f} | {pred:>8.5f} | {err:>8.2e} {'✓' if err<1e-10 else '✗'}")

# ══════════════════════════════════════════════════════════════
# TEST 2 — Bx sürekli değişiyor: VNE'de sürekli geçiş var mı?
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("TEST 2 — Bx sürekli değişiyor (B = Ry(φ)|0⟩, Bx = sin(φ))")
print("=" * 65)

theta_A_test = np.pi / 2  # A = H|0⟩
phi_vals = np.linspace(0, np.pi, 18)

print(f"{'φ/π':>6} | {'Bx':>7} | {'Obs VNE':>9} | {'Formül':>9} | {'T10_old':>9} | {'Err':>8}")
print("-" * 65)

obs_list, pred_list, t10_old_list = [], [], []
for phi in phi_vals:
    b_fn = lambda qc, p=phi: qc.ry(p, 1)
    obs  = vne_single(theta_A_test, b_fn)
    Bx   = np.sin(phi)          # Ry(φ)|0⟩ → Bloch = (sin φ, 0, cos φ)
    pred = formula_vne(theta_A_test, Bx)
    t10_old = h_bin(np.sin(theta_A_test/2)**2) * (1 if abs(Bx) < 0.999 else 0)
    err  = abs(obs - pred)
    obs_list.append(obs); pred_list.append(pred); t10_old_list.append(t10_old)
    flag = "✓" if err < 1e-10 else "✗"
    print(f"{phi/np.pi:>6.3f} | {Bx:>7.4f} | {obs:>9.5f} | {pred:>9.5f} | {t10_old:>9.5f} | {err:>8.2e} {flag}")

max_err_new = max(abs(o-p) for o,p in zip(obs_list, pred_list))
max_err_old = max(abs(o-p) for o,p in zip(obs_list, t10_old_list))
print(f"\nYeni formül max err: {max_err_new:.2e}")
print(f"Eski T10 max err:    {max_err_old:.4f}")
print(f"Sonuç: VNE {'SÜREKLI değişiyor' if max_err_old > 0.05 else 'binary'}  — Bx etkisi {'VAR ✓' if max_err_old > 0.05 else 'yok'}")

# ══════════════════════════════════════════════════════════════
# TEST 3 — By ve Bz etkisi: sadece Bx mi önemli?
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("TEST 3 — By ve Bz etkisi (Bx = 0, By ve Bz değişiyor)")
print("=" * 65)
print("Hipotez: VNE sadece Bx'e bağlı, By ve Bz önemsiz")
print(f"{'B durumu':>20} | {'Bx':>6} | {'By':>6} | {'Bz':>6} | {'Obs':>8} | {'Pred':>8} | {'Err':>8}")
print("-" * 75)

# Bx=0 ama farklı By, Bz: |0⟩, |1⟩, |+i⟩, |-i⟩, Rz(π/4)|0⟩ vb.
cases_byz = [
    ("|0⟩",       lambda qc: None,                    0, 0, 1),
    ("|1⟩",       lambda qc: qc.x(1),                 0, 0,-1),
    ("|+i⟩",      lambda qc: (qc.h(1), qc.s(1)),      0, 1, 0),
    ("|-i⟩",      lambda qc: (qc.h(1), qc.sdg(1)),    0,-1, 0),
    ("Rz(π/4)|+⟩", lambda qc: (qc.h(1), qc.rz(np.pi/4, 1)), None, None, None),
    ("Ry(π/6)|0⟩", lambda qc: qc.ry(np.pi/6, 1),      None, None, None),
]

theta_A_t3 = np.pi / 2
all_ok_t3 = True
for label, b_fn, bx0, by0, bz0 in cases_byz:
    obs = vne_single(theta_A_t3, b_fn)
    # Gerçek Bloch koordinatları
    qc_b = QuantumCircuit(2); b_fn(qc_b); sv_b = Statevector(qc_b)
    rho_b = partial_trace(sv_b, [0])
    Bx = float(np.real(rho_b.expectation_value(SparsePauliOp('X'))))
    By = float(np.real(rho_b.expectation_value(SparsePauliOp('Y'))))
    Bz = float(np.real(rho_b.expectation_value(SparsePauliOp('Z'))))
    pred = formula_vne(theta_A_t3, Bx)
    err = abs(obs - pred)
    ok = err < 1e-10
    if not ok: all_ok_t3 = False
    print(f"{label:>20} | {Bx:>6.3f} | {By:>6.3f} | {Bz:>6.3f} | {obs:>8.5f} | {pred:>8.5f} | {err:>8.2e} {'✓' if ok else '✗'}")

print(f"\nSadece Bx önemli, By ve Bz önemsiz: {'DOĞRULANDI ✓' if all_ok_t3 else 'BAŞARISIZ ✗'}")

# ══════════════════════════════════════════════════════════════
# TEST 4 — Genel (θ_A, Bx) ızgara taraması
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("TEST 4 — Tam ızgara: tüm (θ_A, Bx) kombinasyonları")
print("=" * 65)

thetas = np.linspace(0, np.pi, 20)
bx_vals = np.linspace(-1, 1, 21)

max_err_grid = 0.0
n_tested = 0

for theta_A in thetas:
    for Bx_target in bx_vals:
        # B = Ry(φ)|0⟩ ile Bx = sin(φ)
        phi = np.arcsin(np.clip(Bx_target, -1, 1))
        b_fn = lambda qc, p=phi: qc.ry(p, 1)
        obs  = vne_single(theta_A, b_fn)
        Bx   = float(np.sin(phi))
        pred = formula_vne(theta_A, Bx)
        err  = abs(obs - pred)
        max_err_grid = max(max_err_grid, err)
        n_tested += 1

print(f"Test edilen kombinasyon: {n_tested}")
print(f"Max hata: {max_err_grid:.2e}")
print(f"Formül {'DOĞRULANDI ✓' if max_err_grid < 1e-9 else 'BAŞARISIZ ✗'}")

# ══════════════════════════════════════════════════════════════
# GRAFİK
# ══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#0f0f0f')
fig.suptitle(
    'v21: T10 Genelleştirme — VNE sürekli Bx\'e bağlı\n'
    'VNE_i = H_binary((1 + √(cos²θ_A + sin²θ_A·Bx²)) / 2)',
    color='white', fontsize=12
)
for ax in axes:
    ax.set_facecolor('#1a1a1a'); ax.tick_params(colors='#888')
    for sp in ax.spines.values(): sp.set_edgecolor('#333')

# Sol: VNE vs Bx (θ_A=π/2 sabit)
bx_fine = np.linspace(-1, 1, 100)
theta_fixed = np.pi/2
pred_curve = [formula_vne(theta_fixed, bx) for bx in bx_fine]
old_curve  = [h_bin(np.sin(theta_fixed/2)**2) if abs(bx) < 0.999 else 0 for bx in bx_fine]

axes[0].plot(bx_fine, pred_curve, '-', color='#EF9F27', lw=2.5, label='Yeni: H_bin((1+√(cos²θ+sin²θBx²))/2)')
axes[0].plot(bx_fine, old_curve,  '--', color='#D85A30', lw=1.5, alpha=0.7, label='Eski T10: binary')
# Gözlem noktaları
phi_obs = np.linspace(0, np.pi, 14)
bx_obs  = np.sin(phi_obs)
vne_obs = [vne_single(theta_fixed, lambda qc,p=p: qc.ry(p,1)) for p in phi_obs]
axes[0].scatter(bx_obs, vne_obs, color='white', s=40, zorder=5, label='Qiskit ölçüm')
axes[0].set_xlabel('Bx (B\'nin X-Bloch bileşeni)', color='#888')
axes[0].set_ylabel('VNE_i', color='#888')
axes[0].set_title('A=H(π/2), B değişiyor\nSürekli VNE vs Bx', color='white')
axes[0].legend(facecolor='#111', labelcolor='white', fontsize=8)
axes[0].axvline(1, color='#1D9E75', lw=1, ls=':', alpha=0.6)
axes[0].axvline(-1, color='#1D9E75', lw=1, ls=':', alpha=0.6)

# Orta: 2D ısı haritası VNE(θ_A, Bx)
theta_g = np.linspace(0, np.pi, 50)
bx_g    = np.linspace(-1, 1, 50)
TT, BB  = np.meshgrid(theta_g, bx_g)
ZZ = np.vectorize(formula_vne)(TT, BB)
im = axes[1].contourf(theta_g/np.pi, bx_g, ZZ, levels=20, cmap='YlOrRd')
axes[1].contour(theta_g/np.pi, bx_g, ZZ, levels=[0.5, 1.0], colors='white', alpha=0.4)
axes[1].set_xlabel('θ_A / π (A\'nın Bloch açısı)', color='#888')
axes[1].set_ylabel('Bx (B\'nin X-bileşeni)', color='#888')
axes[1].set_title('VNE_i(θ_A, Bx)\nGenel 2D harita', color='white')
plt.colorbar(im, ax=axes[1])

# Sağ: Eski T10 ile yeni formülün farkı
ZZ_old = np.where(np.abs(BB) < 0.999,
                  np.vectorize(lambda t: h_bin(np.sin(t/2)**2))(TT), 0)
ZZ_diff = np.abs(ZZ - ZZ_old)
im2 = axes[2].contourf(theta_g/np.pi, bx_g, ZZ_diff, levels=20, cmap='RdYlGn_r')
axes[2].set_xlabel('θ_A / π', color='#888')
axes[2].set_ylabel('Bx', color='#888')
axes[2].set_title('|Yeni − Eski T10|\nKırmızı = büyük fark', color='white')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig('qse_v21_T10_continuous.png', dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
plt.show()

print("\nKaydedildi: qse_v21_T10_continuous.png")
print("\n" + "="*65)
print("ÖZET")
print("="*65)
print(f"""
Eski T10: VNE_i = H_bin(sin²_Ai) · [Bx ≠ ±1]
  → max_err = {max_err_old:.4f}  YANLIŞ

Yeni formül: VNE_i = H_bin((1 + √(cos²θ_A + sin²θ_A·Bx²)) / 2)
  → max_err = {max_err_grid:.2e}  DOĞRU

Fiziksel anlam:
  Bx = ±1 (X-eigenstate) → VNE = 0  (eski kural doğru)
  Bx = 0                  → VNE = H_bin(sin²_Ai)  (eski T10 ile aynı)
  0 < |Bx| < 1            → VNE sürekli ara değerde  (eski T10 YANLIŞ)

Sadece Bx önemli — By ve Bz VNE'yi etkilemiyor.
""")
