import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import cvxpy as cp

# B.1 Generate Data
n = 200
rng = np.random.default_rng(42)
t = np.linspace(0, 1, n)
x_true = np.piecewise(t, [t < 0.3, (t >= 0.3) & (t < 0.7), t >= 0.7],
                      [lambda s: 0.5*np.sin(8*np.pi*s),
                       lambda s: 1.0,
                       lambda s: 0.2*np.cos(6*np.pi*s)])
noise_std = 0.15
y = x_true + noise_std * rng.standard_normal(n)

# B.2 Difference operator D (sparse)
def diff_matrix(n: int) -> sp.csc_matrix:
    data = np.r_[ -np.ones(n-1), np.ones(n-1) ]
    row  = np.r_[ np.arange(n-1), np.arange(n-1) ]
    col  = np.r_[ np.arange(n-1), np.arange(1,n) ]
    return sp.coo_matrix((data, (row, col)), shape=(n-1, n)).tocsc()

D = diff_matrix(n)

# B.3 L2 (Tikhonov) solve: (I + λ D^T D) x = y
lam = 1
I   = sp.eye(n, format='csc')
DTD = (D.T @ D).tocsc()
A   = I + lam * DTD
x_l2 = spla.spsolve(A, y)

# B.4 TV (L1) with CVXPy
x = cp.Variable(n)
obj = cp.Minimize(cp.sum_squares(x - y) + lam * cp.norm1(D @ x))
prob = cp.Problem(obj)
prob.solve(solver="OSQP", eps_abs=1e-6, eps_rel=1e-6, verbose=False)
x_tv = x.value

# B.5 Evaluation & Plot
mse = lambda a,b: np.mean((np.asarray(a)-np.asarray(b))**2)
print("MSE L2:", mse(x_l2, x_true))
print("MSE TV:", mse(x_tv, x_true))

# --- SAVE PLOT TO ./plots ---
os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(8,4))
plt.plot(t, x_true, label='True')
plt.plot(t, y, label='Noisy', alpha=0.6)
plt.plot(t, x_l2, label='L2 (Tikhonov)')
plt.plot(t, x_tv, label='TV (L1)')
plt.legend(); plt.xlabel('t'); plt.ylabel('x')
plt.title('Signal Denoising: True vs Noisy vs L2 vs TV')
plt.tight_layout()

out_path = f"plots/denoising_lambda_{lam:g}.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"[saved] {out_path}")

plt.show()
# plt.close()  # προαιρετικό αν τρέχεις πολλά plots σε loop
