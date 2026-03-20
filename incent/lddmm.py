"""
lddmm.py — LDDMM Diffeomorphic Deformation for INCENT-SE Stage 5
=================================================================
Recovers the *spatial deformation field* between two slices from different
developmental timepoints.

Why diffeomorphic deformation?
-------------------------------
Between timepoints the brain undergoes growth, folding, and regional
expansion.  The transformation is NOT a simple rigid rotation + translation.
We need a *diffeomorphism* φ: ℝ² → ℝ² — a smooth, invertible, continuously
differentiable map that can model realistic tissue deformation.

What is LDDMM?
--------------
Large Deformation Diffeomorphic Metric Mapping (LDDMM) models the
diffeomorphism as the *flow* of a time-varying velocity field:

    dφ_t / dt = v_t(φ_t),  φ_0 = Identity

The endpoint φ = φ_1 is our deformation.  The velocity field v_t lives in
a Reproducing Kernel Hilbert Space (RKHS) with a Gaussian kernel:

    ||v||²_V = Σ_{t} ∫ v_t(x) K^{-1} v_t(x) dx

This RKHS norm penalises fast-changing or spatially rough deformations,
so the recovered φ is guaranteed to be smooth and biologically plausible.

Adaptation for INCENT-SE
------------------------
In STalign's original formulation, the deformation is driven by minimising
the dissimilarity between two density *images*.  We replace that image-loss
with the OT-plan transport loss:

    E_transport(φ) = Σ_{i,j} π_{ij} · ||φ(y_j) - x_i||²

where π is the current transport plan (from the FGW step), x_i are cell
coordinates in sliceA, and y_j are cell coordinates in sliceB.

This means: "deform sliceB's coordinates so that each cell j in B lands
as close as possible to its matched cell i in A, weighted by how strongly
they are matched (π_ij)".

Public API
----------
LDDMMDeformation          — class that holds the velocity field and computes φ
estimate_deformation(     — main function: given π, x_A, y_B → updates φ
    pi, coords_A, coords_B, sigma_v, n_steps, lr, n_iter)
apply_deformation(coords, phi) → deformed coords
deformed_distances(coords_B, phi, normalise=True) → D_B_deformed
"""

import numpy as np
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Gaussian RKHS kernel
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_kernel(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Evaluate the Gaussian kernel matrix K(x, y).

    K[i,j] = exp( -||x_i - y_j||² / (2σ²) )

    This kernel defines the smoothness of the velocity field.
    σ (sigma_v) controls the spatial scale of deformations:
      - Large σ: smooth, global deformations (brain-scale growth).
      - Small σ: fine-grained, local deformations (cell-level adjustment).
    For MERFISH brain, σ ≈ 200–500 μm is a good starting point.

    Parameters
    ----------
    x : (n, 2) float array — first set of points.
    y : (m, 2) float array — second set of points.
    sigma : float — kernel bandwidth.

    Returns
    -------
    K : (n, m) float64 array.
    """
    # ||x_i - y_j||² = ||x_i||² + ||y_j||² - 2 x_i·y_j
    sq_x = (x ** 2).sum(axis=1, keepdims=True)   # (n, 1)
    sq_y = (y ** 2).sum(axis=1, keepdims=True).T  # (1, m)
    D2   = sq_x + sq_y - 2.0 * (x @ y.T)         # (n, m)
    D2   = np.maximum(D2, 0.0)                    # numerical safety
    return np.exp(-D2 / (2.0 * sigma ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: LDDMMDeformation class
# ─────────────────────────────────────────────────────────────────────────────

class LDDMMDeformation:
    """
    Parametric LDDMM deformation defined by momentum vectors at control points.

    We use the *stationary* (time-constant) velocity field approximation for
    efficiency.  The velocity field is parameterised by momentum vectors
    α_j at control points c_j (which we place at the cell locations of sliceB):

        v(x) = Σ_j  K(x, c_j) · α_j

    The deformation is then Euler-integrated:
        φ(y) = y + Σ_{t=0}^{T-1} v_t(y)  ·  (1/T)

    For large deformations more steps T are needed (at the cost of computation).

    Parameters
    ----------
    control_points : (m, 2) float array — positions where momentum is defined.
        Typically set to the cell coordinates of sliceB.
    sigma_v : float — kernel bandwidth (spatial scale of deformation).
    n_steps : int — number of Euler integration steps.
    """

    def __init__(
        self,
        control_points: np.ndarray,
        sigma_v: float,
        n_steps: int = 5,
    ):
        self.control_points = control_points.astype(np.float64)
        self.sigma_v        = sigma_v
        self.n_steps        = n_steps
        m                   = len(control_points)

        # Momentum α: shape (m, 2) — one 2-D vector per control point
        # Initialised to zero (identity deformation at the start)
        self.alpha          = np.zeros((m, 2), dtype=np.float64)

        # Pre-compute kernel matrix between control points (used for RKHS norm)
        self._K_cc = _gaussian_kernel(control_points, control_points, sigma_v)

    def velocity_at(self, query_points: np.ndarray) -> np.ndarray:
        """
        Evaluate velocity field v(x) = Σ_j K(x, c_j) · α_j at query_points.

        Parameters
        ----------
        query_points : (n, 2) float array.

        Returns
        -------
        v : (n, 2) float array — velocity vectors at each query point.
        """
        K  = _gaussian_kernel(query_points, self.control_points, self.sigma_v)
        # K: (n, m),  alpha: (m, 2)  →  K @ alpha: (n, 2)
        return K @ self.alpha

    def apply(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply the deformation φ to a set of coordinates.

        Uses Euler integration with `n_steps` steps:
            y_{t+1} = y_t + v(y_t) / n_steps

        Parameters
        ----------
        coords : (n, 2) float array — points to deform.

        Returns
        -------
        deformed : (n, 2) float array — φ(coords).
        """
        y = coords.astype(np.float64).copy()
        dt = 1.0 / self.n_steps
        for _ in range(self.n_steps):
            v  = self.velocity_at(y)
            y  = y + dt * v
        return y

    def rkhs_norm_squared(self) -> float:
        """
        Compute ||v||²_V = α^T · K_cc · α (the RKHS regularisation term).

        This penalises large deformations — gradient of this w.r.t. α
        is 2 · K_cc · α, which drives α towards 0 (identity deformation).

        Returns
        -------
        float — RKHS norm squared (scalar).
        """
        # (m, 2) · (m, m) · (m, 2) → scalar
        # = Σ_{d=0,1}  alpha[:,d]^T · K_cc · alpha[:,d]
        return float(np.einsum('id,ij,jd->', self.alpha, self._K_cc, self.alpha))

    def rkhs_norm_gradient(self) -> np.ndarray:
        """
        Gradient of ||v||²_V w.r.t. α.

        ∂||v||²_V / ∂α = 2 · K_cc · α

        Returns
        -------
        (m, 2) float64 — gradient w.r.t. momentum α.
        """
        return 2.0 * self._K_cc @ self.alpha


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Transport loss and its gradient w.r.t. α
# ─────────────────────────────────────────────────────────────────────────────

def _transport_loss(
    phi: LDDMMDeformation,
    pi: np.ndarray,
    coords_A: np.ndarray,
    coords_B: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Compute the transport loss E_transport and its gradient w.r.t. α.

    E_transport(φ) = Σ_{i,j} π_{ij} · ||φ(y_j) - x_i||²

    Gradient derivation
    -------------------
    Let y_j^φ = φ(y_j) = y_j + Σ_{t} v_t(y_j^{t-1}) / T   (Euler approximation)

    For the stationary velocity field approximation:
        ∂E / ∂α_k  =  Σ_{i,j} π_{ij} · 2·(y_j^φ - x_i) · K(y_j^φ, c_k)

    In matrix form (using the deformed positions y^φ):
        ∂E / ∂α  =  2 · K(y^φ, c)^T · (Σ_i π_{ij} (y_j^φ - x_i))_{j}

    Parameters
    ----------
    phi : LDDMMDeformation — current deformation.
    pi : (n_A, n_B) float array — current transport plan.
    coords_A : (n_A, 2) float array — cell positions in sliceA.
    coords_B : (n_B, 2) float array — cell positions in sliceB (control points).

    Returns
    -------
    loss : float — E_transport value.
    grad_alpha : (m, 2) float array — ∂E/∂α.
    """
    # Apply current deformation to sliceB coordinates
    y_phi = phi.apply(coords_B)          # (n_B, 2) — deformed positions

    # For each j in B: weighted residual Σ_i π_{ij} (y_j^φ - x_i)
    # = y_j^φ · (Σ_i π_{ij})  -  Σ_i π_{ij} · x_i
    pi_col_sum = pi.sum(axis=0)                       # (n_B,) = Σ_i π_{ij}
    weighted_target = pi.T @ coords_A                 # (n_B, 2) = Σ_i π_{ij} x_i
    residuals = (pi_col_sum[:, None] * y_phi          # Σ_i π_{ij} · y_j^φ
                 - weighted_target)                   # minus Σ_i π_{ij} · x_i

    # Transport loss: Σ_{i,j} π_{ij} ||y_j^φ - x_i||²
    # Expand: Σ_j Σ_i π_{ij} (||y_j^φ||² - 2 y_j^φ·x_i + ||x_i||²)
    loss = 0.0
    for j in range(len(coords_B)):
        diff_j = y_phi[j] - coords_A                  # (n_A, 2)
        loss  += (pi[:, j] * (diff_j ** 2).sum(axis=1)).sum()

    # Gradient w.r.t. α via chain rule through the Euler integration
    # Using the approximation: ∂φ/∂α ≈ K(y_j^φ, control_points)
    K_yB_c   = _gaussian_kernel(y_phi, phi.control_points, phi.sigma_v)  # (n_B, m)
    grad_alpha = 2.0 * K_yB_c.T @ residuals                              # (m, 2)

    return float(loss), grad_alpha


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTION: estimate_deformation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_deformation(
    pi: np.ndarray,
    coords_A: np.ndarray,
    coords_B: np.ndarray,
    sigma_v: float,
    lambda_v: float = 1.0,
    n_steps: int = 5,
    lr: float = 0.01,
    n_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False,
) -> LDDMMDeformation:
    """
    Estimate the LDDMM diffeomorphism φ that maps sliceB's cells onto sliceA.

    This is the Stage 5 Block B update in INCENT-SE.  Given the current
    OT plan π (from the FGW step), we find the smooth deformation field
    that minimises the transport-weighted distance:

        E(φ) = Σ_{i,j} π_{ij} ||φ(y_j) - x_i||²  +  λ_V · ||v||²_V

    The first term pulls the deformed sliceB coordinates towards their matched
    sliceA positions.  The second term (RKHS norm) ensures the deformation
    is smooth and biologically plausible.

    Parameters
    ----------
    pi : (n_A, n_B) float array — OT plan from the FGW step.
        Larger π[i,j] means cell i in A and cell j in B are matched more
        strongly — so their distance after deformation is penalised more.
    coords_A : (n_A, 2) float array — spatial coordinates of cells in sliceA.
    coords_B : (n_B, 2) float array — spatial coordinates of cells in sliceB.
        These are also used as the control points for the velocity field.
    sigma_v : float
        Kernel bandwidth for the RKHS velocity field.
        Good range for MERFISH: 200–1000 μm.
        Larger → smoother but less flexible deformation.
    lambda_v : float, default 1.0
        Weight of the RKHS regularisation term.
        Larger → smaller, smoother deformation (closer to identity).
        Smaller → larger deformations allowed.
        If the two slices are from adjacent timepoints, try 0.5–2.0.
        For widely-separated timepoints, lower to 0.1.
    n_steps : int, default 5
        Euler integration steps for the flow (more = more accurate flow).
    lr : float, default 0.01
        Gradient descent learning rate for the momentum α.
    n_iter : int, default 100
        Maximum number of gradient descent steps.
    tol : float, default 1e-6
        Convergence tolerance on relative loss change.
    verbose : bool, default False
        Print loss at each iteration.

    Returns
    -------
    phi : LDDMMDeformation — the estimated deformation.
        Use phi.apply(coords_B) to get the deformed coordinates.
        Use deformed_distances(coords_B, phi) to get the updated D_B.

    Notes
    -----
    For n_B = 15k cells the gradient involves a (15k × 15k) kernel matrix
    which is memory-intensive.  We subsample control points to a random
    subset of min(n_B, 2000) cells if n_B is large.
    """
    coords_A = coords_A.astype(np.float64)
    coords_B = coords_B.astype(np.float64)

    # ── Subsample control points if n_B is large ───────────────────────────
    n_B      = len(coords_B)
    max_ctrl = 2000
    if n_B > max_ctrl:
        idx_ctrl = np.random.choice(n_B, max_ctrl, replace=False)
        ctrl_pts = coords_B[idx_ctrl]
        if verbose:
            print(f"[lddmm] Subsampled {max_ctrl} control points from {n_B} cells")
    else:
        ctrl_pts = coords_B

    phi = LDDMMDeformation(ctrl_pts, sigma_v=sigma_v, n_steps=n_steps)

    prev_loss = np.inf
    for it in range(n_iter):
        # ── Forward: compute transport loss + RKHS norm ─────────────────
        E_t,  grad_t = _transport_loss(phi, pi, coords_A, coords_B)
        E_v          = phi.rkhs_norm_squared()
        grad_v       = phi.rkhs_norm_gradient()

        total_loss = E_t + lambda_v * E_v
        total_grad = grad_t + lambda_v * grad_v

        if verbose and it % 10 == 0:
            print(f"[lddmm] iter={it:4d}  E_transport={E_t:.4f}  "
                  f"E_rkhs={E_v:.4f}  total={total_loss:.4f}")

        # ── Convergence check ─────────────────────────────────────────────
        rel_change = abs(total_loss - prev_loss) / (abs(prev_loss) + 1e-12)
        if rel_change < tol and it > 5:
            if verbose:
                print(f"[lddmm] Converged at iteration {it}")
            break
        prev_loss = total_loss

        # ── Gradient descent step on α ────────────────────────────────────
        # Simple gradient descent (for production use Adam or L-BFGS)
        phi.alpha -= lr * total_grad

    return phi


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC HELPER: deformed_distances
# ─────────────────────────────────────────────────────────────────────────────

def deformed_distances(
    coords_B: np.ndarray,
    phi: LDDMMDeformation,
    normalise: bool = True,
) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between deformed sliceB coordinates.

    After estimating the deformation φ, the GW term in the next INCENT-SE
    FGW iteration should use D_B(φ) = pairwise distances of φ(coords_B)
    rather than the original D_B.  This is the "Block B → FGW" information
    flow in the joint optimisation.

    Parameters
    ----------
    coords_B : (n_B, 2) float array — original sliceB cell coordinates.
    phi : LDDMMDeformation — the estimated deformation.
    normalise : bool, default True
        If True, divide by max(D_B(φ)) for consistent shared-scale
        normalisation with INCENT's D_A (which was already normalised).

    Returns
    -------
    D_B_deformed : (n_B, n_B) float64 array.
    """
    y_phi    = phi.apply(coords_B)                          # (n_B, 2)
    # Pairwise Euclidean: ||y_i^φ - y_j^φ||
    diff     = y_phi[:, None, :] - y_phi[None, :, :]       # (n_B, n_B, 2)
    D_B_def  = np.sqrt((diff ** 2).sum(axis=2))             # (n_B, n_B)

    if normalise:
        m = D_B_def.max()
        if m > 1e-12:
            D_B_def /= m

    return D_B_def.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC HELPER: growth_vector
# ─────────────────────────────────────────────────────────────────────────────

def estimate_growth_vector(
    pi: np.ndarray,
    b: np.ndarray,
    kappa: float = 0.1,
) -> np.ndarray:
    """
    Estimate the per-cell growth vector ξ for the semi-relaxed OT formulation.

    In the cross-timepoint OT, the target marginal is relaxed:
        π^T · 1 ≈ ξ ⊙ b
    where ξ[j] > 1 means cell j in sliceB has "more" mass than expected
    (it proliferated or expanded) and ξ[j] < 1 means it contracted or
    represents a dying population.

    Estimation: ξ_j = (π^T · 1)[j] / b[j]
    Regularised towards 1 by a prior ||ξ - 1||² with weight κ:
        ξ_j = ((π^T · 1)[j] + κ · b[j]) / (b[j] + κ · b[j])
            = ((π^T · 1)[j] / b[j] + κ) / (1 + κ)

    Parameters
    ----------
    pi : (n_A, n_B) float array — current transport plan.
    b : (n_B,) float array — target marginal (uniform = 1/n_B).
    kappa : float, default 0.1 — prior weight towards ξ=1 (no growth).

    Returns
    -------
    xi : (n_B,) float array — growth vector.
        xi > 1: proliferating regions.
        xi < 1: contracting/apoptotic regions.
        xi ≈ 1: stable populations.
    """
    pi_col = pi.sum(axis=0)                    # (n_B,) actual marginal
    raw    = pi_col / (b + 1e-12)              # raw per-cell growth rate
    # Regularise towards 1 (weak prior: growth is approximately 1)
    xi     = (raw + kappa) / (1.0 + kappa)
    return xi.astype(np.float64)
