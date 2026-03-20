"""
contiguity.py — Spatial Contiguity Regulariser for INCENT-SE
=============================================================
Enforces that the matched cells form a *spatially contiguous* region.

The problem without this regulariser
-------------------------------------
Standard partial OT (PASTE2, FUGW) only constrains the total mass of the
transport plan π (Σ π_ij = s ≤ 1).  It does not care whether the matched
cells form a connected tissue patch or scattered isolated cells.

In practice, when cell types are evenly distributed (as in brain cortex),
partial OT often produces fragmented matchings — cells from different parts
of sliceA are matched to cells from different parts of sliceB.  This is
biologically unrealistic: we cut a *contiguous* piece of tissue.

The regulariser
---------------
R_spatial(π) penalises plans where two nearby cells in A are matched to
two distant cells in B:

    R_spatial(π) = Σ_{i,i' near in A}  Σ_{j,j' far in B}  π_ij · π_i'j'

Or equivalently, using the spatial distance matrices D_A and D_B:

    R_spatial(π) = <W_A, π D_B π^T>

where W_A[i,i'] = exp(-d_A(i,i')/σ) is a local affinity weight.

This is equivalent to saying: cells that are spatially close in A should
be matched to cells that are spatially close in B.

Gradient
--------
∂R/∂π = W_A · π · D_B + W_A^T · π · D_B^T   =   2 · W_A · π · D_B
(since W_A and D_B are symmetric).

This gradient can be added to the FGW gradient during the conditional-
gradient step without changing the rest of the solver.

Sparse implementation
---------------------
W_A is sparse: only pairs (i, i') with d_A(i,i') ≤ 3σ have significant
weight.  We use a CSR sparse matrix for W_A so the gradient computation
is O(n · k_NN) rather than O(n²).

Public API
----------
build_spatial_affinity(coords, sigma, k_nn) -> scipy.sparse.csr_matrix
contiguity_regulariser(pi, W_A, D_B) -> float
contiguity_gradient(pi, W_A, D_B) -> np.ndarray
"""

import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Build sparse spatial affinity matrix W_A
# ─────────────────────────────────────────────────────────────────────────────

def build_spatial_affinity(
    coords: np.ndarray,
    sigma: float,
    k_nn: int = 20,
) -> sp.csr_matrix:
    """
    Build the sparse spatial affinity matrix W_A for sliceA.

    W_A[i, i'] = exp( -d(i, i') / sigma )  if i' is among the k_nn nearest
                                              neighbours of i
               = 0                           otherwise

    This matrix encodes "how locally related are pairs of cells in A".
    We use a k-NN sparse structure to keep memory and computation manageable.

    Parameters
    ----------
    coords : (n_A, 2) float array — spatial coordinates of cells in sliceA.
    sigma : float
        Decay length for the affinity.  A good default is sigma = radius / 3
        (i.e. the neighbourhood radius used elsewhere in INCENT).
        Smaller sigma → only very nearby pairs are considered local.
    k_nn : int, default 20
        Number of nearest neighbours to consider for each cell.
        More neighbours → smoother regularisation but slower gradient.

    Returns
    -------
    W_A : (n_A, n_A) scipy.sparse.csr_matrix of float32.
        Symmetric, non-negative.

    Examples
    --------
    >>> W_A = build_spatial_affinity(sliceA.obsm['spatial'], sigma=100.0)
    """
    from sklearn.neighbors import NearestNeighbors

    n      = len(coords)
    # Find k nearest neighbours for every cell
    nn     = NearestNeighbors(n_neighbors=k_nn + 1, algorithm='ball_tree')
    nn.fit(coords)
    dists, indices = nn.kneighbors(coords)
    # Column 0 is the point itself (distance=0) — skip it
    dists   = dists[:, 1:]           # (n, k_nn)
    indices = indices[:, 1:]         # (n, k_nn)

    # Build sparse matrix in COO format
    rows   = np.repeat(np.arange(n), k_nn)
    cols   = indices.ravel()
    vals   = np.exp(-dists.ravel() / (sigma + 1e-10)).astype(np.float32)

    # Make symmetric by taking W = (W_raw + W_raw^T) / 2
    W      = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    W      = (W + W.T) * 0.5
    return W


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Compute the regulariser value
# ─────────────────────────────────────────────────────────────────────────────

def contiguity_regulariser(
    pi: np.ndarray,
    W_A: sp.csr_matrix,
    D_B: np.ndarray,
) -> float:
    """
    Evaluate the spatial contiguity regulariser R_spatial(π).

    R_spatial(π) = trace( W_A · π · D_B · π^T )
                 = <W_A, π · D_B · π^T>_F

    Interpretation:
      - If nearby cells in A (high W_A[i,i']) are matched to nearby cells in B
        (low D_B[j,j']), the term is small  → good, contiguous overlap.
      - If nearby cells in A are matched to distant cells in B, the term is
        large → penalised → optimiser avoids fragmented matchings.

    Parameters
    ----------
    pi : (n_A, n_B) float array — current transport plan.
    W_A : (n_A, n_A) sparse matrix — spatial affinity for sliceA.
    D_B : (n_B, n_B) float array — pairwise distances in sliceB
          (already normalised by max(D_B) from the shared-scale step).

    Returns
    -------
    float — regulariser value.  0 ≤ R_spatial < ∞.
    """
    # π · D_B :  (n_A, n_B) @ (n_B, n_B) = (n_A, n_B)
    pi_DB = pi @ D_B               # (n_A, n_B)

    # (π · D_B) · π^T : (n_A, n_B) @ (n_B, n_A) = (n_A, n_A)
    pi_DB_piT = pi_DB @ pi.T      # (n_A, n_A)

    # W_A ⊙ (π · D_B · π^T) summed = trace(W_A^T · M) = <W_A, M>_F
    # Since W_A is sparse we use the sparse multiply trick:
    val = W_A.multiply(pi_DB_piT).sum()
    return float(val)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Compute the gradient of the regulariser w.r.t. π
# ─────────────────────────────────────────────────────────────────────────────

def contiguity_gradient(
    pi: np.ndarray,
    W_A: sp.csr_matrix,
    D_B: np.ndarray,
) -> np.ndarray:
    """
    Compute ∂R_spatial/∂π.

    Derivation
    ----------
    R = <W_A, π D_B π^T>
      = Σ_{i,i'} W_A[i,i'] Σ_j Σ_j' π[i,j] D_B[j,j'] π[i',j']

    Taking derivative w.r.t. π[i,j]:
      ∂R/∂π[i,j] = Σ_{i'} W_A[i,i'] Σ_{j'} D_B[j,j'] π[i',j']
                 + Σ_{i'} W_A[i',i] Σ_{j'} π[i',j'] D_B[j',j]
                 = 2 · (W_A · π · D_B)[i,j]     (since W_A, D_B symmetric)

    In matrix form:  ∂R/∂π = 2 · W_A · π · D_B

    Parameters
    ----------
    pi : (n_A, n_B) float array — current transport plan.
    W_A : (n_A, n_A) sparse matrix — spatial affinity for sliceA.
    D_B : (n_B, n_B) float array — pairwise distances in sliceB.

    Returns
    -------
    grad : (n_A, n_B) float array — gradient ∂R_spatial/∂π.
        Add λ_spatial * grad to the FGW linear cost gradient at each step.
    """
    # W_A · π : sparse (n_A, n_A) @ dense (n_A, n_B) = dense (n_A, n_B)
    WA_pi = W_A @ pi        # (n_A, n_B)

    # (W_A · π) · D_B : (n_A, n_B) @ (n_B, n_B) = (n_A, n_B)
    grad  = WA_pi @ D_B     # (n_A, n_B)

    return 2.0 * grad.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Jointly estimate the overlap fraction s
# ─────────────────────────────────────────────────────────────────────────────

def estimate_overlap_fraction(
    pi: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    tol: float = 1e-4,
    max_iter: int = 50,
) -> float:
    """
    Estimate the overlap fraction s = Σ π_ij from the current transport plan.

    The overlap fraction represents what proportion of total mass is actually
    transported — i.e. what fraction of cells in sliceA have a match in sliceB.

    Strategy: bisection on the KKT dual variable for the total-mass constraint.
    We look for s such that KL(π·1 || s·a) + KL(π^T·1 || s·b) is minimised.

    In practice, a simpler and robust heuristic works well:
      s = (n_A / n_B) × (Σ_j max_i π_{ij}) / max_j(b_j)
    But we instead use the direct mass as the estimate: s = Σ_{ij} π_{ij}.
    This is exact when the plan is computed by partial OT (FUGW with
    unbalanced marginals) — the solver already accounts for partial mass.

    Parameters
    ----------
    pi : (n_A, n_B) float array — current transport plan.
    a : (n_A,) float — source marginal (uniform = 1/n_A each).
    b : (n_B,) float — target marginal (uniform = 1/n_B each).
    tol, max_iter : convergence parameters for bisection (currently unused —
        we use the direct estimate, kept for future extension).

    Returns
    -------
    s : float ∈ (0, 1] — estimated overlap fraction.
        s close to 1 means nearly full overlap.
        s ≈ 0.4 means ~40% of cells in A have a match in B.
    """
    s = float(pi.sum())
    # Clip to (0, 1] — plan mass should never exceed 1 for normalised marginals
    s = float(np.clip(s, 1e-6, 1.0))
    return s


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTION: augment_fgw_gradient
# ─────────────────────────────────────────────────────────────────────────────

def augment_fgw_gradient(
    pi: np.ndarray,
    W_A: sp.csr_matrix,
    D_B: np.ndarray,
    lambda_spatial: float,
) -> np.ndarray:
    """
    Compute the contiguity regulariser gradient to add to the FGW gradient.

    This is the function called inside the conditional-gradient loop (in
    ``fused_gromov_wasserstein_incent`` in utils.py).  It returns the
    additional gradient term that, when added to the existing FGW gradient,
    biases the solver towards spatially contiguous matchings.

    Parameters
    ----------
    pi : (n_A, n_B) float array — current transport plan.
    W_A : (n_A, n_A) sparse matrix — spatial affinity (from build_spatial_affinity).
    D_B : (n_B, n_B) float array — pairwise distances in sliceB (normalised).
    lambda_spatial : float
        Weight for the contiguity regulariser.
        0.0 → no contiguity enforcement (standard partial FUGW).
        0.1 → mild bias towards contiguous overlap.
        1.0 → strong bias.
        Typical values: 0.05 – 0.2.  Start with 0.1 and tune.

    Returns
    -------
    grad_augment : (n_A, n_B) float array — the contiguity gradient term.
        Add this to ``df(G)`` in ``cg_incent`` before the LP sub-problem.

    Examples
    --------
    >>> W_A = build_spatial_affinity(sliceA.obsm['spatial'], sigma=100.0)
    >>> # Inside the CG loop:
    >>> extra_grad = augment_fgw_gradient(pi, W_A, D_B, lambda_spatial=0.1)
    >>> Mi = Mi + extra_grad
    """
    if lambda_spatial == 0.0:
        return np.zeros_like(pi)
    return lambda_spatial * contiguity_gradient(pi, W_A, D_B)
