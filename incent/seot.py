"""
seot.py  SE(2)-OT EM: Explicit Rigid Transformation Recovery
This module solves the alignment problem that GW/FGW cannot solve.
"""
"""
seot.py  SE(2)-OT EM: Explicit Transformation Recovery via OT
==============================================================
This module solves the problem that FGW / GW fundamentally CANNOT solve:
recovering the rigid SE(2) transformation (rotation R + translation t)
that maps sliceA into sliceB's coordinate frame.

Why GW cannot recover the transformation
-----------------------------------------
GW minimises  sum_{ijkl} pi_ij pi_kl (D_A[i,k] - D_B[j,l])^2
using pairwise DISTANCES. Distances are rotation-invariant, so R is invisible
to the GW objective. After GW, the centroid-based translation is a heuristic
that fails when:
  - Both slices are partial (centroids don't correspond to same location)
  - Either slice contains extra symmetric regions
  - Slices are from different timepoints with shifted coordinate frames

The correct objective
-----------------------
E(pi, R, t) = (1-alpha) * sum_ij pi_ij * M_bio[i,j]
            +    alpha  * sum_ij pi_ij * ||R*x_i + t - y_j||^2 / D^2
            + KL(pi*1 || rho_A * a) + KL(pi^T*1 || rho_B * b)

where x_i are coordinates of cell i in sliceA, y_j in sliceB.
The second term EXPLICITLY contains R and t  ->  they are jointly optimised.

EM structure (alternating minimisation)
----------------------------------------
E-step (OT)    fix (R, t) -> find pi* via unbalanced Sinkhorn OT
               Cost[i,j] = (1-alpha)*M_bio[i,j] + alpha*||R*x_i+t-y_j||^2/D^2

M-step (SE(2)) fix pi     -> find (R*, t*) via weighted Procrustes (closed form)
               x_bar = sum_ij pi_ij x_i / Z         (pi-weighted centroid of A)
               y_bar = sum_ij pi_ij y_j / Z         (pi-weighted centroid of B)
               H = (x - x_bar)^T @ pi @ (y - y_bar)   shape (2, 2)
               U, S, V^T = SVD(H)
               R = V diag(1, det(V @ U^T)) @ U^T    (guaranteed rotation)
               t = y_bar - R @ x_bar

Each step strictly decreases E  ->  guaranteed convergence.

Initialisation via BISPA community matching
--------------------------------------------
The EM converges to the GLOBAL optimum only when started near the right basin.
For bilateral symmetry, the two basins (left / right hemisphere) have nearly
identical energies. We use BISPA community matching to:
  1. Identify matched community pairs (k_A, k_B).
  2. Compute R0 via Fourier-Mellin on matched cells only.
  3. Compute t0 from matched-community centroid offsets.
This provides an initialisation that breaks symmetry before EM starts.

Generalisation
---------------
  - Same-timepoint: M_bio = expression cosine + cell-type penalty + topology
  - Cross-timepoint: M_bio = cVAE latent cosine + topology
  - Any organ: BISPA community K determined adaptively
  - Any partial overlap: rho_A and rho_B set from matched fractions

Public API
----------
weighted_procrustes(pi, coords_A, coords_B) -> (R, t, residual)
build_spatial_cost(R, t, coords_A, coords_B, D_normalise) -> (n_A, n_B) array
seot_em_step(pi, M_bio, coords_A, coords_B, alpha, rho_A, rho_B, reg_sinkhorn, D)
pairwise_align_seot(sliceA, sliceB, ...)
"""

import os
import time
import datetime
import warnings
import numpy as np
from typing import Optional, Tuple, List
from anndata import AnnData

from ._gpu import resolve_device, to_torch, to_numpy


# ==========================================================================
# Core maths: weighted Procrustes (M-step, closed form)
# ==========================================================================

def weighted_procrustes(
    pi: np.ndarray,
    coords_A: np.ndarray,
    coords_B: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Closed-form SE(2) solution from a soft correspondence matrix pi.

    Given pi[i,j] = how strongly cell i in A corresponds to cell j in B,
    find rotation R and translation t that minimise:
        sum_ij pi_ij ||R * x_i + t - y_j||^2

    Solution (weighted Procrustes / Kabsch algorithm):
        Z  = pi.sum()
        x_bar = pi.sum(1) @ coords_A / Z   (pi-weighted centroid of A)
        y_bar = pi.sum(0) @ coords_B / Z   (pi-weighted centroid of B)
        H  = (coords_A - x_bar)^T @ pi @ (coords_B - y_bar)   (2x2)
        U, S, V^T = SVD(H)
        R  = V diag(1, det(V U^T)) U^T     (det correction prevents reflection)
        t  = y_bar - R @ x_bar

    Parameters
    ----------
    pi       : (n_A, n_B) float64 -- soft correspondence (transport plan).
    coords_A : (n_A, 2) float64  -- cell coordinates in sliceA.
    coords_B : (n_B, 2) float64  -- cell coordinates in sliceB.

    Returns
    -------
    R        : (2, 2) float64 -- rotation matrix.
    t        : (2,)   float64 -- translation vector.
    residual : float          -- weighted MSE = sum_ij pi_ij ||R x_i + t - y_j||^2 / Z.
    """
    Z = pi.sum()
    if Z < 1e-12:
        return np.eye(2), np.zeros(2), np.inf

    # Pi-weighted centroids
    row_sums = pi.sum(axis=1)   # (n_A,)
    col_sums = pi.sum(axis=0)   # (n_B,)
    x_bar = (row_sums @ coords_A) / Z   # (2,)
    y_bar = (col_sums @ coords_B) / Z   # (2,)

    # Cross-covariance matrix H = X_c^T @ pi @ Y_c   shape (2, 2)
    X_c = coords_A - x_bar    # (n_A, 2)  centred A
    Y_c = coords_B - y_bar    # (n_B, 2)  centred B
    H   = X_c.T @ pi @ Y_c   # (2, n_A) @ (n_A, n_B) @ (n_B, 2) = (2, 2)

    # SVD of H
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # Determinant correction: ensure R is a proper rotation (det=+1), not reflection
    d = np.linalg.det(V @ U.T)
    R = V @ np.diag([1.0, d]) @ U.T   # (2, 2)

    # Translation
    t = y_bar - R @ x_bar   # (2,)

    # Weighted residual
    coords_A_transformed = (R @ coords_A.T).T + t   # (n_A, 2)
    diff_sq = ((coords_A_transformed[:, None, :] - coords_B[None, :, :]) ** 2).sum(axis=2)
    residual = float((pi * diff_sq).sum() / Z)

    return R, t, residual


# ==========================================================================
# E-step: build spatial cost and solve linear OT
# ==========================================================================

def build_spatial_cost(
    R: np.ndarray,
    t: np.ndarray,
    coords_A: np.ndarray,
    coords_B: np.ndarray,
    D_normalise: float,
) -> np.ndarray:
    """
    Compute the normalised squared Euclidean cost ||R x_i + t - y_j||^2 / D^2.

    Parameters
    ----------
    R           : (2, 2) rotation matrix.
    t           : (2,) translation.
    coords_A    : (n_A, 2) cell coordinates in sliceA.
    coords_B    : (n_B, 2) cell coordinates in sliceB.
    D_normalise : float -- normalisation scale (typically max pairwise distance in B).

    Returns
    -------
    C_spatial : (n_A, n_B) float32 -- normalised squared distances.
    """
    # Transform A coordinates into B's frame
    coords_A_t = (R @ coords_A.T).T + t   # (n_A, 2)

    # Pairwise squared Euclidean: ||x_i_transformed - y_j||^2
    sq_A = (coords_A_t ** 2).sum(axis=1, keepdims=True)   # (n_A, 1)
    sq_B = (coords_B   ** 2).sum(axis=1, keepdims=True).T  # (1, n_B)
    D2   = sq_A + sq_B - 2.0 * (coords_A_t @ coords_B.T)  # (n_A, n_B)
    D2   = np.maximum(D2, 0.0)

    return (D2 / (D_normalise ** 2 + 1e-12)).astype(np.float32)


def solve_ot_step(
    cost: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    rho_A: float,
    rho_B: float,
    reg_sinkhorn: float = 0.01,
) -> np.ndarray:
    """
    E-step: solve unbalanced OT with the combined cost matrix.

    Uses POT's sinkhorn_unbalanced which handles the KL marginal relaxation:
      min_pi <cost, pi> + rho_A * KL(pi*1 || a) + rho_B * KL(pi^T*1 || b)
      + reg * H(pi)    (entropic regularisation for stability)

    Parameters
    ----------
    cost         : (n_A, n_B) float32 -- combined M_bio + alpha * C_spatial.
    a            : (n_A,) float64     -- source marginal (uniform).
    b            : (n_B,) float64     -- target marginal (uniform).
    rho_A        : float -- source marginal relaxation.
    rho_B        : float -- target marginal relaxation.
    reg_sinkhorn : float -- Sinkhorn entropic regularisation.

    Returns
    -------
    pi : (n_A, n_B) float64 -- transport plan.
    """
    import ot
    pi = ot.unbalanced.sinkhorn_unbalanced(
        a=a.astype(np.float64),
        b=b.astype(np.float64),
        M=cost.astype(np.float64),
        reg=reg_sinkhorn,
        reg_m=(rho_A, rho_B),
        numItermax=1000,
        stopThr=1e-7,
        log=False,
    )
    return np.asarray(pi, dtype=np.float64)


# ==========================================================================
# Full EM loop
# ==========================================================================

def seot_em(
    M_bio: np.ndarray,
    coords_A: np.ndarray,
    coords_B: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    R_init: np.ndarray,
    t_init: np.ndarray,
    alpha: float = 0.5,
    rho_A: float = 0.5,
    rho_B: float = 0.5,
    reg_sinkhorn: float = 0.01,
    max_iter: int = 50,
    tol: float = 1e-5,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
    """
    SE(2)-OT EM algorithm: jointly optimise (R, t) and correspondence pi.

    Alternates between:
      E-step: fix (R, t) -> solve unbalanced linear OT
              Cost = (1-alpha)*M_bio + alpha*||R x_i + t - y_j||^2 / D^2
      M-step: fix pi     -> solve weighted Procrustes for (R, t)

    Guaranteed to converge (each step strictly decreases the objective).

    Parameters
    ----------
    M_bio       : (n_A, n_B) float32  -- biological cost (expression + topology).
    coords_A    : (n_A, 2)  float64   -- sliceA cell coordinates.
    coords_B    : (n_B, 2)  float64   -- sliceB cell coordinates.
    a           : (n_A,)    float64   -- source marginal.
    b           : (n_B,)    float64   -- target marginal.
    R_init      : (2, 2)    float64   -- initial rotation (e.g. from BISPA).
    t_init      : (2,)      float64   -- initial translation.
    alpha       : float -- spatial weight [0=biology only, 1=spatial only].
    rho_A, rho_B: float -- KL marginal relaxation per side.
                   rho ~ s (matched fraction): smaller = more cells unmatched.
    reg_sinkhorn: float -- Sinkhorn entropic regularisation.
                   Smaller = more precise but slower. Try 0.005 - 0.05.
    max_iter    : int  -- maximum EM iterations.
    tol         : float -- convergence threshold on residual change.
    verbose     : bool.

    Returns
    -------
    pi          : (n_A, n_B) float64 -- final transport plan.
    R           : (2, 2) float64     -- recovered rotation.
    t           : (2,)   float64     -- recovered translation.
    history     : list of float      -- residual per iteration.
    """
    # Normalise spatial scale by the diameter of B
    D_norm = float(np.linalg.norm(coords_B.max(axis=0) - coords_B.min(axis=0)))
    if D_norm < 1e-6:
        D_norm = 1.0

    R, t = R_init.copy(), t_init.copy()
    history = []
    pi = np.outer(a, b)   # uniform initialisation

    for it in range(max_iter):
        # ── E-step: OT with current (R, t) ────────────────────────────────
        C_spatial = build_spatial_cost(R, t, coords_A, coords_B, D_norm)
        cost = ((1.0 - alpha) * M_bio.astype(np.float32)
                + alpha        * C_spatial).astype(np.float64)

        pi = solve_ot_step(cost, a, b, rho_A, rho_B, reg_sinkhorn)

        # ── M-step: weighted Procrustes ────────────────────────────────────
        R, t, residual = weighted_procrustes(pi, coords_A, coords_B)

        history.append(residual)

        if verbose:
            theta = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
            print(f"  [SEOT EM] iter={it+1:3d}  residual={residual:.6f}  "
                  f"theta={theta:.2f}  tx={t[0]:.1f}  ty={t[1]:.1f}  "
                  f"pi_mass={pi.sum():.4f}")

        if it > 0 and abs(history[-1] - history[-2]) / (abs(history[-2]) + 1e-12) < tol:
            if verbose:
                print(f"  [SEOT EM] Converged at iteration {it+1}.")
            break

    return pi, R, t, history


# ==========================================================================
# Initialisation via BISPA community matching
# ==========================================================================

def _initialise_from_bispa(
    sliceA: AnnData,
    sliceB: AnnData,
    target_min_region_frac: float,
    matching_threshold: float,
    rough_grid_size: int,
    verbose: bool,
) -> Tuple[np.ndarray, np.ndarray, float, dict]:
    """
    Use BISPA community matching to compute (R_init, t_init) for the EM loop.

    This breaks bilateral symmetry before EM starts: BISPA identifies which
    community of A corresponds to which community of B, then computes the
    transformation that aligns matched community centroids.

    Returns R_init, t_init, match_score, bispa_info dict.
    """
    from .bispa import (
        decompose_slice, build_community_similarity, hungarian_matching,
        recover_pose_matched, compute_overlap_fractions,
    )
    from .pose import _rotation_matrix

    if verbose:
        print("[SEOT init] BISPA community decomposition ...")

    # Rough rotation for symmetry-breaking before decomposition
    from .pose import estimate_pose
    theta_rough, _, _, _ = estimate_pose(
        sliceA, sliceB, grid_size=rough_grid_size, verbose=False)
    from .rapa import apply_rotation_only_pose
    sliceA_rough = apply_rotation_only_pose(sliceA, sliceB, theta_rough, verbose=False)

    labels_A = decompose_slice(
        sliceA_rough,
        target_min_region_frac=target_min_region_frac,
        slice_label="A_init", verbose=verbose)
    labels_B = decompose_slice(
        sliceB,
        target_min_region_frac=target_min_region_frac,
        slice_label="B_init", verbose=verbose)

    S, comms_A, comms_B = build_community_similarity(
        sliceA_rough, labels_A, sliceB, labels_B, verbose=False)

    matched_pairs, unmatched_A, unmatched_B = hungarian_matching(
        S, comms_A, comms_B, threshold=matching_threshold, verbose=verbose)

    theta_ref, tx_ref, ty_ref, pose_score = recover_pose_matched(
        sliceA_rough, labels_A, sliceB, labels_B,
        matched_pairs, grid_size=rough_grid_size, verbose=verbose)

    R_init = _rotation_matrix(theta_ref)
    t_init = np.array([tx_ref, ty_ref], dtype=np.float64)

    s_A, s_B = compute_overlap_fractions(labels_A, labels_B, matched_pairs)

    bispa_info = {
        "labels_A": labels_A, "labels_B": labels_B,
        "matched_pairs": matched_pairs,
        "unmatched_A": unmatched_A, "unmatched_B": unmatched_B,
        "s_A": s_A, "s_B": s_B,
        "theta_init": theta_ref, "pose_score": pose_score,
    }

    if verbose:
        print(f"[SEOT init] R_init: theta={theta_ref:.1f}  "
              f"t=({tx_ref:.1f},{ty_ref:.1f})  s_A={s_A:.3f}  s_B={s_B:.3f}")

    return R_init, t_init, pose_score, bispa_info


# ==========================================================================
# Main public function
# ==========================================================================

def pairwise_align_seot(
    sliceA: AnnData,
    sliceB: AnnData,
    alpha: float,
    beta: float,
    gamma: float,
    radius: float,
    filePath: str,
    # EM parameters
    max_em_iter: int = 50,
    tol_em: float = 1e-5,
    reg_sinkhorn: float = 0.01,
    # Marginal relaxation (rho = None -> computed from BISPA match fractions)
    rho_A: Optional[float] = None,
    rho_B: Optional[float] = None,
    base_rho: float = 0.5,
    # BISPA initialisation
    target_min_region_frac: float = 0.20,
    matching_threshold: float = 0.85,
    rough_grid_size: int = 256,
    # Anchor (penalise matching outside matched communities)
    use_anchor: bool = True,
    lambda_anchor: float = 2.0,
    boundary_sigma_frac: float = 0.05,
    # Bilateral contiguity refinement
    lambda_spatial: float = 0.05,
    lambda_target: float = 0.05,
    # cVAE for cross-timepoint
    cvae_model=None,
    cvae_path: Optional[str] = None,
    cvae_epochs: int = 80,
    cvae_latent_dim: int = 32,
    cross_timepoint: bool = False,
    # Standard
    use_rep: Optional[str] = None,
    numItermax: int = 2000,
    use_gpu: bool = False,
    gpu_verbose: bool = True,
    verbose: bool = True,
    sliceA_name: Optional[str] = None,
    sliceB_name: Optional[str] = None,
    overwrite: bool = False,
    neighborhood_dissimilarity: str = "jsd",
    return_diagnostics: bool = False,
):
    """
    SE(2)-OT EM: jointly recover the rigid transformation and cell correspondences.

    This solves the fundamental problem that GW/FGW CANNOT solve:
    recovering the rotation R and translation t that places sliceA at its
    true position in sliceB's coordinate frame.

    Why GW fails at this
    ---------------------
    GW uses pairwise DISTANCES (rotation-invariant). The transformation
    (R, t) is completely invisible to the GW objective. After GW, the
    post-hoc centroid-based translation is wrong whenever:
      - Both slices are partial with non-matching centroids
      - Either slice has extra symmetric regions (extra hemisphere)
      - Slices are from different scanners with incompatible coordinate origins

    What SEOT does instead
    -----------------------
    Minimises E(pi, R, t) = (1-alpha) * <M_bio, pi>
                           + alpha    * sum_ij pi_ij ||R x_i + t - y_j||^2 / D^2
                           + KL marginal terms

    This explicitly includes (R, t) in the objective. The EM algorithm
    alternates between:
      E-step: fix (R,t) -> solve unbalanced Sinkhorn OT -> pi
      M-step: fix pi    -> solve weighted Procrustes -> (R, t)  [CLOSED FORM]

    Guaranteed to converge. Typically 15-30 iterations.

    Initialisation
    --------------
    BISPA community matching provides (R_init, t_init) that breaks bilateral
    symmetry: Fourier rotation on matched cells + centroid-offset translation.
    The EM then refines this to the exact transformation.

    Parameters
    ----------
    sliceA, sliceB : AnnData with .obsm['spatial'] and .obs['cell_type_annot'].
    alpha  : float -- spatial weight in the cost. 0=biology only, 1=spatial only.
             Recommended: 0.3-0.5 for same-timepoint; 0.5-0.7 for cross-timepoint.
    beta   : float -- cell-type mismatch weight inside M_bio.
    gamma  : float -- neighbourhood dissimilarity weight.
    radius : float -- neighbourhood radius (spatial coordinate units).
    filePath : str -- directory for cache files and logs.

    max_em_iter  : int, default 50 -- maximum EM iterations.
    tol_em       : float, default 1e-5 -- convergence tolerance on residual.
    reg_sinkhorn : float, default 0.01 -- Sinkhorn entropic regularisation.
                   Smaller = more precise, slower. Range: 0.001 - 0.1.

    rho_A, rho_B : float or None -- KL marginal relaxation per side.
                   None = computed from BISPA matched fractions (recommended).
                   Smaller = more cells unmatched on that side.
                   For full overlap: rho ~ 1.0.  For 50% overlap: rho ~ 0.3.
    base_rho     : float, default 0.5 -- scale factor for auto-computed rho.

    target_min_region_frac : float, default 0.20
        For BISPA init: each community must cover >= this fraction of n_cells.
        0.20 -> K=2 (brain hemispheres).
        0.10 -> K<=4 (heart chambers).
    matching_threshold : float, default 0.85 -- max distance for matched pair.

    use_anchor     : bool, default True
        Add anchor cost penalising transport outside matched communities.
    lambda_anchor  : float, default 2.0 -- anchor penalty weight.

    lambda_spatial, lambda_target : float, default 0.05
        Bilateral contiguity regularisation weights (post-EM refinement).

    cross_timepoint : bool, default False
        True -> use cVAE latent cost for M_bio (temporal expression drift).
        False -> use raw cosine + cell-type cost.
    cvae_model / cvae_path : pre-trained INCENT_cVAE or saved model path.

    return_diagnostics : bool, default False
        True -> returns (pi, diagnostics_dict).

    Returns
    -------
    pi : (n_A, n_B) float64 -- final transport plan.
         argmax_j pi[i, :] gives the best-match cell in B for each cell in A.
         pi.sum() < 1 indicates partial overlap.

    If return_diagnostics=True:
        (pi, {
          "R": (2,2) rotation matrix recovered by EM,
          "t": (2,) translation vector,
          "theta_deg": float rotation angle in degrees,
          "residual_history": list of float,
          "pi_mass": float,
          "s_A": float, "s_B": float,
          "matched_pairs": list of (k_A, k_B),
          "sliceA_aligned": AnnData with transformed .obsm['spatial'],
          "bispa_info": dict,
        })
    """
    import ot as pot
    start_time = time.time()
    os.makedirs(filePath, exist_ok=True)

    log_name = (f"{filePath}/log_seot_{sliceA_name}_{sliceB_name}.txt"
                if sliceA_name and sliceB_name else f"{filePath}/log_seot.txt")
    log = open(log_name, "w")
    log.write(f"pairwise_align_seot -- INCENT-SE SEOT\n{datetime.datetime.now()}\n")
    log.write(f"alpha={alpha}  beta={beta}  gamma={gamma}  radius={radius}\n")
    log.write(f"max_em_iter={max_em_iter}  reg_sinkhorn={reg_sinkhorn}\n\n")

    # ==================================================================
    # STEP 1: BISPA initialisation  -> (R_init, t_init, s_A, s_B)
    # ==================================================================
    print("[SEOT] Step 1: BISPA initialisation ...")
    R_init, t_init, pose_score, bispa_info = _initialise_from_bispa(
        sliceA, sliceB,
        target_min_region_frac=target_min_region_frac,
        matching_threshold=matching_threshold,
        rough_grid_size=rough_grid_size,
        verbose=gpu_verbose,
    )
    s_A = bispa_info["s_A"]
    s_B = bispa_info["s_B"]
    matched_pairs  = bispa_info["matched_pairs"]
    unmatched_A    = bispa_info["unmatched_A"]
    unmatched_B    = bispa_info["unmatched_B"]
    labels_A       = bispa_info["labels_A"]
    labels_B       = bispa_info["labels_B"]

    # Set marginal relaxation from matched fractions (or user override)
    rho_A_use = rho_A if rho_A is not None else float(base_rho * max(s_A, 0.1))
    rho_B_use = rho_B if rho_B is not None else float(base_rho * max(s_B, 0.1))

    log.write(f"BISPA init: theta={bispa_info['theta_init']:.1f}  "
              f"pose_score={pose_score:.3f}  s_A={s_A:.3f}  s_B={s_B:.3f}\n")
    log.write(f"matched_pairs={matched_pairs}\n")
    log.write(f"rho_A={rho_A_use:.4f}  rho_B={rho_B_use:.4f}\n\n")

    # ==================================================================
    # STEP 2: Build biological cost M_bio (expression + cell-type + topology)
    # ==================================================================
    print("[SEOT] Step 2: Building M_bio ...")

    if cross_timepoint:
        from .cvae import INCENT_cVAE, train_cvae, latent_cost
        if cvae_model is not None:
            model = cvae_model
        elif cvae_path is not None and os.path.exists(cvae_path):
            model = INCENT_cVAE.load(cvae_path)
        else:
            print("[SEOT] Training cVAE ...")
            model = train_cvae([sliceA, sliceB], latent_dim=cvae_latent_dim,
                               epochs=cvae_epochs, verbose=gpu_verbose)
            if cvae_path:
                model.save(cvae_path)

    # Apply BISPA rotation-only pose to sliceA so coordinates are in B's rough frame
    from .rapa import apply_rotation_only_pose
    from .pose import _rotation_matrix
    sliceA_rough = apply_rotation_only_pose(sliceA, sliceB, bispa_info["theta_init"],
                                            verbose=False)

    # INCENT preprocessing (shared genes, cell types, cosine dist, JSD)
    from .core import _preprocess, _to_np
    log2 = open(f"{filePath}/log_seot_pre.txt", "w")
    p = _preprocess(
        sliceA_rough, sliceB, alpha, beta, gamma, radius, filePath,
        use_rep, None, None, None,
        numItermax, pot.backend.NumpyBackend(), use_gpu, gpu_verbose,
        sliceA_name, sliceB_name, overwrite, neighborhood_dissimilarity,
        log2)
    log2.close()

    sA_filt  = p["sliceA"]
    sB_filt  = p["sliceB"]
    a_np     = _to_np(p["a"])
    b_np     = _to_np(p["b"])
    n_A, n_B = sA_filt.shape[0], sB_filt.shape[0]

    if cross_timepoint:
        from .cvae import latent_cost
        M1_np = latent_cost(sA_filt, sB_filt, model).astype(np.float32)
    else:
        M1_np = _to_np(p["cosine_dist_gene_expr"]).astype(np.float32)

    M2_np = _to_np(p["M2"]).astype(np.float32)

    # Topological fingerprint cost
    from .topology import compute_fingerprints, fingerprint_cost
    fp_A = compute_fingerprints(sA_filt, radius=radius, n_bins=16,
                                 cache_path=filePath,
                                 slice_name=f"{sliceA_name or 'A'}_seot",
                                 overwrite=overwrite, verbose=gpu_verbose)
    fp_B = compute_fingerprints(sB_filt, radius=radius, n_bins=16,
                                 cache_path=filePath,
                                 slice_name=f"{sliceB_name or 'B'}_seot",
                                 overwrite=overwrite, verbose=gpu_verbose)
    M_topo = fingerprint_cost(fp_A, fp_B, metric="cosine", use_gpu=use_gpu).astype(np.float32)

    # Anchor cost (penalise transport outside matched communities)
    if use_anchor and matched_pairs:
        from .bispa import build_bidirectional_anchor
        def _remap(labels_full, adata_full, adata_filt):
            bc_full = np.array(adata_full.obs_names)
            bc_filt = np.array(adata_filt.obs_names)
            lab_map = {bc: labels_full[i] for i, bc in enumerate(bc_full)}
            return np.array([lab_map.get(bc, -1) for bc in bc_filt], dtype=np.int32)
        la_f = _remap(labels_A, sliceA_rough, sA_filt)
        lb_f = _remap(labels_B, sliceB,       sB_filt)
        M_anchor = build_bidirectional_anchor(
            sA_filt, la_f, sB_filt, lb_f,
            matched_pairs, unmatched_A, unmatched_B,
            lambda_anchor=lambda_anchor,
            boundary_sigma_frac=boundary_sigma_frac,
            use_gpu=use_gpu, verbose=gpu_verbose).astype(np.float32)
    else:
        M_anchor = np.zeros((n_A, n_B), dtype=np.float32)

    # Combined M_bio (biological cost, independent of spatial transformation)
    M_bio = M1_np + gamma * M2_np + 0.3 * M_topo + M_anchor

    # Coordinates of filtered slices in rough frame
    coords_A = sA_filt.obsm["spatial"].astype(np.float64)
    coords_B = sB_filt.obsm["spatial"].astype(np.float64)

    # Adjust R_init and t_init to account for the rough rotation already applied
    # (sliceA_rough has theta_init baked in; R_init from Procrustes is the
    #  ADDITIONAL rotation needed. Total rotation = R_procrustes @ R_rough)
    R_rough = _rotation_matrix(bispa_info["theta_init"])
    # The EM starts from the identity (rough rotation already applied to coords)
    # and the centroid-based translation from BISPA
    # Adjust t_init: it was computed relative to sliceA_rough coordinates
    t_init_em = t_init.astype(np.float64)
    R_init_em = np.eye(2)   # identity: rough rotation is already in coords_A

    # ==================================================================
    # STEP 3: SE(2)-OT EM
    # ==================================================================
    print(f"[SEOT] Step 3: SE(2)-OT EM (max_iter={max_em_iter}, "
          f"rho_A={rho_A_use:.3f}, rho_B={rho_B_use:.3f}) ...")

    pi, R_em, t_em, history = seot_em(
        M_bio=M_bio,
        coords_A=coords_A,
        coords_B=coords_B,
        a=a_np, b=b_np,
        R_init=R_init_em,
        t_init=t_init_em,
        alpha=alpha,
        rho_A=rho_A_use,
        rho_B=rho_B_use,
        reg_sinkhorn=reg_sinkhorn,
        max_iter=max_em_iter,
        tol=tol_em,
        verbose=verbose,
    )

    # Total transformation: rough rotation + EM refinement
    R_total = R_em @ R_rough   # (2,2) total rotation
    t_total = t_em             # translation was computed in rough-rotated frame
    theta_total = float(np.degrees(np.arctan2(R_total[1, 0], R_total[0, 0])))

    # ==================================================================
    # STEP 4: Bilateral contiguity post-refinement
    # ==================================================================
    if lambda_spatial > 0.0 or lambda_target > 0.0:
        print("[SEOT] Step 4: Bilateral contiguity refinement ...")
        from .contiguity import contiguity_gradient, build_spatial_affinity
        from .rapa import target_contiguity_gradient, build_target_affinity
        sigma_c = radius / 3.0
        D_B_np = _to_np(p["D_B"])
        D_A_np = _to_np(p["D_A"])
        W_A = build_spatial_affinity(coords_A, sigma=sigma_c, k_nn=20)
        W_B = build_target_affinity(sB_filt, sigma=sigma_c, k_nn=20)
        for _ in range(10):
            grad = np.zeros_like(pi)
            if lambda_spatial > 0.0:
                grad += lambda_spatial * contiguity_gradient(pi, W_A, D_B_np, use_gpu=use_gpu)
            if lambda_target > 0.0:
                grad += lambda_target * target_contiguity_gradient(pi, W_B, D_A_np, use_gpu=use_gpu)
            pi = np.maximum(pi - 0.05 * grad, 0.0)
            rs = pi.sum(axis=1, keepdims=True)
            pi = pi / np.maximum(rs, 1e-12) * a_np[:, None]

    pi_mass = float(pi.sum())
    runtime = time.time() - start_time

    log.write(f"EM converged: {len(history)} iterations\n")
    log.write(f"R_total:\n{R_total}\nt_total={t_total}\n")
    log.write(f"theta_total={theta_total:.2f}  pi_mass={pi_mass:.4f}\n")
    log.write(f"Runtime={runtime:.1f}s\n")
    log.close()

    print(f"[SEOT] Done.  theta={theta_total:.1f}  "
          f"t=({t_total[0]:.1f},{t_total[1]:.1f})  "
          f"pi_mass={pi_mass:.4f}  Runtime={runtime:.1f}s")

    # Build aligned sliceA for downstream use
    sliceA_aligned = sliceA.copy()
    sliceA_aligned.obsm["spatial"] = (
        (R_total @ sliceA.obsm["spatial"].astype(np.float64).T).T + t_total)

    if return_diagnostics:
        return pi, {
            "R": R_total,
            "t": t_total,
            "theta_deg": theta_total,
            "residual_history": history,
            "pi_mass": pi_mass,
            "s_A": s_A, "s_B": s_B,
            "matched_pairs": matched_pairs,
            "sliceA_aligned": sliceA_aligned,
            "bispa_info": bispa_info,
        }
    return pi
