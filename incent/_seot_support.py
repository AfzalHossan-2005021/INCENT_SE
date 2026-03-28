"""
Internal helpers for the retained SEOT-backed spatiotemporal pipeline.

These functions were extracted from the old standalone BISPA/RAPA modules so
the package can keep `pairwise_align_spatiotemporal(..., use_rapa=True)` while
exposing only the two public alignment entry points.
"""

import warnings

import numpy as np
from anndata import AnnData

from ._gpu import resolve_device, to_torch


def apply_rotation_only_pose(
    sliceA: AnnData,
    sliceB: AnnData,
    theta_deg: float,
    verbose: bool = True,
) -> AnnData:
    """Rotate sliceA, then place its centroid on sliceB's centroid."""
    from .pose import _rotation_matrix

    sliceA = sliceA.copy()
    R = _rotation_matrix(theta_deg)
    coords = sliceA.obsm["spatial"].astype(np.float64)

    cA = coords.mean(axis=0)
    rotated = (R @ (coords - cA).T).T + cA

    cB = sliceB.obsm["spatial"].astype(np.float64).mean(axis=0)
    t_neutral = cB - rotated.mean(axis=0)
    sliceA.obsm["spatial"] = rotated + t_neutral

    if verbose:
        print(
            f"[SEOT init] theta={theta_deg:.1f}  "
            f"neutral tx={t_neutral[0]:.1f}  ty={t_neutral[1]:.1f}"
        )

    return sliceA


def decompose_slice(
    adata: AnnData,
    n_neighbors: int = 15,
    resolution=None,
    min_community_size_frac: float = 0.15,
    target_min_region_frac: float = 0.20,
    slice_label: str = "slice",
    verbose: bool = True,
) -> np.ndarray:
    """Split one slice into coarse spatial communities."""
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        warnings.warn(
            "[SEOT] leidenalg/igraph not found; falling back to spectral clustering.",
            stacklevel=2,
        )
        return _spectral_fallback(adata, verbose=verbose)

    from sklearn.neighbors import NearestNeighbors

    coords = adata.obsm["spatial"].astype(np.float64)
    n = len(coords)

    if verbose:
        print(f"[SEOT decompose:{slice_label}] kNN graph (k={n_neighbors}) on {n} cells")

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="ball_tree")
    nn.fit(coords)
    dists, indices = nn.kneighbors(coords)
    dists, indices = dists[:, 1:], indices[:, 1:]

    sigma = np.median(dists)
    weights = np.exp(-(dists ** 2) / (2 * sigma ** 2)).ravel()
    rows = np.repeat(np.arange(n), n_neighbors)
    cols = indices.ravel()

    G_ig = ig.Graph(n=n, edges=list(zip(rows.tolist(), cols.tolist())), directed=False)
    G_ig.es["weight"] = weights.tolist()

    if resolution is not None:
        res_final = resolution
    else:
        lo, hi = 1e-6, 1.0
        best_res = lo
        best_lbl = None

        for _ in range(25):
            mid = (lo + hi) / 2.0
            part = leidenalg.find_partition(
                G_ig,
                leidenalg.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=mid,
                seed=42,
            )
            lbl = np.array(part.membership, dtype=np.int32)
            sizes = np.array([(lbl == k).sum() for k in np.unique(lbl)])
            min_f = sizes.min() / n

            if min_f >= target_min_region_frac:
                best_res, best_lbl = mid, lbl
                lo = mid
            else:
                hi = mid

            if (hi - lo) < 1e-7:
                break

        if best_lbl is None:
            return np.zeros(n, dtype=np.int32)

        res_final = best_res

    part = leidenalg.find_partition(
        G_ig,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=res_final,
        seed=42,
    )
    labels = np.array(part.membership, dtype=np.int32)
    labels = _merge_small(labels, coords, min_community_size_frac)

    if len(np.unique(labels)) == 1:
        labels = _expression_guided_spectral(adata, n_clusters=2, verbose=verbose)

    if verbose:
        for k in np.unique(labels):
            sz = (labels == k).sum()
            print(f"  [{slice_label}] C_{k}: {sz} cells ({sz / n * 100:.1f}%)")

    return labels


def _expression_guided_spectral(
    adata: AnnData,
    n_clusters: int = 2,
    n_neighbors: int = 15,
    verbose: bool = True,
) -> np.ndarray:
    import scipy.sparse as sp
    from sklearn.cluster import SpectralClustering
    from sklearn.neighbors import NearestNeighbors

    coords = adata.obsm["spatial"].astype(np.float64)
    n = len(coords)

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="ball_tree")
    nn.fit(coords)
    dists, indices = nn.kneighbors(coords)
    dists, indices = dists[:, 1:], indices[:, 1:]
    sigma_sp = np.median(dists) + 1e-6
    W_sp = np.exp(-(dists ** 2) / (2 * sigma_sp ** 2))

    ct = np.asarray(adata.obs["cell_type_annot"].astype(str))
    uct = np.unique(ct)
    ct_idx = np.array([np.where(uct == c)[0][0] for c in ct])

    W_expr = np.zeros_like(W_sp)
    for i in range(n):
        nbrs = indices[i]
        same = (ct_idx[nbrs] == ct_idx[i]).astype(np.float32)
        W_expr[i] = same + 0.1

    W_comb = W_sp * W_expr
    rows = np.repeat(np.arange(n), n_neighbors)
    cols = indices.ravel()
    vals = W_comb.ravel()

    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    A = (A + A.T).toarray() * 0.5

    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=42,
        n_init=10,
    )
    labels = sc.fit_predict(A).astype(np.int32)

    if verbose:
        for k in np.unique(labels):
            sz = (labels == k).sum()
            print(f"  [spectral] C_{k}: {sz} cells ({sz / n * 100:.1f}%)")

    return labels


def _spectral_fallback(adata: AnnData, n_components: int = 2, verbose: bool = True) -> np.ndarray:
    from sklearn.cluster import SpectralClustering

    coords = adata.obsm["spatial"].astype(np.float64)
    if verbose:
        print(f"[SEOT] spectral clustering fallback (K={n_components})")

    sc = SpectralClustering(
        n_clusters=n_components,
        affinity="nearest_neighbors",
        n_neighbors=15,
        random_state=42,
    )
    return sc.fit_predict(coords).astype(np.int32)


def _merge_small(labels: np.ndarray, coords: np.ndarray, min_frac: float) -> np.ndarray:
    """Merge communities smaller than min_frac * n into their nearest neighbor."""
    n = len(labels)
    labels = labels.copy()
    changed = True

    while changed:
        changed = False
        unique, counts = np.unique(labels, return_counts=True)
        for k, cnt in zip(unique, counts):
            if cnt >= min_frac * n:
                continue

            mask_k = labels == k
            c_k = coords[mask_k].mean(axis=0)
            best_k2, best_d = -1, np.inf
            for k2 in unique:
                if k2 == k:
                    continue
                c_k2 = coords[labels == k2].mean(axis=0)
                d = np.linalg.norm(c_k - c_k2)
                if d < best_d:
                    best_d, best_k2 = d, k2

            if best_k2 >= 0:
                labels[mask_k] = best_k2
                changed = True
                break

    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    return np.array([remap[l] for l in labels], dtype=np.int32)


def _region_profile(adata: AnnData, mask=None) -> dict:
    """Compute a simple multimodal profile for one whole slice/community."""
    import scipy.sparse as sp

    if mask is not None:
        adata = adata[mask]

    ct = np.asarray(adata.obs["cell_type_annot"].astype(str))
    uct = np.unique(ct)
    cnt = np.array([(ct == c).sum() for c in uct], dtype=np.float32)

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    X = np.where(np.isfinite(X), X, 0.0)

    coords = adata.obsm["spatial"].astype(np.float64)
    xr = coords[:, 0].max() - coords[:, 0].min() + 1e-6
    yr = coords[:, 1].max() - coords[:, 1].min() + 1e-6

    return {
        "cell_types": uct,
        "ct_dist": cnt / cnt.sum(),
        "expr_mean": X.mean(axis=0),
        "centroid": coords.mean(axis=0),
        "aspect": xr / yr,
        "n_cells": len(adata),
    }


def _profile_dist(
    pA: dict,
    pB: dict,
    shared_ct: np.ndarray,
    cross_timepoint: bool = False,
) -> float:
    """Composite region distance used for community matching."""

    def _ct_vec(profile: dict) -> np.ndarray:
        v = np.zeros(len(shared_ct), dtype=np.float64)
        for i, ct in enumerate(shared_ct):
            idx = np.where(profile["cell_types"] == ct)[0]
            if len(idx):
                v[i] = profile["ct_dist"][idx[0]]
        v += 1e-10
        return v / v.sum()

    va, vb = _ct_vec(pA), _ct_vec(pB)
    mix = (va + vb) / 2.0
    jsd = max(float(np.sum(va * np.log(va / mix) + vb * np.log(vb / mix)) / 2.0), 0.0)

    ea = pA["expr_mean"].astype(np.float64)
    eb = pB["expr_mean"].astype(np.float64)
    na, nb = np.linalg.norm(ea), np.linalg.norm(eb)
    expr_d = 1.0 - float(ea @ eb) / (na * nb) if na > 1e-10 and nb > 1e-10 else 1.0

    asp_d = min(abs(np.log(max(pA["aspect"], 1e-3)) - np.log(max(pB["aspect"], 1e-3))), 1.0)

    w_jsd = 0.70
    w_expr = 0.05 if cross_timepoint else 0.20
    w_asp = 0.25 if cross_timepoint else 0.10
    return w_jsd * jsd + w_expr * expr_d + w_asp * asp_d


def build_community_similarity(
    sliceA: AnnData,
    labels_A: np.ndarray,
    sliceB: AnnData,
    labels_B: np.ndarray,
    cross_timepoint: bool = False,
    verbose: bool = True,
):
    """Build the community-to-community distance matrix."""
    ct_A = set(sliceA.obs["cell_type_annot"].astype(str).unique())
    ct_B = set(sliceB.obs["cell_type_annot"].astype(str).unique())
    shared = np.array(sorted(ct_A & ct_B))

    comms_A = np.unique(labels_A)
    comms_B = np.unique(labels_B)
    K_A, K_B = len(comms_A), len(comms_B)

    coords_B = sliceB.obsm["spatial"].astype(np.float64)
    max_diam = float(np.linalg.norm(coords_B.max(axis=0) - coords_B.min(axis=0))) + 1e-6

    pA_list = {k: _region_profile(sliceA, labels_A == k) for k in comms_A}
    pB_list = {l: _region_profile(sliceB, labels_B == l) for l in comms_B}

    S = np.zeros((K_A, K_B), dtype=np.float32)
    for i, k in enumerate(comms_A):
        for j, l in enumerate(comms_B):
            bio_d = _profile_dist(pA_list[k], pB_list[l], shared, cross_timepoint)
            sp_d = float(np.linalg.norm(pA_list[k]["centroid"] - pB_list[l]["centroid"])) / max_diam
            S[i, j] = bio_d + 0.10 * sp_d

    if verbose:
        print(f"[SEOT match] K_A={K_A}  K_B={K_B}  S range=[{S.min():.3f}, {S.max():.3f}]")

    return S, comms_A, comms_B


def hungarian_matching(
    S: np.ndarray,
    comms_A: np.ndarray,
    comms_B: np.ndarray,
    threshold: float = 0.85,
    verbose: bool = True,
):
    """Solve the 1-to-1 community assignment problem."""
    from scipy.optimize import linear_sum_assignment

    K_A, K_B = S.shape
    K_max = max(K_A, K_B)
    S_pad = np.full((K_max, K_max), fill_value=1e6, dtype=np.float64)
    S_pad[:K_A, :K_B] = S.astype(np.float64)

    row_ind, col_ind = linear_sum_assignment(S_pad)

    matched_pairs = []
    for r, c in zip(row_ind, col_ind):
        if r >= K_A or c >= K_B:
            continue
        if S[r, c] > threshold:
            continue
        matched_pairs.append((int(comms_A[r]), int(comms_B[c])))

    matched_A = {k for k, _ in matched_pairs}
    matched_B = {l for _, l in matched_pairs}
    unmatched_A = np.array([k for k in comms_A if k not in matched_A], dtype=np.int32)
    unmatched_B = np.array([l for l in comms_B if l not in matched_B], dtype=np.int32)

    if verbose:
        print(f"[SEOT match] {len(matched_pairs)} matched pairs: {matched_pairs}")

    return matched_pairs, unmatched_A, unmatched_B


def recover_pose_matched(
    sliceA: AnnData,
    labels_A: np.ndarray,
    sliceB: AnnData,
    labels_B: np.ndarray,
    matched_pairs,
    grid_size: int = 256,
    verbose: bool = True,
):
    """Estimate pose from matched communities only."""
    from .pose import estimate_pose, _rotation_matrix

    if not matched_pairs:
        if verbose:
            print("[SEOT pose] No matched pairs; theta=0, t=(0, 0)")
        return 0.0, 0.0, 0.0, 0.0

    mask_A = np.isin(labels_A, [k for k, _ in matched_pairs])
    mask_B = np.isin(labels_B, [l for _, l in matched_pairs])

    if mask_A.sum() < 100 or mask_B.sum() < 100:
        sA_sub, sB_sub = sliceA, sliceB
    else:
        sA_sub = sliceA[mask_A]
        sB_sub = sliceB[mask_B]

    theta, _, _, score = estimate_pose(sA_sub, sB_sub, grid_size=grid_size, verbose=verbose)

    R = _rotation_matrix(theta)
    coords_A = sliceA.obsm["spatial"].astype(np.float64)
    coords_B = sliceB.obsm["spatial"].astype(np.float64)
    cA_global = coords_A.mean(axis=0)
    coords_A_r = (R @ (coords_A - cA_global).T).T + cA_global

    tx_sum, ty_sum, w_sum = 0.0, 0.0, 0.0
    for k_A, k_B in matched_pairs:
        w = float((labels_A == k_A).sum())
        c_A_r = coords_A_r[labels_A == k_A].mean(axis=0)
        c_B = coords_B[labels_B == k_B].mean(axis=0)
        t = c_B - c_A_r
        tx_sum += w * t[0]
        ty_sum += w * t[1]
        w_sum += w

    tx = tx_sum / w_sum
    ty = ty_sum / w_sum

    if verbose:
        print(f"[SEOT pose] theta={theta:.1f}  tx={tx:.1f}  ty={ty:.1f}  score={score:.3f}")

    return float(theta), float(tx), float(ty), float(score)


def build_bidirectional_anchor(
    sliceA: AnnData,
    labels_A: np.ndarray,
    sliceB: AnnData,
    labels_B: np.ndarray,
    matched_pairs,
    unmatched_A,
    unmatched_B,
    lambda_anchor: float = 2.0,
    boundary_sigma_frac: float = 0.05,
    use_gpu: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """Build the anchor matrix used to keep mass inside matched communities."""
    from sklearn.neighbors import NearestNeighbors

    del unmatched_A, unmatched_B, use_gpu

    coords_B = sliceB.obsm["spatial"].astype(np.float64)
    n_A, n_B = len(sliceA), len(sliceB)
    diam = float(np.linalg.norm(coords_B.max(axis=0) - coords_B.min(axis=0))) + 1e-6
    sigma = boundary_sigma_frac * diam + 1e-6

    M_anchor = np.full((n_A, n_B), lambda_anchor, dtype=np.float32)

    for k_A, k_B in matched_pairs:
        mask_A_k = labels_A == k_A
        mask_B_k = labels_B == k_B

        coords_in = coords_B[mask_B_k]
        coords_out = coords_B[~mask_B_k]

        soft_B = np.zeros(n_B, dtype=np.float32)
        soft_B[mask_B_k] = 1.0

        if (~mask_B_k).sum() > 0 and mask_B_k.sum() > 0:
            nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
            nn.fit(coords_in)
            min_dists, _ = nn.kneighbors(coords_out)
            min_dists = min_dists.ravel()
            soft_B[~mask_B_k] = np.exp(-(min_dists ** 2) / (2 * sigma ** 2)).astype(np.float32)

        rows_kA = np.where(mask_A_k)[0]
        M_anchor[np.ix_(rows_kA, np.arange(n_B))] = lambda_anchor * (1.0 - soft_B)[None, :]

    if verbose:
        n_zero = (M_anchor == 0).sum()
        print(
            f"[SEOT anchor] {n_zero}/{n_A * n_B} entries free "
            f"({n_zero / (n_A * n_B) * 100:.1f}%)"
        )

    return M_anchor


def compute_overlap_fractions(labels_A: np.ndarray, labels_B: np.ndarray, matched_pairs) -> tuple:
    """Return the matched-cell fraction on each side."""
    n_A = len(labels_A)
    n_B = len(labels_B)
    if not matched_pairs:
        return 0.0, 0.0

    mask_A = np.isin(labels_A, [k for k, _ in matched_pairs])
    mask_B = np.isin(labels_B, [l for _, l in matched_pairs])
    s_A = float(mask_A.sum()) / n_A
    s_B = float(mask_B.sum()) / n_B
    return s_A, s_B


def build_target_affinity(sliceB: AnnData, sigma: float, k_nn: int = 20):
    """Build the target-side sparse contiguity graph."""
    from .contiguity import build_spatial_affinity

    return build_spatial_affinity(sliceB.obsm["spatial"].astype(np.float64), sigma=sigma, k_nn=k_nn)


def target_contiguity_gradient(
    pi: np.ndarray,
    W_B,
    D_A: np.ndarray,
    use_gpu: bool = False,
) -> np.ndarray:
    """Gradient of the target-side contiguity regularizer."""
    device = resolve_device(use_gpu)

    if device == "cuda":
        import torch

        pi_t = to_torch(pi, device, dtype=torch.float32)
        D_A_t = to_torch(D_A, device, dtype=torch.float32)
        from ._gpu import sparse_to_torch

        W_B_t = sparse_to_torch(W_B, device, dtype=torch.float32)
        grad = torch.mm(D_A_t @ pi_t, W_B_t.to_dense())
        return (2.0 * grad).cpu().numpy().astype(np.float64)

    grad = (D_A @ pi) @ W_B
    return 2.0 * np.asarray(grad, dtype=np.float64)
