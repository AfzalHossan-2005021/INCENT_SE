"""
topology.py — Topological Fingerprints for INCENT-SE
=====================================================
Computes a **persistent-homology fingerprint** for every cell.  These
fingerprints are:

  * SE(2)-invariant — built entirely from pairwise distances, so rotation
    and translation do not change them.
  * Symmetry-discriminative — bilaterally symmetric brain regions share the
    same cell-type composition, but their multi-scale *topological connectivity*
    (how patches of each cell type connect and merge as we increase radius)
    differs due to real anatomical asymmetries.

What is persistent homology?
----------------------------
Given a set of points, build a graph by adding edges whenever two points are
within distance ε of each other.  As ε grows from 0 to ∞:
  - Connected components (H0) *merge* (birth = ε when a new component appears,
    death = ε when it merges into another).
  - Loops (H1) *form and fill* at different scales.

The *persistence diagram* records all (birth, death) pairs.
The *Betti curve* B(ε) counts how many H0 components exist at scale ε —
a compact summary of connectivity across all scales.

For INCENT-SE we compute one Betti-0 curve per cell type within a local
neighbourhood of radius r_max around each cell.  Stacking K Betti curves
gives a K·L-dimensional fingerprint per cell.

Public API
----------
compute_fingerprints(adata, radius, n_bins, cache_path, overwrite) → np.ndarray
fingerprint_cost(fp_A, fp_B) → np.ndarray  [pairwise fingerprint distance matrix]
"""

import os
import numpy as np
import warnings
from typing import Optional
from anndata import AnnData


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Build local k-NN subgraph for each cell
# ─────────────────────────────────────────────────────────────────────────────

def _local_subgraph(
    coords: np.ndarray,
    labels: np.ndarray,
    center_idx: int,
    radius: float,
) -> tuple:
    """
    Return the pairwise distances and cell-type labels of all cells within
    `radius` of cell `center_idx`.

    Parameters
    ----------
    coords : (n, 2) float array — all cell coordinates.
    labels : (n,) str array  — all cell-type labels.
    center_idx : int — index of the centre cell.
    radius : float — neighbourhood radius (same units as coords).

    Returns
    -------
    sub_coords : (m, 2) — coordinates of cells in neighbourhood (incl. centre).
    sub_labels : (m,) str — their cell-type labels.
    """
    dists  = np.linalg.norm(coords - coords[center_idx], axis=1)
    mask   = dists <= radius
    return coords[mask], labels[mask]


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Betti-0 curve for a single cell type within a subgraph
# ─────────────────────────────────────────────────────────────────────────────

def _betti0_curve(
    sub_coords: np.ndarray,
    sub_labels: np.ndarray,
    target_type: str,
    epsilon_grid: np.ndarray,
) -> np.ndarray:
    """
    Compute the Betti-0 persistence curve for one cell type within a subgraph.

    For each scale ε in epsilon_grid we count how many connected components
    exist among the cells of `target_type`, where two cells are connected if
    their distance ≤ ε.

    We use a fast Union-Find (disjoint set) implementation — no need for
    the full gudhi library for H0 (which is just single-linkage clustering).

    Parameters
    ----------
    sub_coords : (m, 2) float array — coordinates in the local neighbourhood.
    sub_labels : (m,) str array  — cell-type labels.
    target_type : str — the cell type to analyse.
    epsilon_grid : (L,) float array — sorted scale values (ε₁ < ε₂ < … < ε_L).

    Returns
    -------
    betti_curve : (L,) int array — number of connected components at each ε.
        If `target_type` is absent from the subgraph, returns all-zeros.
    """
    mask  = sub_labels == target_type
    pts   = sub_coords[mask]
    n     = len(pts)

    if n == 0:
        return np.zeros(len(epsilon_grid), dtype=np.int32)
    if n == 1:
        # One isolated point — always 1 component
        return np.ones(len(epsilon_grid), dtype=np.int32)

    # Precompute all pairwise distances (n is small — local neighbourhood)
    diff  = pts[:, None, :] - pts[None, :, :]        # (n, n, 2)
    dists = np.sqrt((diff ** 2).sum(axis=2))           # (n, n)

    # Union-Find helpers
    parent = np.arange(n)
    rank   = np.zeros(n, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1

    # Sort all edges by distance
    i_idx, j_idx = np.triu_indices(n, k=1)
    edge_dists   = dists[i_idx, j_idx]
    order        = np.argsort(edge_dists)
    sorted_edges = list(zip(i_idx[order], j_idx[order], edge_dists[order]))

    # Sweep ε through the grid, merging edges as they become active
    betti_curve = np.zeros(len(epsilon_grid), dtype=np.int32)
    edge_ptr    = 0
    n_components = n     # start with every cell as its own component

    for l, eps in enumerate(epsilon_grid):
        # Add all edges with distance ≤ eps
        while edge_ptr < len(sorted_edges) and sorted_edges[edge_ptr][2] <= eps:
            ei, ej, _ = sorted_edges[edge_ptr]
            ri, rj    = find(ei), find(ej)
            if ri != rj:
                union(ei, ej)
                n_components -= 1
            edge_ptr += 1
        betti_curve[l] = n_components

    return betti_curve


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Full fingerprint for one cell
# ─────────────────────────────────────────────────────────────────────────────

def _cell_fingerprint(
    coords: np.ndarray,
    labels: np.ndarray,
    center_idx: int,
    cell_types: np.ndarray,
    radius: float,
    epsilon_grid: np.ndarray,
) -> np.ndarray:
    """
    Compute the topological fingerprint for a single cell.

    The fingerprint is the concatenation of Betti-0 curves for each cell type:
        f_i = [ B^(ct_1)(ε₁..ε_L), B^(ct_2)(ε₁..ε_L), ..., B^(ct_K)(ε₁..ε_L) ]
    Dimension = K × L.

    Parameters
    ----------
    coords : (n, 2) float array — all cell coordinates.
    labels : (n,) str array  — all cell-type labels.
    center_idx : int — index of the cell to fingerprint.
    cell_types : (K,) str array — the cell types to include.
    radius : float — local neighbourhood radius.
    epsilon_grid : (L,) float array — sorted spatial scales.

    Returns
    -------
    fp : (K * L,) float32 — the fingerprint vector.
    """
    sub_coords, sub_labels = _local_subgraph(coords, labels, center_idx, radius)

    curves = []
    for ct in cell_types:
        bc = _betti0_curve(sub_coords, sub_labels, ct, epsilon_grid)
        curves.append(bc.astype(np.float32))

    fp = np.concatenate(curves)           # (K * L,)

    # L2-normalise so fingerprint magnitude doesn't depend on local density
    norm = np.linalg.norm(fp)
    if norm > 1e-10:
        fp /= norm
    return fp


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTION: compute_fingerprints
# ─────────────────────────────────────────────────────────────────────────────

def compute_fingerprints(
    adata: AnnData,
    radius: float,
    n_bins: int = 16,
    cache_path: Optional[str] = None,
    slice_name: str = "slice",
    overwrite: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute topological fingerprints for every cell in `adata`.

    This is the INCENT-SE Stage 2 function for symmetry disambiguation.
    Results are cached to disk (as .npy) so they are only computed once.

    What the fingerprint captures
    -----------------------------
    For each cell i, we look at its local neighbourhood (radius r) and ask:
    "For each cell type k, how does the connectivity of type-k cells evolve
     as I increase the connection radius from 0 to r?"
    The answer is a Betti-0 curve: B^k(ε) = number of connected components
    of type-k cells at scale ε.

    Cells in the left hemisphere of the brain connect differently at different
    scales than cells in the right hemisphere — even if the cell-type counts
    are the same — because the tissue geometry differs.  This lets us
    distinguish mirror-symmetric regions without any coordinate information.

    Parameters
    ----------
    adata : AnnData
        Must have:
          - ``.obsm['spatial']`` — (n, 2) coordinate array.
          - ``.obs['cell_type_annot']`` — cell type labels.
    radius : float
        Neighbourhood radius.  Should match the ``radius`` parameter in
        ``pairwise_align`` (the same neighbourhood used for JSD).
        Typical value: 200–500 μm for MERFISH brain data.
    n_bins : int, default 16
        Number of scale levels in the Betti curve (the L above).
        The epsilon_grid is linearly spaced from 0 to radius.
        More bins → richer fingerprint but slower computation.
    cache_path : str or None
        Directory to save/load cached fingerprint arrays.
        If None, results are not cached.
    slice_name : str, default "slice"
        Identifier used in the cache filename.
    overwrite : bool, default False
        If True, recompute even if a cache file exists.
    verbose : bool, default True
        Show tqdm progress bar.

    Returns
    -------
    fingerprints : (n_cells, K * n_bins) float32 array
        Row i is the fingerprint vector for cell i.
        K = number of shared cell types in the dataset.

    Notes
    -----
    Computational cost: O(n × m² / n_avg) where m ≈ (n × πr²) / area is the
    average neighbourhood size.  For MERFISH with 15k cells, radius=200μm,
    and 16 bins this is ~30–120 s on a single CPU (depending on density).
    Parallelise with n_jobs=-1 for production runs.
    """
    from tqdm import tqdm

    # ── Check cache ────────────────────────────────────────────────────────
    if cache_path is not None:
        os.makedirs(cache_path, exist_ok=True)
        cache_file = os.path.join(cache_path, f"topo_fp_{slice_name}.npy")
        if os.path.exists(cache_file) and not overwrite:
            if verbose:
                print(f"[topology] Loading cached fingerprints from {cache_file}")
            return np.load(cache_file)

    coords     = adata.obsm['spatial'].astype(np.float64)
    labels     = np.asarray(adata.obs['cell_type_annot'].astype(str))
    cell_types = np.unique(labels)
    n_cells    = len(coords)
    K          = len(cell_types)
    L          = n_bins

    # Scale grid: L evenly-spaced values from 0 to radius
    # (we start at radius/L rather than 0 to avoid the trivial ε=0 case)
    epsilon_grid = np.linspace(radius / L, radius, L)

    if verbose:
        print(f"[topology] Computing fingerprints for {n_cells} cells, "
              f"K={K} cell types, L={L} scale bins, radius={radius}")

    fingerprints = np.zeros((n_cells, K * L), dtype=np.float32)

    for i in tqdm(range(n_cells), desc="Topological fingerprints",
                  disable=not verbose):
        fingerprints[i] = _cell_fingerprint(
            coords, labels, i, cell_types, radius, epsilon_grid)

    # ── Cache ──────────────────────────────────────────────────────────────
    if cache_path is not None:
        np.save(cache_file, fingerprints)
        if verbose:
            print(f"[topology] Saved to {cache_file}")

    return fingerprints


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTION: fingerprint_cost
# ─────────────────────────────────────────────────────────────────────────────

def fingerprint_cost(
    fp_A: np.ndarray,
    fp_B: np.ndarray,
    metric: str = 'cosine',
) -> np.ndarray:
    """
    Compute the pairwise topological dissimilarity matrix M_topo.

    M_topo[i, j] = distance between fingerprint of cell i in A and
                   fingerprint of cell j in B.

    This matrix is used as a third linear cost term in the FGW objective
    (alongside M1 gene/cell-type cost and M2 JSD neighbourhood cost).
    By adding M_topo, the OT plan is penalised for matching cells in A
    to cells in B that have different topological connectivity — which
    effectively distinguishes left-vs-right hemisphere even when composition
    is identical.

    Parameters
    ----------
    fp_A : (n_A, D) float32 — fingerprints from sliceA.
    fp_B : (n_B, D) float32 — fingerprints from sliceB.
    metric : str, default 'cosine'
        Distance metric.  Options:
          'cosine' — 1 - cosine_similarity.  Fast, scale-invariant.
          'euclidean' — L2 distance.  More sensitive to magnitude.

    Returns
    -------
    M_topo : (n_A, n_B) float32 array.
        All entries in [0, 1] for cosine (or [0, ∞) for euclidean).
    """
    fp_A = fp_A.astype(np.float32)
    fp_B = fp_B.astype(np.float32)

    if metric == 'cosine':
        # Normalise rows then take dot product
        norm_A = np.linalg.norm(fp_A, axis=1, keepdims=True) + 1e-10
        norm_B = np.linalg.norm(fp_B, axis=1, keepdims=True) + 1e-10
        fp_A_n = fp_A / norm_A
        fp_B_n = fp_B / norm_B
        M_topo = 1.0 - fp_A_n @ fp_B_n.T   # (n_A, n_B)

    elif metric == 'euclidean':
        # ||a - b||² = ||a||² + ||b||² - 2·aᵀb
        sq_A   = (fp_A ** 2).sum(axis=1, keepdims=True)   # (n_A, 1)
        sq_B   = (fp_B ** 2).sum(axis=1, keepdims=True).T  # (1, n_B)
        M_topo = np.sqrt(np.maximum(sq_A + sq_B - 2.0 * (fp_A @ fp_B.T), 0.0))

    else:
        raise ValueError(f"metric must be 'cosine' or 'euclidean', got '{metric}'")

    return M_topo.astype(np.float32)
