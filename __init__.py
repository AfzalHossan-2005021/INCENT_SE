"""
__init__.py — INCENT-SE package
================================
INCENT-SE extends the original INCENT with:

  Same-timepoint alignment (pairwise_align_se):
    - Fourier-Mellin SE(2) pose estimation
    - Topological fingerprint cost for bilateral symmetry disambiguation
    - Spatial contiguity regularisation for realistic partial overlap

  Cross-timepoint alignment (pairwise_align_spatiotemporal):
    - All of the above, plus:
    - Conditional VAE for drift-corrected expression embeddings
    - LDDMM diffeomorphic spatial deformation (BCD joint optimisation)

Original INCENT functions are unchanged and still exported.
"""

# ── Original INCENT (unchanged) ───────────────────────────────────────────────
from .incent import (
    pairwise_align,
    pairwise_align_unbalanced,
    neighborhood_distribution,
    cosine_distance,
    _preprocess,
    _to_np,

    fused_gromov_wasserstein_incent,
    jensenshannon_divergence_backend,
    pairwise_msd,
    to_dense_array,
    extract_data_matrix,

# ── INCENT-SE: new alignment functions ───────────────────────────────────────
    pairwise_align_se,
    pairwise_align_spatiotemporal,

# ── INCENT-SE: pose estimation ────────────────────────────────────────────────
    estimate_pose,
    apply_pose,

# ── INCENT-SE: topological fingerprints ──────────────────────────────────────
    compute_fingerprints,
    fingerprint_cost,

# ── INCENT-SE: spatial contiguity regulariser ────────────────────────────────
    build_spatial_affinity,
    augment_fgw_gradient,
    contiguity_regulariser,
    contiguity_gradient,
    estimate_overlap_fraction,

# ── INCENT-SE: cross-timepoint cVAE ──────────────────────────────────────────
    INCENT_cVAE,
    train_cvae,
    latent_cost,

# ── INCENT-SE: LDDMM deformation ─────────────────────────────────────────────
    LDDMMDeformation,
    estimate_deformation,
    deformed_distances,
    estimate_growth_vector,
)

__all__ = [
    # Original INCENT
    'pairwise_align',
    'pairwise_align_unbalanced',
    'neighborhood_distribution',
    'cosine_distance',
    '_preprocess',
    '_to_np',
    'fused_gromov_wasserstein_incent',
    'jensenshannon_divergence_backend',
    'pairwise_msd',
    'to_dense_array',
    'extract_data_matrix',
    # INCENT-SE alignment
    'pairwise_align_se',
    'pairwise_align_spatiotemporal',
    # Pose
    'estimate_pose',
    'apply_pose',
    # Topology
    'compute_fingerprints',
    'fingerprint_cost',
    # Contiguity
    'build_spatial_affinity',
    'augment_fgw_gradient',
    'contiguity_regulariser',
    'contiguity_gradient',
    'estimate_overlap_fraction',
    # cVAE
    'INCENT_cVAE',
    'train_cvae',
    'latent_cost',
    # LDDMM
    'LDDMMDeformation',
    'estimate_deformation',
    'deformed_distances',
    'estimate_growth_vector',
]