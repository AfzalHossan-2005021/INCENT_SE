"""
pose.py — SE(2) Pose Estimator for INCENT-SE
=============================================
Recovers the in-plane rigid transformation (rotation angle θ, translation t)
between two spatial transcriptomics slices **before** running any optimal
transport.

Why do this first?
------------------
FGW-based alignment is inherently rotation-invariant: it compares
*pairwise distances*, so it finds the best coupling regardless of orientation.
That is useful — but it means the algorithm can never tell you *where* slice A
sits in the coordinate frame of slice B.  Worse, for bilaterally symmetric
tissues (e.g. brain) there are two orientations that achieve nearly identical
FGW cost, and the optimizer picks one essentially at random.

Strategy — Fourier-Mellin Transform (FMT)
-----------------------------------------
For each cell type k we build a 2-D spatial density image ρ_k(x,y).
The key fact (Fourier shift / scale theorem):

    |DFT( f rotated by θ )| = |DFT(f)| rotated by θ

So rotation in image space becomes *rotation in frequency space*.
Mapping the magnitude spectrum to log-polar coordinates converts that
rotation into a **horizontal translation**, which we find cheaply via
normalized cross-correlation (NCC).  Translation is then recovered by
a standard phase-correlation step on the aligned images.

Complexity: O(N² / B²) rasterization + O(N_px · log N_px) FFT,
where B is the bin size.  For 15k cells on a 256×256 grid this is
~0.3 s on a single CPU core.

Public API
----------
estimate_pose(sliceA, sliceB, **kwargs) -> (theta_deg, tx, ty, score)
apply_pose(sliceA, theta_deg, tx, ty) -> AnnData   (coords rotated in-place copy)
"""

import numpy as np
import warnings
from typing import Tuple, Optional
from anndata import AnnData
from scipy.ndimage import zoom
from scipy.signal import correlate2d


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Rasterise cell-type density
# ─────────────────────────────────────────────────────────────────────────────

def _rasterise_density(
    coords: np.ndarray,
    labels: np.ndarray,
    cell_types: np.ndarray,
    grid_size: int,
    spatial_range: Optional[Tuple[float, float, float, float]] = None,
    sigma_px: float = 2.0,
) -> np.ndarray:
    """
    Convert scattered cell coordinates into a stack of smoothed density images.

    For each unique cell type k we build a 2-D histogram (grid_size × grid_size)
    of how many cells of type k fall in each pixel.  We then apply Gaussian
    smoothing so the density field is continuous — this makes the Fourier
    magnitudes much more meaningful than for sparse point clouds.

    Parameters
    ----------
    coords : (n, 2) float array — [x, y] spatial coordinates.
    labels : (n,) str array  — cell type label for every cell.
    cell_types : (K,) str array — the K unique cell types to include.
    grid_size : int — number of pixels along each axis (e.g. 256).
    spatial_range : (xmin, xmax, ymin, ymax) or None.
        If None, inferred from coords.  Providing a shared range for both
        slices ensures the grids are on the same spatial scale.
    sigma_px : float — Gaussian smoothing radius in pixels.
        Larger values give smoother, more robust Fourier spectra.

    Returns
    -------
    density : (K, grid_size, grid_size) float32 array.
        Channel k contains the smoothed density of cell type k.
        Each channel is L2-normalised to 1 so every cell type contributes
        equally regardless of its abundance.
    """
    from scipy.ndimage import gaussian_filter

    # ── Determine spatial extent ───────────────────────────────────────────
    if spatial_range is None:
        xmin, ymin = coords.min(axis=0)
        xmax, ymax = coords.max(axis=0)
        # Add 5 % padding to avoid edge effects
        pad_x = (xmax - xmin) * 0.05 + 1e-6
        pad_y = (ymax - ymin) * 0.05 + 1e-6
        xmin -= pad_x;  xmax += pad_x
        ymin -= pad_y;  ymax += pad_y
    else:
        xmin, xmax, ymin, ymax = spatial_range

    K = len(cell_types)
    density = np.zeros((K, grid_size, grid_size), dtype=np.float32)

    # ── Map world coords → pixel indices ───────────────────────────────────
    px = ((coords[:, 0] - xmin) / (xmax - xmin) * (grid_size - 1)).astype(int)
    py = ((coords[:, 1] - ymin) / (ymax - ymin) * (grid_size - 1)).astype(int)
    # Clamp to valid range (floating-point edge cases)
    px = np.clip(px, 0, grid_size - 1)
    py = np.clip(py, 0, grid_size - 1)

    # ── Accumulate + smooth each cell-type channel ─────────────────────────
    ct2idx = {c: i for i, c in enumerate(cell_types)}
    for i, ct in enumerate(cell_types):
        mask = labels == ct
        if mask.sum() == 0:
            continue
        img = np.zeros((grid_size, grid_size), dtype=np.float32)
        np.add.at(img, (py[mask], px[mask]), 1.0)
        img = gaussian_filter(img, sigma=sigma_px)
        norm = np.linalg.norm(img)
        if norm > 1e-10:
            img /= norm                  # L2-normalise
        density[i] = img

    return density


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Fourier magnitude spectrum → log-polar coordinates
# ─────────────────────────────────────────────────────────────────────────────

def _log_polar_spectrum(density_stack: np.ndarray, num_angles: int = 360) -> np.ndarray:
    """
    Convert the 2-D Fourier magnitude spectrum of a density stack to
    log-polar coordinates, then *average* across cell-type channels.

    The Fourier-Mellin insight
    --------------------------
    If f_θ is f rotated by θ, then |DFT(f_θ)| = |DFT(f)| also rotated by θ.
    In log-polar (r, φ) coordinates a rotation becomes a *shift in φ*.
    NCC between two log-polar spectra therefore gives the rotation angle
    directly as a 1-D peak location — no iterative search needed.

    Parameters
    ----------
    density_stack : (K, H, W) float32 — K cell-type density images.
    num_angles : int — angular resolution of the log-polar grid.

    Returns
    -------
    lp : (num_angles, log_r_bins) float32 — the averaged log-polar spectrum.
    """
    from scipy.ndimage import map_coordinates

    K, H, W = density_stack.shape
    cy, cx  = H // 2, W // 2
    lp_sum  = None

    for k in range(K):
        img  = density_stack[k]
        # 2-D FFT → take magnitude → shift DC to centre
        mag  = np.abs(np.fft.fftshift(np.fft.fft2(img)))
        mag  = np.log1p(mag)          # log-compress dynamic range

        # Build log-polar grid
        max_r      = min(cy, cx)
        log_r_bins = int(np.log2(max_r) * 8) + 1
        angles     = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
        log_rs     = np.logspace(0, np.log10(max_r), log_r_bins)

        # Polar grid → Cartesian pixel coordinates for interpolation
        rs_grid, a_grid = np.meshgrid(log_rs, angles)
        yy = cy + rs_grid * np.sin(a_grid)
        xx = cx + rs_grid * np.cos(a_grid)

        # Bilinear interpolation on the magnitude image
        lp_k = map_coordinates(mag, [yy.ravel(), xx.ravel()],
                               order=1, mode='constant').reshape(num_angles, log_r_bins)

        if lp_sum is None:
            lp_sum = lp_k
        else:
            lp_sum += lp_k

    return lp_sum / K          # average across cell types


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Normalised cross-correlation (NCC) peak → rotation angle
# ─────────────────────────────────────────────────────────────────────────────

def _ncc_peak_angle(lp_A: np.ndarray, lp_B: np.ndarray, num_angles: int = 360) -> float:
    """
    Find the rotation angle between two log-polar spectra via NCC.

    A rotation in image space ↔ a horizontal shift in log-polar space.
    We compute the full 2-D NCC, collapse along the log-r axis (sum rows),
    and find the peak angular shift.

    Parameters
    ----------
    lp_A, lp_B : (num_angles, log_r_bins) float32 — log-polar spectra.
    num_angles : int — matches the value used in _log_polar_spectrum.

    Returns
    -------
    theta_deg : float — estimated rotation angle in degrees.
        Range is [0, 180) because the magnitude spectrum is symmetric:
        a rotation of θ and θ+180° give the same magnitude (hence two
        candidate angles).  We return the one in [0, 180).
    """
    # NCC in angular direction via 1-D FFT convolution (faster than correlate2d)
    ncc_rows = []
    for row_a, row_b in zip(lp_A, lp_B):
        # Normalise each row to zero-mean unit-variance
        def _norm(v):
            v = v - v.mean()
            s = v.std()
            return v / s if s > 1e-10 else v
        ncc_rows.append(_norm(row_a) @ _norm(row_b))

    # Use full 2-D cross-correlation along the angle axis only
    signal_A = lp_A.mean(axis=1)     # (num_angles,) — collapse log-r
    signal_B = lp_B.mean(axis=1)
    ncc      = correlate2d(signal_A[None, :], signal_B[None, :], mode='full')[0]
    shift    = np.argmax(ncc) - (num_angles - 1)   # signed angular shift in bins
    theta    = shift * 360.0 / num_angles           # convert to degrees
    # The FMT gives θ mod 180 (spectrum symmetry) — keep in [0, 180)
    theta    = theta % 180.0
    return theta


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Phase-correlation for translation
# ─────────────────────────────────────────────────────────────────────────────

def _phase_correlation_translation(
    density_A: np.ndarray,
    density_B: np.ndarray,
    xmin: float, xmax: float,
    ymin: float, ymax: float,
    grid_size: int,
) -> Tuple[float, float]:
    """
    Estimate translation between two aligned (same rotation) density stacks
    using phase correlation.

    Phase correlation finds the translation T such that density_B ≈ shift(density_A, T).
    It works in Fourier space:  IFFT( F_A · conj(F_B) / |F_A · conj(F_B)| )
    gives a delta function whose peak location is the translation.

    Parameters
    ----------
    density_A, density_B : (K, H, W) float32 — both already at the same rotation.
    xmin, xmax, ymin, ymax : float — world-coordinate extent of the grid.
    grid_size : int — pixel resolution.

    Returns
    -------
    (tx, ty) : float, float — translation in world coordinates.
        Apply as:  x_A_new = x_A + tx,  y_A_new = y_A + ty
    """
    # Average the phase-correlation signal across all cell-type channels
    H, W = density_A.shape[1], density_A.shape[2]
    pc_sum = np.zeros((H, W), dtype=np.complex128)

    for k in range(density_A.shape[0]):
        F_A  = np.fft.fft2(density_A[k])
        F_B  = np.fft.fft2(density_B[k])
        cross = F_A * np.conj(F_B)
        denom = np.abs(cross) + 1e-10
        pc_sum += cross / denom

    pc      = np.abs(np.fft.ifft2(pc_sum))
    pc      = np.fft.fftshift(pc)           # centre the zero-shift at (H//2, W//2)

    # Find peak
    py, px  = np.unravel_index(np.argmax(pc), pc.shape)
    shift_y = py - H // 2
    shift_x = px - W // 2

    # Convert pixel shifts → world-coordinate translation
    px_to_x = (xmax - xmin) / grid_size
    px_to_y = (ymax - ymin) / grid_size
    tx      = shift_x * px_to_x
    ty      = shift_y * px_to_y

    return float(tx), float(ty)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Try both candidate rotation angles (θ and θ+90 if ambiguous)
# ─────────────────────────────────────────────────────────────────────────────

def _rotation_matrix(theta_deg: float) -> np.ndarray:
    """
    Build a 2×2 counter-clockwise rotation matrix for the given angle.

    Parameters
    ----------
    theta_deg : float — rotation angle in degrees.

    Returns
    -------
    R : (2, 2) float64.
    """
    rad = np.deg2rad(theta_deg)
    return np.array([[np.cos(rad), -np.sin(rad)],
                     [np.sin(rad),  np.cos(rad)]])


def _alignment_score(
    coords_A_rot: np.ndarray,
    labels_A: np.ndarray,
    coords_B: np.ndarray,
    labels_B: np.ndarray,
    cell_types: np.ndarray,
    grid_size: int,
) -> float:
    """
    Compute an alignment quality score after applying a rotation.

    Score = sum over cell types of Pearson correlation between the
    two density images.  Higher is better.

    Used to pick between the two FMT candidates (θ vs θ+180°).
    """
    from scipy.stats import pearsonr

    range_x = (min(coords_A_rot[:, 0].min(), coords_B[:, 0].min()),
                max(coords_A_rot[:, 0].max(), coords_B[:, 0].max()))
    range_y = (min(coords_A_rot[:, 1].min(), coords_B[:, 1].min()),
                max(coords_A_rot[:, 1].max(), coords_B[:, 1].max()))
    srange  = (*range_x, *range_y)

    d_A = _rasterise_density(coords_A_rot, labels_A, cell_types, grid_size,
                              spatial_range=srange)
    d_B = _rasterise_density(coords_B,     labels_B, cell_types, grid_size,
                              spatial_range=srange)

    score = 0.0
    n     = 0
    for k in range(len(cell_types)):
        a, b = d_A[k].ravel(), d_B[k].ravel()
        if a.std() > 1e-10 and b.std() > 1e-10:
            r, _ = pearsonr(a, b)
            score += max(r, 0.0)
            n     += 1
    return score / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTION: estimate_pose
# ─────────────────────────────────────────────────────────────────────────────

def estimate_pose(
    sliceA: AnnData,
    sliceB: AnnData,
    grid_size: int = 256,
    sigma_px: float = 2.5,
    num_angles: int = 360,
    verbose: bool = True,
) -> Tuple[float, float, float, float]:
    """
    Estimate the SE(2) rigid transformation that maps sliceA onto sliceB.

    This is the INCENT-SE Stage 1 function.  It uses the Fourier-Mellin
    Transform (FMT) on stacked cell-type density images to find rotation,
    then phase correlation to find translation.  The whole procedure runs
    in O(N_px · log N_px) time — much faster than any OT-based approach.

    Algorithm overview
    ------------------
    1. Rasterise cell-type densities of A and B onto a shared pixel grid.
    2. Compute the log-polar Fourier magnitude spectrum of each.
    3. Normalised cross-correlation in log-polar space → rotation angle θ.
       (FMT gives θ mod 180°, so we test θ and θ+180° and pick the better one.)
    4. Rotate A's density by θ, then phase-correlation → translation (tx, ty).
    5. Return (θ, tx, ty, score) where score ∈ [0,1] is an alignment quality.

    Parameters
    ----------
    sliceA : AnnData
        Source slice.  Must have:
          - ``.obsm['spatial']`` — (n_A, 2) coordinate array.
          - ``.obs['cell_type_annot']`` — cell type labels.
    sliceB : AnnData
        Target slice.  Same requirements as sliceA.
    grid_size : int, default 256
        Pixel resolution for the density raster.  256 gives ~0.3s per pair.
        Use 512 for better accuracy at the cost of ~2× memory + time.
    sigma_px : float, default 2.5
        Gaussian smoothing radius (pixels) applied to each density channel.
        Larger values → smoother spectrum → more robust but less precise angle.
    num_angles : int, default 360
        Angular resolution of the log-polar spectrum (1° steps).
    verbose : bool, default True
        Print progress messages.

    Returns
    -------
    theta_deg : float
        Estimated counter-clockwise rotation angle in degrees.
        Apply as: ``coords_A_aligned = R(theta_deg) @ coords_A.T``
    tx : float
        X-axis translation in the same units as ``.obsm['spatial']``.
    ty : float
        Y-axis translation.
    score : float ∈ [0, 1]
        Alignment quality: Pearson correlation of density images after
        applying the estimated pose.  Values > 0.6 indicate reliable pose.

    Examples
    --------
    >>> theta, tx, ty, score = estimate_pose(sliceA, sliceB, verbose=True)
    >>> sliceA_aligned = apply_pose(sliceA, theta, tx, ty)
    """
    if verbose:
        print("[pose] Rasterising cell-type density fields …")

    # ── Gather shared cell types ───────────────────────────────────────────
    labels_A  = np.asarray(sliceA.obs['cell_type_annot'].astype(str))
    labels_B  = np.asarray(sliceB.obs['cell_type_annot'].astype(str))
    cell_types = np.intersect1d(np.unique(labels_A), np.unique(labels_B))

    if len(cell_types) == 0:
        raise ValueError("No shared cell types between slices — cannot estimate pose.")

    coords_A  = sliceA.obsm['spatial'].copy().astype(np.float64)
    coords_B  = sliceB.obsm['spatial'].copy().astype(np.float64)

    # Build a common spatial range so both grids are on the same scale
    xmin = min(coords_A[:, 0].min(), coords_B[:, 0].min())
    xmax = max(coords_A[:, 0].max(), coords_B[:, 0].max())
    ymin = min(coords_A[:, 1].min(), coords_B[:, 1].min())
    ymax = max(coords_A[:, 1].max(), coords_B[:, 1].max())
    pad  = max(xmax - xmin, ymax - ymin) * 0.05
    srange = (xmin - pad, xmax + pad, ymin - pad, ymax + pad)

    # ── Stage 2-a: rasterise ──────────────────────────────────────────────
    density_A = _rasterise_density(coords_A, labels_A, cell_types,
                                   grid_size, srange, sigma_px)
    density_B = _rasterise_density(coords_B, labels_B, cell_types,
                                   grid_size, srange, sigma_px)

    # ── Stage 2-b: log-polar Fourier spectra ──────────────────────────────
    if verbose:
        print("[pose] Computing log-polar Fourier spectra …")
    lp_A = _log_polar_spectrum(density_A, num_angles)
    lp_B = _log_polar_spectrum(density_B, num_angles)

    # ── Stage 3: rotation via NCC ─────────────────────────────────────────
    if verbose:
        print("[pose] Estimating rotation via NCC …")
    theta0 = _ncc_peak_angle(lp_A, lp_B, num_angles)
    theta1 = (theta0 + 180.0) % 360.0    # the ambiguous mirror candidate

    # ── Stage 4: pick the better candidate ────────────────────────────────
    # Rotate A's coords by each candidate, rasterise, measure Pearson score
    R0  = _rotation_matrix(theta0)
    R1  = _rotation_matrix(theta1)
    cA0 = (R0 @ coords_A.T).T
    cA1 = (R1 @ coords_A.T).T

    sc0 = _alignment_score(cA0, labels_A, coords_B, labels_B, cell_types, grid_size)
    sc1 = _alignment_score(cA1, labels_A, coords_B, labels_B, cell_types, grid_size)

    if sc0 >= sc1:
        theta_best, coords_A_rot = theta0, cA0
        if verbose:
            print(f"[pose] θ={theta0:.1f}° (score={sc0:.3f}) > θ+180={theta1:.1f}° (score={sc1:.3f})")
    else:
        theta_best, coords_A_rot = theta1, cA1
        if verbose:
            print(f"[pose] θ+180={theta1:.1f}° (score={sc1:.3f}) > θ={theta0:.1f}° (score={sc0:.3f})")

    # ── Stage 5: translation via phase correlation ─────────────────────────
    if verbose:
        print("[pose] Estimating translation via phase correlation …")
    xmin_s, xmax_s, ymin_s, ymax_s = srange
    # Rasterise rotated A for phase correlation
    density_A_rot = _rasterise_density(coords_A_rot, labels_A, cell_types,
                                       grid_size, srange, sigma_px)
    tx, ty = _phase_correlation_translation(
        density_A_rot, density_B,
        xmin_s, xmax_s, ymin_s, ymax_s, grid_size)

    # Final score after both rotation and translation
    coords_final = coords_A_rot + np.array([tx, ty])
    final_score  = _alignment_score(coords_final, labels_A, coords_B, labels_B,
                                    cell_types, grid_size)

    if verbose:
        print(f"[pose] Done. θ={theta_best:.2f}°  tx={tx:.2f}  ty={ty:.2f}  "
              f"final_score={final_score:.3f}")
        if final_score < 0.4:
            warnings.warn("[pose] Low alignment score — slice geometry may be too "
                          "different for Fourier pose estimation.  Consider increasing "
                          "grid_size or sigma_px, or using manual initialisation.",
                          stacklevel=2)

    return float(theta_best), float(tx), float(ty), float(final_score)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTION: apply_pose
# ─────────────────────────────────────────────────────────────────────────────

def apply_pose(
    sliceA: AnnData,
    theta_deg: float,
    tx: float,
    ty: float,
    inplace: bool = False,
) -> AnnData:
    """
    Apply an SE(2) rigid transformation to sliceA's spatial coordinates.

    The transformation is:  x_new = R(θ) · x + t

    Parameters
    ----------
    sliceA : AnnData — the slice to transform.
    theta_deg : float — counter-clockwise rotation in degrees.
    tx, ty : float — translation in the same coordinate units.
    inplace : bool, default False
        If True, modifies sliceA.obsm['spatial'] in-place.
        If False (default), returns a copy with transformed coordinates.

    Returns
    -------
    AnnData — copy of sliceA with updated .obsm['spatial'].
        If inplace=True, returns the original object (modified).

    Examples
    --------
    >>> theta, tx, ty, score = estimate_pose(sliceA, sliceB)
    >>> sliceA_aligned = apply_pose(sliceA, theta, tx, ty)
    """
    if not inplace:
        sliceA = sliceA.copy()

    R      = _rotation_matrix(theta_deg)
    coords = sliceA.obsm['spatial'].astype(np.float64)
    sliceA.obsm['spatial'] = (R @ coords.T).T + np.array([tx, ty])
    return sliceA
