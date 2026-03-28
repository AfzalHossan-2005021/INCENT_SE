"""
Public INCENT-SE alignment entry points.

This cleaned module keeps:
  - `pairwise_align_se` for same-timepoint SE-aware alignment
  - `pairwise_align_spatiotemporal` for the maintained SEOT-backed
    spatiotemporal alignment path (`use_rapa=True`)
"""

import datetime
import os
import time
from typing import Optional, Tuple, Union

import numpy as np
import ot
from anndata import AnnData
from numpy.typing import NDArray

from ._seot_support import compute_objective_summary, objective_summary_tuple
from .contiguity import build_spatial_affinity, contiguity_gradient
from .core import _preprocess, _to_np
from .pose import apply_pose, estimate_pose
from .topology import compute_fingerprints, fingerprint_cost
from .utils import fused_gromov_wasserstein_incent


def pairwise_align_se(
    sliceA: AnnData,
    sliceB: AnnData,
    alpha: float,
    beta: float,
    gamma: float,
    radius: float,
    filePath: str,
    estimate_rotation: bool = True,
    pose_grid_size: int = 256,
    pose_sigma_px: float = 2.5,
    eta: float = 0.3,
    topo_n_bins: int = 16,
    topo_metric: str = "cosine",
    lambda_spatial: float = 0.1,
    contiguity_sigma: float = None,
    contiguity_k_nn: int = 20,
    use_rep: Optional[str] = None,
    G_init=None,
    a_distribution=None,
    b_distribution=None,
    numItermax: int = 6000,
    backend=ot.backend.NumpyBackend(),
    use_gpu: bool = False,
    return_obj: bool = False,
    return_objectives: bool = False,
    verbose: bool = False,
    gpu_verbose: bool = True,
    sliceA_name: Optional[str] = None,
    sliceB_name: Optional[str] = None,
    overwrite: bool = False,
    neighborhood_dissimilarity: str = "jsd",
    **kwargs,
) -> Union[NDArray, Tuple]:
    """
    Same-timepoint INCENT-SE alignment.

    Default return:
      `pi`

    `return_obj=True`:
      `(pi, pose_theta, pose_tx, pose_ty, pose_score)`

    `return_objectives=True`:
      `(pi, initial_obj_neighbor, initial_obj_gene_cos,
           final_obj_neighbor, final_obj_gene_cos)`

    `return_obj=True, return_objectives=True`:
      `(pi, pose_theta, pose_tx, pose_ty, pose_score, objective_summary_dict)`
    """
    del kwargs

    start = time.time()
    os.makedirs(filePath, exist_ok=True)

    log_name = (
        f"{filePath}/log_se_{sliceA_name}_{sliceB_name}.txt"
        if sliceA_name and sliceB_name
        else f"{filePath}/log_se.txt"
    )
    with open(log_name, "w") as logFile:
        logFile.write("pairwise_align_se -- INCENT-SE (same-timepoint)\n")
        logFile.write(f"{datetime.datetime.now()}\n")
        logFile.write(f"sliceA={sliceA_name}  sliceB={sliceB_name}\n")
        logFile.write(
            f"alpha={alpha}  beta={beta}  gamma={gamma}  "
            f"eta={eta}  lambda_spatial={lambda_spatial}  radius={radius}\n\n"
        )

        pose_theta = pose_tx = pose_ty = pose_score = 0.0
        if estimate_rotation:
            print("[INCENT-SE] Stage 1: Fourier-Mellin pose estimation ...")
            pose_theta, pose_tx, pose_ty, pose_score = estimate_pose(
                sliceA,
                sliceB,
                grid_size=pose_grid_size,
                sigma_px=pose_sigma_px,
                verbose=gpu_verbose,
            )
            sliceA = apply_pose(sliceA, pose_theta, pose_tx, pose_ty, inplace=False)
            logFile.write(
                f"Pose: theta={pose_theta:.2f}  tx={pose_tx:.2f}  "
                f"ty={pose_ty:.2f}  score={pose_score:.3f}\n\n"
            )
        else:
            logFile.write("Pose estimation skipped (estimate_rotation=False)\n\n")

        print("[INCENT-SE] Stage 2: Standard INCENT preprocessing ...")
        p = _preprocess(
            sliceA,
            sliceB,
            alpha,
            beta,
            gamma,
            radius,
            filePath,
            use_rep,
            G_init,
            a_distribution,
            b_distribution,
            numItermax,
            backend,
            use_gpu,
            gpu_verbose,
            sliceA_name,
            sliceB_name,
            overwrite,
            neighborhood_dissimilarity,
            logFile,
        )

        nx = p["nx"]
        M1 = p["M1"]
        M2 = p["M2"]
        D_A = p["D_A"]
        D_B = p["D_B"]
        a = p["a"]
        b = p["b"]
        sliceA_filt = p["sliceA"]
        sliceB_filt = p["sliceB"]

        if eta > 0.0:
            print("[INCENT-SE] Stage 3: Computing topological fingerprints ...")
            fp_A = compute_fingerprints(
                sliceA_filt,
                radius=radius,
                n_bins=topo_n_bins,
                cache_path=filePath,
                slice_name=f"{sliceA_name}_se" if sliceA_name else "A_se",
                overwrite=overwrite,
                verbose=gpu_verbose,
            )
            fp_B = compute_fingerprints(
                sliceB_filt,
                radius=radius,
                n_bins=topo_n_bins,
                cache_path=filePath,
                slice_name=f"{sliceB_name}_se" if sliceB_name else "B_se",
                overwrite=overwrite,
                verbose=gpu_verbose,
            )
            M_topo_np = fingerprint_cost(fp_A, fp_B, metric=topo_metric, use_gpu=use_gpu)
            M1_np = _to_np(M1).astype(np.float32)
            M_comb_np = M1_np + eta * M_topo_np.astype(np.float32)
            if use_gpu and isinstance(nx, ot.backend.TorchBackend):
                import torch as _torch

                M_combined = _torch.from_numpy(M_comb_np).cuda()
            else:
                M_combined = nx.from_numpy(M_comb_np.astype(np.float64))
            logFile.write(
                f"M_topo: shape={M_topo_np.shape}  min={M_topo_np.min():.4f}  "
                f"max={M_topo_np.max():.4f}  eta={eta}\n"
            )
        else:
            M_combined = M1
            logFile.write("Topological cost skipped (eta=0)\n")

        W_A = None
        D_B_dense = None
        if lambda_spatial > 0.0:
            print("[INCENT-SE] Stage 4: Building spatial affinity matrix W_A ...")
            sigma_c = contiguity_sigma if contiguity_sigma is not None else radius / 3.0
            W_A = build_spatial_affinity(
                sliceA_filt.obsm["spatial"].astype(np.float64),
                sigma=sigma_c,
                k_nn=contiguity_k_nn,
            )
            D_B_dense = _to_np(D_B)
            logFile.write(
                f"Contiguity: sigma={sigma_c:.1f}  k_nn={contiguity_k_nn}  "
                f"lambda={lambda_spatial}\n"
            )
        else:
            logFile.write("Contiguity regularisation skipped (lambda_spatial=0)\n")

        print("[INCENT-SE] Stage 5: Solving FGW ...")
        pi, _logw = fused_gromov_wasserstein_incent(
            M_combined,
            M2,
            D_A,
            D_B,
            a,
            b,
            G_init=p["G_init_t"],
            loss_fun="square_loss",
            alpha=alpha,
            gamma=gamma,
            log=True,
            numItermax=numItermax,
            verbose=verbose,
            use_gpu=p["use_gpu"],
        )
        pi_np = _to_np(pi).astype(np.float64)

        if lambda_spatial > 0.0 and W_A is not None:
            print("[INCENT-SE] Applying contiguity post-refinement ...")
            a_np = _to_np(a)
            for _ in range(10):
                grad = lambda_spatial * contiguity_gradient(
                    pi_np,
                    W_A,
                    D_B_dense,
                    use_gpu=use_gpu,
                )
                pi_np = np.maximum(pi_np - 0.05 * grad, 0.0)
                row_sums = pi_np.sum(axis=1, keepdims=True)
                pi_np = pi_np / np.maximum(row_sums, 1e-12) * a_np[:, None]

        objective_summary = compute_objective_summary(p, pi_np)
        logFile.write(
            f"Initial obj neighbour: {objective_summary['initial_obj_neighbor']:.6f}\n"
        )
        logFile.write(
            f"Initial obj gene cos:  {objective_summary['initial_obj_gene_cos']:.6f}\n"
        )
        logFile.write(
            f"Final obj neighbour:   {objective_summary['final_obj_neighbor']:.6f}\n"
        )
        logFile.write(
            f"Final obj gene cos:    {objective_summary['final_obj_gene_cos']:.6f}\n"
        )
        logFile.write(f"pi mass: {pi_np.sum():.4f}\n")
        logFile.write(f"Runtime: {time.time() - start:.1f}s\n")

    if p["use_gpu"] and isinstance(nx, ot.backend.TorchBackend):
        import torch as _torch

        _torch.cuda.empty_cache()

    print(f"[INCENT-SE] Done. Runtime={time.time() - start:.1f}s  pi_mass={pi_np.sum():.4f}")

    if return_obj:
        if return_objectives:
            return (
                pi_np,
                pose_theta,
                pose_tx,
                pose_ty,
                pose_score,
                objective_summary,
            )
        return pi_np, pose_theta, pose_tx, pose_ty, pose_score

    if return_objectives:
        return (pi_np, *objective_summary_tuple(objective_summary))

    return pi_np


def pairwise_align_spatiotemporal(
    sliceA: AnnData,
    sliceB: AnnData,
    alpha: float,
    beta: float,
    gamma: float,
    radius: float,
    filePath: str,
    use_rapa: bool = True,
    cross_timepoint: bool = True,
    use_lddmm: bool = False,
    max_em_iter: int = 50,
    reg_sinkhorn: float = 0.01,
    leiden_resolution: float = None,
    target_min_region_frac: float = 0.20,
    lambda_anchor: float = 2.0,
    lambda_target: float = 0.1,
    cvae_model: Optional[object] = None,
    cvae_path: Optional[str] = None,
    cvae_epochs: int = 100,
    cvae_latent_dim: int = 32,
    sigma_v: float = 300.0,
    lambda_v: float = 1.0,
    lddmm_lr: float = 0.01,
    lddmm_n_iter: int = 50,
    n_bcd_rounds: int = 3,
    kappa_growth: float = 0.1,
    estimate_rotation: bool = True,
    eta: float = 0.3,
    lambda_spatial: float = 0.1,
    use_rep: Optional[str] = None,
    G_init=None,
    a_distribution=None,
    b_distribution=None,
    numItermax: int = 2000,
    backend=ot.backend.NumpyBackend(),
    use_gpu: bool = False,
    return_obj: bool = False,
    return_objectives: bool = False,
    verbose: bool = False,
    gpu_verbose: bool = True,
    sliceA_name: Optional[str] = None,
    sliceB_name: Optional[str] = None,
    overwrite: bool = False,
    neighborhood_dissimilarity: str = "jsd",
    **kwargs,
) -> Union[NDArray, Tuple]:
    """
    Spatiotemporal alignment via the maintained SEOT path.

    Default return:
      `pi`

    `return_obj=True`:
      `(pi, sliceA_aligned, diagnostics_dict, residual_history)`

    `return_objectives=True`:
      `(pi, initial_obj_neighbor, initial_obj_gene_cos,
           final_obj_neighbor, final_obj_gene_cos)`

    When both flags are true, the returned `diagnostics_dict` includes the same
    objective summary under `diagnostics_dict["objective_summary"]`.
    """
    del use_lddmm, sigma_v, lambda_v, lddmm_lr, lddmm_n_iter
    del n_bcd_rounds, kappa_growth, estimate_rotation, eta
    del G_init, a_distribution, b_distribution, backend, leiden_resolution

    if not use_rapa:
        raise NotImplementedError(
            "This cleaned codebase keeps only the SEOT-backed "
            "pairwise_align_spatiotemporal path. Use use_rapa=True."
        )

    os.makedirs(filePath, exist_ok=True)

    from .seot import pairwise_align_seot

    need_diagnostics = return_obj or return_objectives
    result = pairwise_align_seot(
        sliceA=sliceA,
        sliceB=sliceB,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        radius=radius,
        filePath=filePath,
        max_em_iter=max_em_iter,
        reg_sinkhorn=reg_sinkhorn,
        target_min_region_frac=target_min_region_frac,
        lambda_anchor=lambda_anchor,
        lambda_spatial=lambda_spatial,
        lambda_target=lambda_target,
        cvae_model=cvae_model,
        cvae_path=cvae_path,
        cvae_epochs=cvae_epochs,
        cvae_latent_dim=cvae_latent_dim,
        cross_timepoint=cross_timepoint,
        use_rep=use_rep,
        numItermax=numItermax,
        use_gpu=use_gpu,
        gpu_verbose=gpu_verbose,
        verbose=verbose,
        sliceA_name=sliceA_name,
        sliceB_name=sliceB_name,
        overwrite=overwrite,
        neighborhood_dissimilarity=neighborhood_dissimilarity,
        return_diagnostics=need_diagnostics,
        **kwargs,
    )

    if not need_diagnostics:
        return result

    pi, diag = result

    if return_obj:
        return pi, diag["sliceA_aligned"], diag, diag["residual_history"]

    return (pi, *objective_summary_tuple(diag["objective_summary"]))
