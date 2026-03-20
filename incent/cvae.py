"""
cvae.py — Conditional VAE for Cross-Timepoint Expression Embedding
===================================================================
Learns a latent embedding of gene expression that is:

  (a) **Cell-type anchored** — cells of the same type cluster together in
      latent space, regardless of which developmental timepoint they come from.
  (b) **Batch/time invariant** — temporal changes in expression are factored
      out, leaving only true biological identity.

Why is this needed?
-------------------
For cross-timepoint alignment, we need a pairwise cost M_latent[i,j]
that measures how similar cell i (from timepoint t_A) is to cell j
(from timepoint t_B) *independent of the time difference*.

Using raw gene expression (cosine distance) fails because the same cell
type has different expression profiles at different developmental stages.
A neuron at E12 looks different from the same neuron at E16 in expression
space — so the cost matrix would penalise their matching even though they
ARE the correct correspondence.

Architecture
------------
Conditional VAE with cell-type label conditioning:
  Encoder:  (x, label_one_hot) → (μ, log σ²)     [recognition network]
  Decoder:  (z, label_one_hot) → x̂               [generative network]
  Training: ELBO + cell-type contrastive (triplet) loss

The cell-type conditioning means the decoder always knows what type to
reconstruct, so the latent z only needs to encode "what's different about
this specific cell from the prototype of its type" — i.e. individual
variation, not type identity.  The triplet loss further enforces that
cells of the same type (across timepoints) are closer in z than cells
of different types.

Public API
----------
INCENT_cVAE         — PyTorch model class
train_cvae(adatas, epochs, ...) → INCENT_cVAE
embed(adata, model)  → np.ndarray   [latent mean vectors, (n, latent_dim)]
latent_cost(adata_A, adata_B, model) → np.ndarray  [(n_A, n_B) M_latent]
"""

import numpy as np
from typing import List, Optional, Tuple
from anndata import AnnData


# ─────────────────────────────────────────────────────────────────────────────
# Helper: extract dense expression matrix
# ─────────────────────────────────────────────────────────────────────────────

def _to_dense(X) -> np.ndarray:
    """Convert sparse or dense matrix to float32 numpy array."""
    import scipy.sparse as sp
    if sp.issparse(X):
        return X.toarray().astype(np.float32)
    return np.asarray(X, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class _SpatialTranscriptomicsDataset:
    """
    PyTorch-compatible dataset wrapping multiple AnnData objects.

    Each item is (expression, cell_type_idx).
    The __getitem__ is called by DataLoader.

    Parameters
    ----------
    adatas : list of AnnData — one per timepoint (or per sample).
    shared_genes : list of str — only these genes are used.
    cell_type_map : dict — maps cell_type_string → integer index.
    """

    def __init__(
        self,
        adatas: List[AnnData],
        shared_genes: List[str],
        cell_type_map: dict,
    ):
        import torch
        exprs  = []
        ctypes = []
        for ad in adatas:
            X  = _to_dense(ad[:, shared_genes].X)
            # Log1p normalise (standard for count data)
            X  = np.log1p(X)
            # Standardise to zero mean, unit std per gene
            X  = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)
            exprs.append(X)
            ct = np.array([cell_type_map[c]
                           for c in ad.obs['cell_type_annot'].astype(str)])
            ctypes.append(ct)

        self.X  = torch.tensor(np.concatenate(exprs,  axis=0), dtype=torch.float32)
        self.ct = torch.tensor(np.concatenate(ctypes, axis=0), dtype=torch.long)
        self.n_genes = self.X.shape[1]
        self.n_types = len(cell_type_map)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.ct[idx]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class INCENT_cVAE:
    """
    Conditional Variational Autoencoder for spatiotemporal gene expression.

    This is a wrapper around PyTorch nn.Module that provides a clean
    train / embed / save / load interface without requiring PyTorch imports
    at the module level.

    Architecture
    ------------
    Encoder:  [x ; one_hot(label)] → FC(512) → ReLU → FC(256) → ReLU
                                   → μ ∈ ℝ^d,  log_σ² ∈ ℝ^d
    Decoder:  [z ; one_hot(label)] → FC(256) → ReLU → FC(512) → ReLU → FC(n_genes)
    Latent:   d = latent_dim (default 32)

    Loss
    ----
    L = -ELBO + λ_triplet · L_triplet
    where:
      ELBO = E[log p(x|z,label)] - KL(q(z|x,label) || N(0,I))
      L_triplet: for each anchor cell (type k), pull positives (same type k,
                 different timepoint) closer and push negatives (different type)
                 further.  Margin = 0.5.

    Parameters
    ----------
    n_genes : int — number of input genes.
    n_types : int — number of cell types (for conditioning).
    latent_dim : int, default 32 — dimension of the latent space.
    hidden_dim : int, default 256 — hidden layer width.
    lambda_triplet : float, default 1.0 — weight of the triplet loss.
    """

    def __init__(
        self,
        n_genes: int,
        n_types: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        lambda_triplet: float = 1.0,
    ):
        self.n_genes        = n_genes
        self.n_types        = n_types
        self.latent_dim     = latent_dim
        self.hidden_dim     = hidden_dim
        self.lambda_triplet = lambda_triplet
        self._model         = None    # built lazily when torch is available

    def _build_torch_model(self):
        """Lazily import torch and build the nn.Module."""
        import torch
        import torch.nn as nn

        n_genes    = self.n_genes
        n_types    = self.n_types
        latent_dim = self.latent_dim
        H          = self.hidden_dim
        cond_dim   = n_genes + n_types   # expression + one-hot cell type

        class _cVAE(nn.Module):
            def __init__(self):
                super().__init__()
                # ── Encoder ───────────────────────────────────────────────
                self.enc = nn.Sequential(
                    nn.Linear(cond_dim, H * 2),
                    nn.ReLU(),
                    nn.Linear(H * 2, H),
                    nn.ReLU(),
                )
                self.mu      = nn.Linear(H, latent_dim)
                self.log_var = nn.Linear(H, latent_dim)

                # ── Decoder ───────────────────────────────────────────────
                self.dec = nn.Sequential(
                    nn.Linear(latent_dim + n_types, H),
                    nn.ReLU(),
                    nn.Linear(H, H * 2),
                    nn.ReLU(),
                    nn.Linear(H * 2, n_genes),
                )

            def encode(self, x, label_oh):
                h       = self.enc(torch.cat([x, label_oh], dim=1))
                return self.mu(h), self.log_var(h)

            def reparameterise(self, mu, log_var):
                """Sample z = μ + ε·σ, ε ~ N(0,I) (reparameterisation trick)."""
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                return mu + eps * std

            def decode(self, z, label_oh):
                return self.dec(torch.cat([z, label_oh], dim=1))

            def forward(self, x, label_oh):
                mu, log_var = self.encode(x, label_oh)
                z           = self.reparameterise(mu, log_var)
                x_hat       = self.decode(z, label_oh)
                return x_hat, mu, log_var, z

        self._model = _cVAE()
        return self._model

    # ── Loss functions ────────────────────────────────────────────────────

    @staticmethod
    def _elbo_loss(x, x_hat, mu, log_var, beta: float = 1.0):
        """
        Evidence Lower BOund (ELBO) = reconstruction loss + KL divergence.

        Reconstruction: mean squared error (appropriate for standardised data).
        KL:  -0.5 · Σ (1 + log_σ² - μ² - σ²)
        """
        import torch
        recon = ((x - x_hat) ** 2).sum(dim=1).mean()
        kl    = -0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum(dim=1).mean()
        return recon + beta * kl

    @staticmethod
    def _triplet_loss(z, labels, margin: float = 0.5):
        """
        Batch semi-hard triplet loss on latent vectors.

        For each anchor i (type k):
          - Positive p: another cell of type k
          - Negative n: a cell of a different type
          - Loss: max(d(a,p) - d(a,n) + margin, 0)

        Semi-hard: pick negatives that are farther than positives but within margin.
        """
        import torch

        # Pairwise squared distances in latent space
        sq     = (z ** 2).sum(dim=1, keepdim=True)   # (B, 1)
        dists  = sq + sq.T - 2.0 * z @ z.T           # (B, B)
        dists  = torch.clamp(dists, min=0.0)

        same   = (labels.unsqueeze(0) == labels.unsqueeze(1))   # (B, B) bool

        loss   = torch.tensor(0.0, device=z.device)
        n_trip = 0

        for i in range(len(z)):
            pos_mask = same[i].clone()
            pos_mask[i] = False          # exclude self
            neg_mask = ~same[i]

            if not pos_mask.any() or not neg_mask.any():
                continue

            # Hard positive: furthest same-class cell
            d_pos = dists[i][pos_mask].max()
            # Semi-hard negatives: negative with d_neg > d_pos (in margin)
            d_negs = dists[i][neg_mask]
            semi   = d_negs[(d_negs > d_pos) & (d_negs < d_pos + margin)]
            if len(semi) == 0:
                d_neg = d_negs.min()     # fall back to hardest negative
            else:
                d_neg = semi.min()

            loss   = loss + torch.clamp(d_pos - d_neg + margin, min=0.0)
            n_trip += 1

        return loss / max(n_trip, 1)

    # ── Training ──────────────────────────────────────────────────────────

    def train(
        self,
        adatas: List[AnnData],
        epochs: int = 100,
        batch_size: int = 512,
        lr: float = 3e-4,
        beta_kl: float = 1.0,
        device: str = 'cpu',
        verbose: bool = True,
    ) -> 'INCENT_cVAE':
        """
        Train the cVAE on a list of AnnData objects (one per timepoint).

        All AnnData objects must share the same genes (var_names) and the same
        'cell_type_annot' label vocabulary.  The model trains on all of them
        jointly, so it learns to embed cells from any timepoint into the same
        latent space.

        Parameters
        ----------
        adatas : list of AnnData — one or more slices/timepoints.
        epochs : int, default 100 — number of full passes through the data.
        batch_size : int, default 512.
        lr : float, default 3e-4 — Adam learning rate.
        beta_kl : float, default 1.0 — weight of KL term in ELBO.
        device : str — 'cpu' or 'cuda'.
        verbose : bool — print epoch losses.

        Returns
        -------
        self — for method chaining.
        """
        import torch
        from torch.utils.data import DataLoader

        # ── Find shared genes and cell types across all AnnData objects ──
        shared_genes = list(adatas[0].var_names)
        for ad in adatas[1:]:
            shared_genes = [g for g in shared_genes if g in ad.var_names]
        if len(shared_genes) == 0:
            raise ValueError("No shared genes found across the provided AnnData objects.")

        all_types = sorted(set(
            ct for ad in adatas
            for ct in ad.obs['cell_type_annot'].astype(str).unique()
        ))
        cell_type_map = {ct: i for i, ct in enumerate(all_types)}

        self.shared_genes  = shared_genes
        self.cell_type_map = cell_type_map
        self.n_genes       = len(shared_genes)
        self.n_types       = len(all_types)

        # ── Build model ───────────────────────────────────────────────────
        model = self._build_torch_model().to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=lr)

        # ── Dataset ───────────────────────────────────────────────────────
        ds     = _SpatialTranscriptomicsDataset(adatas, shared_genes, cell_type_map)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            drop_last=True, num_workers=0)

        # ── Training loop ─────────────────────────────────────────────────
        n_types = len(all_types)
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            n_batches  = 0

            for x_batch, ct_batch in loader:
                x_batch  = x_batch.to(device)
                ct_batch = ct_batch.to(device)

                # One-hot encode cell types for conditioning
                label_oh = torch.zeros(len(ct_batch), n_types, device=device)
                label_oh.scatter_(1, ct_batch.unsqueeze(1), 1.0)

                # Forward pass
                x_hat, mu, log_var, z = model(x_batch, label_oh)

                # Losses
                elbo    = self._elbo_loss(x_batch, x_hat, mu, log_var, beta=beta_kl)
                triplet = self._triplet_loss(z, ct_batch)
                loss    = elbo + self.lambda_triplet * triplet

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

                total_loss += loss.item()
                n_batches  += 1

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"[cVAE] Epoch {epoch:4d}/{epochs}  "
                      f"loss={total_loss/n_batches:.4f}")

        self._model = model.eval()
        return self

    # ── Inference ─────────────────────────────────────────────────────────

    def embed(self, adata: AnnData, device: str = 'cpu') -> np.ndarray:
        """
        Encode cells from `adata` into the latent space.

        Returns the **mean** of the posterior distribution (not a sample),
        so embeddings are deterministic and reproducible.

        Parameters
        ----------
        adata : AnnData — must contain the genes from training.
        device : str — 'cpu' or 'cuda'.

        Returns
        -------
        z_mu : (n_cells, latent_dim) float32 — latent mean vectors.
            Cells of the same type (across timepoints) will cluster together.
        """
        import torch

        if self._model is None:
            raise RuntimeError("Model has not been trained yet.  Call .train() first.")

        model = self._model.to(device).eval()

        # Subset to training genes (in training order)
        adata_sub = adata[:, self.shared_genes]
        X         = _to_dense(adata_sub.X)
        X         = np.log1p(X)
        X         = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)

        ct_indices = np.array([self.cell_type_map.get(c, 0)
                               for c in adata.obs['cell_type_annot'].astype(str)])

        X_t  = torch.tensor(X, dtype=torch.float32).to(device)
        ct_t = torch.tensor(ct_indices, dtype=torch.long).to(device)

        n_types  = len(self.cell_type_map)
        label_oh = torch.zeros(len(X_t), n_types, device=device)
        label_oh.scatter_(1, ct_t.unsqueeze(1), 1.0)

        with torch.no_grad():
            mu, _ = model.encode(X_t, label_oh)

        return mu.cpu().numpy().astype(np.float32)

    def save(self, path: str):
        """
        Save model weights and metadata to a file.

        Parameters
        ----------
        path : str — path to save file (e.g. 'cvae_model.pt').
        """
        import torch
        torch.save({
            'model_state': self._model.state_dict(),
            'n_genes':     self.n_genes,
            'n_types':     self.n_types,
            'latent_dim':  self.latent_dim,
            'hidden_dim':  self.hidden_dim,
            'lambda_triplet': self.lambda_triplet,
            'shared_genes':  self.shared_genes,
            'cell_type_map': self.cell_type_map,
        }, path)
        print(f"[cVAE] Saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'INCENT_cVAE':
        """
        Load a previously saved INCENT_cVAE model.

        Parameters
        ----------
        path : str — path to the .pt file saved by .save().
        device : str — device to load onto.

        Returns
        -------
        INCENT_cVAE — ready for .embed() calls.
        """
        import torch
        ck = torch.load(path, map_location=device)
        obj = cls(
            n_genes=ck['n_genes'],
            n_types=ck['n_types'],
            latent_dim=ck['latent_dim'],
            hidden_dim=ck['hidden_dim'],
            lambda_triplet=ck['lambda_triplet'],
        )
        obj.shared_genes  = ck['shared_genes']
        obj.cell_type_map = ck['cell_type_map']
        model = obj._build_torch_model().to(device)
        model.load_state_dict(ck['model_state'])
        obj._model = model.eval()
        print(f"[cVAE] Loaded from {path}")
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC CONVENIENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def train_cvae(
    adatas: List[AnnData],
    latent_dim: int = 32,
    hidden_dim: int = 256,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 3e-4,
    lambda_triplet: float = 1.0,
    device: str = 'cpu',
    verbose: bool = True,
) -> INCENT_cVAE:
    """
    Train a conditional VAE on all provided slices/timepoints.

    This is the recommended entry point for training.  Pass all your
    MERFISH AnnData objects (one per timepoint) so the model sees all
    developmental stages and learns a time-invariant embedding.

    Parameters
    ----------
    adatas : list of AnnData — all slices to train on.
    latent_dim : int, default 32 — latent space dimensionality.
        Higher = more expressive but may overfit with few genes.
        For MERFISH (~250 genes) we recommend 16–32.
    hidden_dim : int, default 256 — hidden layer width.
    epochs : int, default 100.
    batch_size : int, default 512.
    lr : float, default 3e-4.
    lambda_triplet : float, default 1.0 — triplet loss weight.
        Increase if same cell types at different timepoints are not clustering.
    device : str, default 'cpu' — 'cuda' for GPU training.
    verbose : bool, default True.

    Returns
    -------
    model : INCENT_cVAE — trained model, ready for .embed().

    Examples
    --------
    >>> model = train_cvae([slice_E10, slice_E12, slice_E14, slice_E16])
    >>> model.save('brain_cvae.pt')
    >>> z_A = model.embed(slice_E12)
    >>> z_B = model.embed(slice_E16)
    """
    # Determine n_genes from first shared set
    shared_genes = list(adatas[0].var_names)
    for ad in adatas[1:]:
        shared_genes = [g for g in shared_genes if g in ad.var_names]

    all_types = sorted(set(
        ct for ad in adatas
        for ct in ad.obs['cell_type_annot'].astype(str).unique()
    ))

    model = INCENT_cVAE(
        n_genes=len(shared_genes),
        n_types=len(all_types),
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        lambda_triplet=lambda_triplet,
    )
    model.train(adatas, epochs=epochs, batch_size=batch_size,
                lr=lr, device=device, verbose=verbose)
    return model


def latent_cost(
    adata_A: AnnData,
    adata_B: AnnData,
    model: INCENT_cVAE,
    device: str = 'cpu',
) -> np.ndarray:
    """
    Compute the pairwise latent cosine distance matrix M_latent.

    M_latent[i, j] = 1 - cosine_similarity(z_i^A, z_j^B)

    This replaces M1 (the raw gene-expression cosine cost) for cross-
    timepoint alignment.  Because the cVAE is conditioned on cell type
    and trained with a triplet loss, cells of the same type have small
    M_latent[i,j] even if their raw expression at timepoints t_A and t_B
    looks quite different.

    Parameters
    ----------
    adata_A : AnnData — source slice (timepoint t_A).
    adata_B : AnnData — target slice (timepoint t_B).
    model : INCENT_cVAE — trained cVAE model.
    device : str — device for inference.

    Returns
    -------
    M_latent : (n_A, n_B) float32 array.  Values in [0, 2].

    Examples
    --------
    >>> M = latent_cost(slice_E12, slice_E16, model)
    >>> # Use M in place of cosine_dist_gene_expr in pairwise_align
    """
    z_A  = model.embed(adata_A, device=device)   # (n_A, d)
    z_B  = model.embed(adata_B, device=device)   # (n_B, d)

    # Normalise rows → unit sphere
    z_A  = z_A / (np.linalg.norm(z_A, axis=1, keepdims=True) + 1e-10)
    z_B  = z_B / (np.linalg.norm(z_B, axis=1, keepdims=True) + 1e-10)

    # Cosine distance = 1 - dot product
    M    = 1.0 - z_A @ z_B.T       # (n_A, n_B)
    return M.astype(np.float32)
