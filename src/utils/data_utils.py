"""Generic helpers for single-cell expression matrices."""

from collections.abc import Container

import numpy as np
import scipy.sparse as sp


def as_csr(block) -> sp.csr_matrix:
    """Return a block of X as float64 CSR, whatever its storage format."""
    if sp.issparse(block):
        return sp.csr_matrix(block, dtype=np.float64)
    return sp.csr_matrix(np.asarray(block, dtype=np.float64))


def is_raw_counts(X: sp.csr_matrix, n_sample: int = 200_000) -> bool:
    """True if the sampled non-zero values are whole numbers."""
    sample = X.data[:n_sample]
    return bool(np.allclose(sample, np.round(sample)))


def drop_empty_cells(X: sp.csr_matrix) -> tuple[sp.csr_matrix, np.ndarray]:
    """Remove cells with zero total counts; also return the kept-row mask."""
    ok = np.asarray(X.sum(axis=1)).ravel() > 0
    return X[ok], ok


def normalize_cp10k(X: sp.csr_matrix, target_sum: float = 1e4) -> sp.csr_matrix:
    """Scale each cell to `target_sum` total counts. Cells must have non-zero totals."""
    lib = np.asarray(X.sum(axis=1)).ravel()
    return (sp.diags(target_sum / lib) @ X).tocsr()


def log1p(X: sp.csr_matrix) -> sp.csr_matrix:
    out = X.copy()
    out.data = np.log1p(out.data)
    return out


def group_indicator(codes: np.ndarray, n_groups: int) -> sp.csr_matrix:
    """(n_groups x n_cells) matrix with G[i, j] = 1 if cell j is in group i; G @ X sums groups."""
    n = len(codes)
    return sp.csr_matrix(
        (np.ones(n, dtype=np.float64), (codes, np.arange(n))), shape=(n_groups, n)
    )


def mean_from_sums(sums: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Per-group mean from (n_groups x n_genes) sums and (n_groups,) cell counts."""
    return sums / n[:, None]


def var_from_sums(sums: np.ndarray, sumsq: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Per-group sample variance from sums and sums of squares; 0 for single-cell groups."""
    nn = n[:, None].astype(np.float64)
    mean = sums / nn
    var = (sumsq - nn * mean**2) / np.maximum(nn - 1, 1)
    return np.where(nn > 1, np.maximum(var, 0.0), 0.0)


def resolve_target(label: str, genes: Container[str]) -> str | None:
    """Map a perturbation label to a gene name in `genes`, or None.

    Tries the label itself, then its '_'/'-' separated tokens, which covers
    older labels like "1_AARS_P1P2_ENSG...".
    """
    if label in genes:
        return label
    for token in str(label).replace("-", "_").split("_"):
        if token in genes:
            return token
    return None
