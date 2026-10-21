"""Effect measures for comparing groups of cells against a reference."""

import numpy as np


def delta(mean_log: np.ndarray, ref_log: np.ndarray) -> np.ndarray:
    """Difference of mean log expression: group minus reference."""
    return mean_log - ref_log


def log2fc(mean_cp: np.ndarray, ref_cp: np.ndarray, pseudocount: float = 0.1) -> np.ndarray:
    """log2 ratio of linear-scale means; the pseudocount keeps tiny values from blowing up."""
    return np.log2((mean_cp + pseudocount) / (ref_cp + pseudocount))


def welch_t(
    diff: np.ndarray, var_a: np.ndarray, n_a: np.ndarray, var_b: np.ndarray, n_b: float
) -> np.ndarray:
    """Welch t-statistic of `diff`; 0 where the standard error is 0.

    `diff` and `var_a` are (n_groups x n_genes), `n_a` is (n_groups,),
    `var_b` is (n_genes,) and `n_b` is the reference group's size.
    """
    se = np.sqrt(var_a / n_a[:, None] + var_b / n_b)
    return np.divide(diff, se, out=np.zeros_like(diff), where=se > 0)


def batch_matched_reference(
    ctrl_sum_b: np.ndarray,
    ctrl_n_b: np.ndarray,
    group_batch_n: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    """Per-group reference: control means per batch, weighted by each group's batch mix.

    `ctrl_sum_b` is (n_batches x n_genes), `ctrl_n_b` is (n_batches,),
    `group_batch_n` is (n_groups x n_batches). Batches without controls use `fallback`.
    """
    has_ctrl = ctrl_n_b > 0
    ctrl_mean_b = np.where(
        has_ctrl[:, None], ctrl_sum_b / np.maximum(ctrl_n_b, 1)[:, None], fallback
    )
    weights = group_batch_n / group_batch_n.sum(axis=1, keepdims=True)
    return weights @ ctrl_mean_b


def target_knockdown(
    mean_cp: np.ndarray, ref_cp: np.ndarray, delta: np.ndarray, target_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row knockdown of each row's own target gene column.

    Returns knockdown fraction (1 - mean_cp / ref_cp) and delta at the target.
    `target_idx` is -1 where a row has no target; those rows, and zero references, get NaN.
    """
    rows = np.arange(len(target_idx))
    has_target = target_idx >= 0
    cols = np.where(has_target, target_idx, 0)
    ref = ref_cp[rows, cols]
    ok = has_target & (ref > 0)
    kd = np.where(ok, 1.0 - mean_cp[rows, cols] / np.where(ok, ref, 1.0), np.nan)
    target_delta = np.where(has_target, delta[rows, cols], np.nan)
    return kd, target_delta


def count_strong_genes(tstat: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    """Number of genes per row with |t| above `threshold`."""
    return (np.abs(tstat) > threshold).sum(axis=1)
