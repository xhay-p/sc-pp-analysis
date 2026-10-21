"""Pseudobulk scPerturb Perturb-seq files into one perturbation x gene matrix."""

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from src.utils.analysis import (
    batch_matched_reference,
    count_strong_genes,
    delta,
    log2fc,
    target_knockdown,
    welch_t,
)
from src.utils.data_utils import (
    as_csr,
    drop_empty_cells,
    group_indicator,
    is_raw_counts,
    log1p,
    mean_from_sums,
    normalize_cp10k,
    resolve_target,
    var_from_sums,
)

logger = logging.getLogger(__name__)


def open_h5ad_backed(
    path: str, pert_col: str, batch_col: str | None = None, gene_col: str | None = None
) -> tuple[ad.AnnData, np.ndarray, dict[str, int]]:
    """Load obs and var into memory; keep the expression matrix on disk."""
    adata = ad.read_h5ad(path, backed="r")
    n_cells, n_genes = adata.shape
    logger.info("Opened %s: n_cells=%d, n_genes=%d", path, n_cells, n_genes)

    obs = adata.obs
    if pert_col not in obs.columns:
        sys.exit(f"obs has no column '{pert_col}'. Columns: {list(obs.columns)}")
    if batch_col and batch_col not in obs.columns:
        sys.exit(f"obs has no column '{batch_col}'. Columns: {list(obs.columns)}")

    if gene_col:
        if gene_col not in adata.var.columns:
            sys.exit(f"var has no column '{gene_col}'. Columns: {list(adata.var.columns)}")
        gene_names = adata.var[gene_col].astype(str).values
    else:
        gene_names = adata.var_names.astype(str).values
    gene_index = pd.Index(gene_names)
    gene_pos = {}
    for i, g in enumerate(gene_names):
        gene_pos.setdefault(g, i)
    if gene_index.has_duplicates:
        logger.warning(
            "%d duplicated gene names in var; knockdown QC uses the first occurrence",
            gene_index.duplicated().sum(),
        )

    return adata, gene_names, gene_pos


def encode(labels: pd.Series, first: str | None = None) -> tuple[np.ndarray, list[str]]:
    """Integer-code labels; `first`, if given, is always code 0."""
    names = sorted(labels.unique())
    if first is not None:
        names.remove(first)
        names.insert(0, first)
    codes = pd.Categorical(labels, categories=names).codes
    return np.asarray(codes), names


def select_cells(
    obs: pd.DataFrame, pert_col: str, control: str, batch_col: str | None = None
) -> dict:
    """Pick single-perturbation cells and integer-code perturbations (and batches).

    Codes cover every cell in obs; dropped cells get -1.
    """
    keep = np.ones(len(obs), dtype=bool)
    if "nperts" in obs:
        keep &= obs["nperts"].to_numpy() <= 1
        logger.info("Dropped %d multi-guide cells", (~keep).sum())

    labels = obs[pert_col].astype(str)[keep]
    if control not in set(labels):
        examples = labels.value_counts().index[:10].tolist()
        raise ValueError(
            f"No control cells labelled '{control}' in '{pert_col}'. "
            f"Pass --control. Most common labels: {examples}"
        )

    pert_codes = np.full(len(obs), -1, dtype=np.int64)
    pert_codes[keep], pert_names = encode(labels, first=control)
    logger.info("Kept %d cells across %d perturbations", keep.sum(), len(pert_names))

    batch_codes, batch_names = None, None
    if batch_col is not None:
        batch_codes = np.full(len(obs), -1, dtype=np.int64)
        batch_codes[keep], batch_names = encode(obs[batch_col].astype(str)[keep])
        logger.info("Found %d batches", len(batch_names))

    return {
        "keep": keep,
        "pert_codes": pert_codes,
        "pert_names": pert_names,
        "batch_codes": batch_codes,
        "batch_names": batch_names,
    }


def accumulate(
    adata: ad.AnnData,
    pert_codes: np.ndarray,
    n_perts: int,
    batch_codes: np.ndarray | None = None,
    n_batches: int = 0,
    chunk_size: int = 20_000,
    keep_counts: bool = False,
    x_is_lognorm: bool = False,
) -> dict[str, np.ndarray]:
    """Stream X in chunks and keep running per-perturbation x gene totals."""
    n_cells, n_genes = adata.shape
    totals = {
        "sum_log": np.zeros((n_perts, n_genes)),
        "sumsq_log": np.zeros((n_perts, n_genes)),
        "sum_cp": np.zeros((n_perts, n_genes)),
        "n_expr": np.zeros((n_perts, n_genes), dtype=np.int64),
        "n_cells": np.zeros(n_perts, dtype=np.int64),
    }
    if keep_counts and not x_is_lognorm:
        totals["sum_counts"] = np.zeros((n_perts, n_genes))
    if batch_codes is not None:
        totals["ctrl_sum_log_b"] = np.zeros((n_batches, n_genes))
        totals["ctrl_sum_cp_b"] = np.zeros((n_batches, n_genes))
        totals["ctrl_n_b"] = np.zeros(n_batches, dtype=np.int64)
        totals["pert_batch_n"] = np.zeros((n_perts, n_batches), dtype=np.int64)

    checked = x_is_lognorm
    n_empty = 0
    for start in range(0, n_cells, chunk_size):
        stop = min(start + chunk_size, n_cells)
        sel = pert_codes[start:stop] >= 0
        if not sel.any():
            continue
        X = as_csr(adata.X[start:stop])[sel]
        codes = pert_codes[start:stop][sel]
        batch = batch_codes[start:stop][sel] if batch_codes is not None else None

        if x_is_lognorm:
            lg = X
            cp = X.copy()
            cp.data = np.expm1(cp.data)
        else:
            if not checked and X.nnz:
                if not is_raw_counts(X):
                    sys.exit(
                        "X does not look like raw counts (non-integer values). "
                        "If it is already log1p-normalized, rerun with --x-is-lognorm."
                    )
                checked = True
            X, ok = drop_empty_cells(X)
            n_empty += int((~ok).sum())
            codes = codes[ok]
            if batch is not None:
                batch = batch[ok]
            cp = normalize_cp10k(X)
            lg = log1p(cp)

        G = group_indicator(codes, n_perts)
        totals["sum_log"] += (G @ lg).toarray()
        totals["sumsq_log"] += (G @ lg.multiply(lg)).toarray()
        totals["sum_cp"] += (G @ cp).toarray()
        totals["n_expr"] += (G @ (lg > 0).astype(np.float64)).toarray().astype(np.int64)
        totals["n_cells"] += np.bincount(codes, minlength=n_perts)
        if "sum_counts" in totals:
            totals["sum_counts"] += (G @ X).toarray()

        if batch is not None:
            np.add.at(totals["pert_batch_n"], (codes, batch), 1)
            is_ctrl = codes == 0
            if is_ctrl.any():
                Gb = group_indicator(batch[is_ctrl], n_batches)
                totals["ctrl_sum_log_b"] += (Gb @ lg[is_ctrl]).toarray()
                totals["ctrl_sum_cp_b"] += (Gb @ cp[is_ctrl]).toarray()
                totals["ctrl_n_b"] += np.bincount(batch[is_ctrl], minlength=n_batches)

        logger.info("Processed %d/%d cells", stop, n_cells)

    if n_empty:
        logger.info("Skipped %d cells with zero total counts", n_empty)
    return totals


def compute_stats(totals: dict[str, np.ndarray], pert_names: list[str]) -> dict:
    """Turn running totals into per-perturbation means and effects vs control (row 0).

    With batches, delta and log2fc use batch-matched controls, but the t-statistic's
    control variance comes from all control cells (per-batch variance is not tracked).
    """
    present = totals["n_cells"] > 0
    if not present[0]:
        sys.exit("No control cells survived filtering.")
    if not present.all():
        logger.info("Dropping %d perturbations with no usable cells", (~present).sum())
    names = [p for p, k in zip(pert_names, present) if k]

    n = totals["n_cells"][present]
    mean_log = mean_from_sums(totals["sum_log"][present], n)
    mean_cp = mean_from_sums(totals["sum_cp"][present], n)
    frac_expr = mean_from_sums(totals["n_expr"][present], n)
    var_log = var_from_sums(totals["sum_log"][present], totals["sumsq_log"][present], n)

    if "pert_batch_n" in totals:
        n_no_ctrl = int((totals["ctrl_n_b"] == 0).sum())
        if n_no_ctrl:
            logger.warning("%d batches have no control cells; global control used", n_no_ctrl)
        pert_batch_n = totals["pert_batch_n"][present]
        ref_log = batch_matched_reference(
            totals["ctrl_sum_log_b"], totals["ctrl_n_b"], pert_batch_n, mean_log[0]
        )
        ref_cp = batch_matched_reference(
            totals["ctrl_sum_cp_b"], totals["ctrl_n_b"], pert_batch_n, mean_cp[0]
        )
    else:
        ref_log = np.broadcast_to(mean_log[0], mean_log.shape)
        ref_cp = np.broadcast_to(mean_cp[0], mean_cp.shape)

    d = delta(mean_log, ref_log)
    lfc = log2fc(mean_cp, ref_cp)
    t = welch_t(d, var_log, n, var_log[0], n[0])
    d[0] = lfc[0] = t[0] = 0.0

    stats = {
        "pert_names": names,
        "n_cells": n,
        "mean_log": mean_log,
        "mean_cp": mean_cp,
        "frac_expr": frac_expr,
        "ref_cp": ref_cp,
        "delta": d,
        "log2fc": lfc,
        "tstat": t,
    }
    if "sum_counts" in totals:
        stats["sum_counts"] = totals["sum_counts"][present]
    return stats


def build_qc_table(
    stats: dict, gene_pos: dict[str, int], min_cells: int, min_knockdown: float
) -> pd.DataFrame:
    """Per-perturbation QC: target gene, knockdown, effect size and pass_qc flag."""
    names = stats["pert_names"]
    targets = [None] + [resolve_target(p, gene_pos) for p in names[1:]]
    target_idx = np.array([gene_pos[t] if t else -1 for t in targets])
    kd, target_delta = target_knockdown(
        stats["mean_cp"], stats["ref_cp"], stats["delta"], target_idx
    )

    qc = pd.DataFrame(
        {
            "n_cells": stats["n_cells"].astype(int),
            "is_control": np.arange(len(names)) == 0,
            "target_gene": targets,
            "target_in_var": target_idx >= 0,
            "knockdown_frac": kd,
            "target_delta": target_delta,
            "n_tstat_abs_gt_5": count_strong_genes(stats["tstat"], 5.0),
        },
        index=pd.Index(names, name="perturbation"),
    )
    qc["pass_qc"] = (
        ~qc["is_control"]
        & (qc["n_cells"] >= min_cells)
        & (~qc["target_in_var"] | (qc["knockdown_frac"] >= min_knockdown))
    )
    return qc


def write_output(
    stats: dict, qc: pd.DataFrame, gene_names: np.ndarray, output: Path, settings: dict
) -> Path:
    """Write the pseudobulk .h5ad and the QC table as <output stem>_qc.csv."""
    output.parent.mkdir(parents=True, exist_ok=True)

    var_index = pd.Index(gene_names.astype(str), name="gene")
    if var_index.has_duplicates:
        var_index = ad.utils.make_index_unique(var_index)
    var = pd.DataFrame(
        {
            "ctrl_mean_log": stats["mean_log"][0],
            "ctrl_mean_cp10k": stats["mean_cp"][0],
            "ctrl_frac_expr": stats["frac_expr"][0],
        },
        index=var_index,
    )

    layers = {k: stats[k].astype(np.float32) for k in ("delta", "log2fc", "tstat", "frac_expr")}
    if "sum_counts" in stats:
        layers["counts"] = stats["sum_counts"].astype(np.float32)

    out = ad.AnnData(
        X=stats["mean_log"].astype(np.float32), obs=qc, var=var, layers=layers
    )
    out.uns["pseudobulk"] = settings
    out.write_h5ad(output, compression="gzip")

    qc_path = output.with_name(output.stem + "_qc.csv")
    qc.to_csv(qc_path)
    logger.info("Wrote %s (%d x %d) and %s", output, out.n_obs, out.n_vars, qc_path.name)
    return qc_path


def log_summary(qc: pd.DataFrame, min_cells: int, min_knockdown: float) -> None:
    perts = qc[~qc["is_control"]]
    logger.info("Perturbations: %d, control cells: %d", len(perts), qc["n_cells"].iloc[0])
    logger.info("Median cells per perturbation: %.0f", perts["n_cells"].median())
    logger.info("Target found in var: %.1f%%", 100 * perts["target_in_var"].mean())
    kd = perts["knockdown_frac"].dropna()
    if len(kd):
        logger.info(
            "Median knockdown: %.1f%%; knockdown >= %.0f%%: %.1f%% of targets in var",
            100 * kd.median(), 100 * min_knockdown, 100 * (kd >= min_knockdown).mean(),
        )
    logger.info(
        "Pass QC: %d of %d (min_cells=%d, min_knockdown=%.2f)",
        perts["pass_qc"].sum(), len(perts), min_cells, min_knockdown,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="scPerturb .h5ad file")
    parser.add_argument(
        "--output",
        default=None,
        help="pseudobulk .h5ad file (default: datasets/pseudobulk/<input name>.h5ad)",
    )
    parser.add_argument("--pert-col", default="perturbation", help="obs column with perturbation labels")
    parser.add_argument("--control", default="control", help="label of control cells")
    parser.add_argument("--batch-col", default=None, help="optional obs column with batch labels")
    parser.add_argument("--gene-col", default=None, help="optional var column with gene names")
    parser.add_argument("--chunk-size", type=int, default=20_000, help="cells read per chunk")
    parser.add_argument("--keep-counts", action="store_true", help="also sum raw counts")
    parser.add_argument(
        "--x-is-lognorm", action="store_true", help="X is already log1p-normalized"
    )
    parser.add_argument(
        "--min-cells", type=int, default=50, help="minimum cells for pass_qc"
    )
    parser.add_argument(
        "--min-knockdown", type=float, default=0.3, help="minimum target knockdown for pass_qc"
    )
    args = parser.parse_args()
    output = Path(args.output or Path("datasets/pseudobulk") / Path(args.input).name)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    adata, gene_names, gene_pos = open_h5ad_backed(
        args.input, args.pert_col, args.batch_col, args.gene_col
    )
    cells = select_cells(adata.obs, args.pert_col, args.control, args.batch_col)
    totals = accumulate(
        adata,
        cells["pert_codes"],
        len(cells["pert_names"]),
        cells["batch_codes"],
        len(cells["batch_names"] or []),
        args.chunk_size,
        args.keep_counts,
        args.x_is_lognorm,
    )
    adata.file.close()
    stats = compute_stats(totals, cells["pert_names"])

    n_perts = len(stats["pert_names"]) - 1
    median_cells = int(np.median(stats["n_cells"][1:]))
    median_strong = int(np.median(count_strong_genes(stats["tstat"][1:], 5.0)))
    logger.info(
        "Stats: %d perturbations x %d genes, %d control cells, "
        "median %d cells/perturbation, median %d genes with |t| > 5",
        n_perts, stats["delta"].shape[1], stats["n_cells"][0], median_cells, median_strong,
    )

    qc = build_qc_table(stats, gene_pos, args.min_cells, args.min_knockdown)
    settings = {
        "source_file": Path(args.input).name,
        "pert_col": args.pert_col,
        "control_label": args.control,
        "batch_col": args.batch_col or "",
        "normalization": "already log1p" if args.x_is_lognorm else "CP10k + log1p",
        "min_cells": args.min_cells,
        "min_knockdown": args.min_knockdown,
    }
    write_output(stats, qc, gene_names, output, settings)
    log_summary(qc, args.min_cells, args.min_knockdown)


if __name__ == "__main__":
    main()
