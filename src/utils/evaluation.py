import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from cell_eval2 import EvalConfig, aggregate_metrics, compute_metrics
from cell_eval2.catalog import PROFILES
from cell_eval2.config import DEParams
from scipy import stats

from src.utils.analysis import welch_t

logger = logging.getLogger(__name__)

SKIP_METRICS = ("pearson_edistance", "clustering_agreement")
HEADLINE_METRICS = ("delta_pearson", "expr_mse", "pds_cosine", "de_wilcoxon_precision_top50")


@dataclass
class EvalData:
    """Real pseudobulk rows: control first, then the test perturbations."""

    real: ad.AnnData
    pert_col: str
    control: str

    @property
    def test_perts(self) -> list[str]:
        return self.real.obs_names[1:].tolist()


class Predictor(Protocol):
    name: str

    def predict(self, data: EvalData) -> np.ndarray:
        """Predicted delta, (n_test_perts x n_genes) in log1p-CP10k space, in test_perts order."""


def de_table(perts: np.ndarray, genes: np.ndarray, delta: np.ndarray, tstat: np.ndarray) -> pl.DataFrame:
    """Long DE table in cell-eval2's schema.

    log2_fold_change is delta on the log2 scale, so real and predicted tables share one
    definition. p-values come from t (normal approximation), BH-adjusted per perturbation.
    """
    p_value = 2 * stats.norm.sf(np.abs(tstat))
    p_adj = stats.false_discovery_control(p_value, axis=1)
    return pl.DataFrame(
        {
            "target": np.repeat(perts, len(genes)),
            "feature": np.tile(genes, len(perts)),
            "log2_fold_change": (delta / np.log(2)).ravel(),
            "p_value": p_value.ravel(),
            "p_adj": p_adj.ravel(),
        }
    )


class Evaluation:
    """Score a model's pseudobulk predictions against held-out perturbations."""

    def __init__(
        self,
        pseudobulk_path: str,
        test_perts: list[str],
        outdir: str = "results",
        min_de_genes: int = 10,
        genes: list[str] | None = None,
    ):
        self.data = self.load(pseudobulk_path, test_perts, min_de_genes, genes)
        self.dataset = Path(pseudobulk_path).stem
        self.outdir = Path(outdir)

    @staticmethod
    def load(
        pseudobulk_path: str, test_perts: list[str], min_de_genes: int = 10, genes: list[str] | None = None
    ) -> EvalData:
        pb = ad.read_h5ad(pseudobulk_path)
        settings = pb.uns["pseudobulk"]
        missing = sorted(set(test_perts) - set(pb.obs_names))
        if missing:
            raise ValueError(f"{len(missing)} test perturbations not in {pseudobulk_path}: {missing[:5]}")

        real = pb[[settings["control_label"], *test_perts]].copy()
        if genes is not None:
            missing_genes = sorted(set(genes) - set(real.var_names))
            if missing_genes:
                raise ValueError(f"{len(missing_genes)} genes not in {pseudobulk_path}: {missing_genes[:5]}")
            real = real[:, genes].copy()
            logger.info("Restricted to %d genes", real.n_vars)
        if min_de_genes > 0:
            real = Evaluation._filter_clear_effects(real, min_de_genes)
        logger.info("Loaded %d test perturbations x %d genes", len(real) - 1, real.n_vars)
        return EvalData(real, settings["pert_col"], settings["control_label"])

    @staticmethod
    def _filter_clear_effects(real: ad.AnnData, min_de_genes: int) -> ad.AnnData:
        """Keep perturbations with >= min_de_genes significant genes, overall and per direction.

        Significant = p_adj < 0.05 from the measured t-statistics (same test as de_table).
        These are the perturbations cell-eval2's DE metrics can score; the rest only
        produce 'not scored' warnings and drop out of the aggregates.
        """
        delta = np.asarray(real.layers["delta"][1:])
        tstat = np.asarray(real.layers["tstat"][1:])
        p_adj = stats.false_discovery_control(2 * stats.norm.sf(np.abs(tstat)), axis=1)
        sig = p_adj < 0.05
        keep = (
            (sig.sum(axis=1) >= min_de_genes)
            & ((sig & (delta > 0)).sum(axis=1) >= min_de_genes)
            & ((sig & (delta < 0)).sum(axis=1) >= min_de_genes)
        )
        if not keep.any():
            raise ValueError(f"No test perturbation has >= {min_de_genes} DE genes per direction")
        logger.info(
            "DE-gene filter: kept %d of %d perturbations (>= %d significant genes per direction)",
            keep.sum(), len(keep), min_de_genes,
        )
        return real[np.concatenate([[True], keep])].copy()

    def predict(self, model: Predictor) -> np.ndarray:
        pred_delta = np.asarray(model.predict(self.data))
        expected = (len(self.data.test_perts), self.data.real.n_vars)
        if pred_delta.shape != expected:
            raise ValueError(f"{model.name} predicted shape {pred_delta.shape}, expected {expected}")
        return pred_delta

    def prepare(self, pred_delta: np.ndarray) -> tuple[ad.AnnData, ad.AnnData, pl.DataFrame, pl.DataFrame]:
        """Pred/real AnnData (one row per perturbation) and their DE tables.

        The predicted t-statistic borrows its noise from the measured control variance,
        since a model predicts only a mean.
        """
        real = self.data.real
        genes = real.var_names.to_numpy(dtype=str)
        labels = real.obs_names.to_numpy(dtype=str)
        n_pert = real.obs["n_cells"].to_numpy()[1:]
        n_ctrl = real.obs["n_cells"].iloc[0]
        ctrl_mean = real.var["ctrl_mean_log"].to_numpy()
        ctrl_var = real.var["ctrl_var_log"].to_numpy()

        pred_X = np.vstack([ctrl_mean, np.clip(ctrl_mean + pred_delta, 0, None)])
        pred_t = welch_t(pred_delta, ctrl_var, n_pert, ctrl_var, n_ctrl)

        pred = self._as_anndata(pred_X, labels, genes)
        real_ad = self._as_anndata(real.X, labels, genes)
        de_pred = de_table(labels[1:], genes, pred_delta, pred_t)
        de_real = de_table(labels[1:], genes, real.layers["delta"][1:], real.layers["tstat"][1:])
        return pred, real_ad, de_pred, de_real

    def score(
        self, pred: ad.AnnData, real: ad.AnnData, de_pred: pl.DataFrame, de_real: pl.DataFrame
    ) -> pl.DataFrame:
        """Full cell-eval2 profile minus SKIP_METRICS; one row per (perturbation, metric)."""
        targets = self.data.real.obs["target_gene"].iloc[1:]
        config = EvalConfig(
            metrics=[m for m in PROFILES["full"] if m not in SKIP_METRICS],
            pert_col=self.data.pert_col,
            control=self.data.control,
            input_type="lognorm",
            de=DEParams(backend="pdex"),
            target_gene_map={p: t if isinstance(t, str) else None for p, t in targets.items()},
        )
        return compute_metrics(pred, real, config=config, de_pred=de_pred, de_real=de_real)

    def report(self, results: pl.DataFrame, model_name: str) -> pl.DataFrame:
        """Write results.csv and agg_results.csv, log the headline metrics, return the aggregate."""
        agg = aggregate_metrics(results)
        outdir = self.outdir / model_name / self.dataset
        outdir.mkdir(parents=True, exist_ok=True)
        results.write_csv(outdir / "results.csv")
        agg.write_csv(outdir / "agg_results.csv")

        values = dict(zip(agg["metric"], agg["mean"]))
        for metric in HEADLINE_METRICS:
            logger.info("%s | %s = %.4f", model_name, metric, values.get(metric, np.nan))
        logger.info("Wrote %s", outdir)
        return agg

    def run(self, model: Predictor) -> pl.DataFrame:
        pred_delta = self.predict(model)
        results = self.score(*self.prepare(pred_delta))
        return self.report(results, model.name)

    def _as_anndata(self, X: np.ndarray, labels: np.ndarray, genes: np.ndarray) -> ad.AnnData:
        obs = pd.DataFrame({self.data.pert_col: labels}, index=[f"row{i}" for i in range(len(labels))])
        return ad.AnnData(X=np.asarray(X, dtype=np.float32), obs=obs, var=pd.DataFrame(index=genes))
