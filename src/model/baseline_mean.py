"""Mean-shift baseline: every perturbation is predicted as the average training delta.

Trains on K562 and is evaluated on RPE1 (cross-context).
"""

import argparse
import logging

import anndata as ad
import numpy as np

from src.utils.evaluation import EvalData, Evaluation

logger = logging.getLogger(__name__)


class MeanShift:
    name = "baseline_mean"

    def __init__(self, train_delta: np.ndarray, train_genes: np.ndarray):
        train_delta = np.asarray(train_delta, dtype=np.float64)
        if train_delta.ndim != 2 or train_delta.shape[1] != len(train_genes):
            raise ValueError("train_delta must be (perturbations x genes) matching train_genes")
        self.mean_delta = train_delta.mean(axis=0)
        self.gene_pos = {g: i for i, g in enumerate(train_genes)}

    def predict(self, data: EvalData) -> np.ndarray:
        # Align the training mean to the test genes by name; genes unseen in training get 0.
        genes = data.real.var_names.astype(str)
        idx = np.array([self.gene_pos.get(g, -1) for g in genes])
        row = np.where(idx >= 0, self.mean_delta[np.maximum(idx, 0)], 0.0)
        logger.info(
            "Predicting %d perturbations: %d of %d test genes shared with training (%d set to 0)",
            len(data.test_perts), (idx >= 0).sum(), len(genes), (idx < 0).sum(),
        )
        return np.tile(row, (len(data.test_perts), 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        default="datasets/pseudobulk/ReplogleWeissman2022_K562_essential.h5ad",
        help="training pseudobulk .h5ad file",
    )
    parser.add_argument(
        "--test",
        default="datasets/pseudobulk/ReplogleWeissman2022_rpe1.h5ad",
        help="test pseudobulk .h5ad file",
    )
    parser.add_argument("--output", default="results", help="results directory")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N test perturbations")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    train = ad.read_h5ad(args.train)
    train_delta = train.layers["delta"][train.obs["pass_qc"].to_numpy()]
    logger.info("Training: %d perturbations from %s", train_delta.shape[0], args.train)

    test_obs = ad.read_h5ad(args.test, backed="r").obs
    test_perts = test_obs.index[test_obs["pass_qc"]].tolist()
    if args.limit:
        test_perts = test_perts[: args.limit]
    logger.info("Test: %d perturbations from %s", len(test_perts), args.test)

    test_genes = ad.read_h5ad(args.test, backed="r").var_names.astype(str)
    shared_genes = [g for g in test_genes if g in set(train.var_names.astype(str))]
    logger.info("Shared genes between train and test: %d", len(shared_genes))

    model = MeanShift(train_delta, train.var_names.to_numpy(dtype=str))
    Evaluation(args.test, test_perts, outdir=args.output, genes=shared_genes).run(model)


if __name__ == "__main__":
    main()
