# Single-Cell Perturbation Prediction

In-silico perturbation prediction on pseudobulk Perturb-seq. Models train from scratch in minutes: no-change, mean shift, PCA-ridge, MLP/kNN on co-expression embeddings, and K562→RPE1 cross-context. A small app on top showcases downstream analysis.

## Setup

Requires Python 3.11+ and [Poetry](https://python-poetry.org/). The virtual environment is created in `.venv/`.

```bash
poetry env use python3.11
poetry install
poetry run python ...   # or: source .venv/bin/activate
```

## Data

Replogle et al. 2022 genome-wide Perturb-seq [1] (K562 essential and RPE1), using the harmonized h5ad files from scPerturb [2] on [Zenodo](https://doi.org/10.5281/zenodo.13350497) [3]:

- `ReplogleWeissman2022_K562_essential.h5ad`
- `ReplogleWeissman2022_rpe1.h5ad`

Optional stretch: 2025 Virtual Cell Challenge H1 data from the Arc Virtual Cell Atlas.

## Split

Hold out 20% of perturbations. Keep those with a clear effect (enough DE genes) in the test set.

## Evaluation

[cell-eval](https://github.com/ArcInstitute/cell-eval) [4]

## Layout

```text
sc-pp-analysis/
├── app/                 # downstream analysis demo
├── src/
│   ├── data/            # pre-processing
│   ├── model/           # perturbation models
│   └── utils/           # evaluation, analysis, model_utils, data_utils
├── pyproject.toml       # Poetry dependencies
└── README.md
```

## Models

## References

1. Replogle, J. M. et al. Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq. *Cell* **185**, 2559–2575.e28 (2022). [doi:10.1016/j.cell.2022.05.013](https://doi.org/10.1016/j.cell.2022.05.013)
2. Peidli, S. et al. scPerturb: harmonized single-cell perturbation data. *Nature Methods* **21**, 531–540 (2024). [doi:10.1038/s41592-023-02144-y](https://doi.org/10.1038/s41592-023-02144-y)
3. Peidli, S. et al. scPerturb Single-Cell Perturbation Data: RNA and protein h5ad files (v1.4). Zenodo (2024). [doi:10.5281/zenodo.13350497](https://doi.org/10.5281/zenodo.13350497)
4. Arc Institute. cell-eval: evaluation suite for perturbation prediction models. [github.com/ArcInstitute/cell-eval](https://github.com/ArcInstitute/cell-eval)
