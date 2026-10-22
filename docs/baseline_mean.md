# Baseline: mean shift (`src/model/baseline_mean.py`)

The simplest trained baseline. It assumes every knockdown does the same thing:
the average of all knockdowns seen in training. There is no per-perturbation
reasoning at all — one vector is predicted for every test perturbation.

$$
\hat{\Delta}_{p,g} = \frac{1}{|T|}\sum_{q \in T} \Delta_{q,g}
\qquad \text{for every test perturbation } p
$$

Setup in this repo (cross-context): **train on K562, test on RPE1.**
Everything is pseudobulk-level; no single-cell data is used.

## Inputs

| Input | Default | Used for |
|---|---|---|
| `--train` pseudobulk `.h5ad` | `datasets/pseudobulk/ReplogleWeissman2022_K562_essential.h5ad` | training `delta` layer |
| `--test` pseudobulk `.h5ad` | `datasets/pseudobulk/ReplogleWeissman2022_rpe1.h5ad` | test rows, control stats, measured `delta`/`tstat` |
| `--output` | `results` | where metrics are written |
| `--limit N` | off | evaluate only the first N test perturbations (quick runs) |

## Algorithm

### Step 1 — Select training perturbations (K562)

Load the training pseudobulk file and keep only rows with `obs.pass_qc == True`.

- Kept: **1,787** of 2,057 K562 perturbations.
- Left out: 270 perturbations that failed QC (too few cells or weak knockdown),
  and the control row (`pass_qc` is False for it; its delta is 0 anyway).

### Step 2 — Extract the training effect matrix

```
train_delta = train.layers["delta"][pass_qc rows]      # 1787 × 8563
```

Each value `Δ_q,g` is the measured effect of perturbation *q* on gene *g*:
mean log1p-CP10k of *q* minus the K562 control mean (produced by
`src/data/pseudobulk.py`). Only this layer is read. The `log2fc`, `tstat`,
`frac_expr` layers and cell counts are **not used**.

### Step 3 — Average ("training")

```
mean_delta = train_delta.mean(axis=0)                  # one value per gene, length 8563
```

That vector is the entire model. Every training perturbation gets equal weight,
regardless of its cell count or effect size. After this step the code keeps
only two things: `mean_delta` and a gene-name → column lookup (`gene_pos`).
All per-perturbation information is discarded.

### Step 4 — Select test perturbations (RPE1)

Read the test file's `obs` (backed mode, no matrix loaded) and list the
`pass_qc` perturbations: **1,579** of 2,393. With `--limit N`, only the first N
(alphabetical order). Non-pass-QC perturbations are left out.

### Step 5 — Load the evaluation set and filter to clear effects

`Evaluation` loads the control row plus the test rows from the RPE1 pseudobulk
file, restricted to the **shared genes** (Step 6), then applies its DE-gene
filter (`min_de_genes=10`): a perturbation stays only if its measured DE table
has ≥10 significant genes (`p_adj < 0.05`, from `tstat`, BH-corrected per
perturbation) overall **and** ≥10 in each direction.

- Kept (full run): **1,213** of 1,579.
- Left out: 366 weak-effect perturbations. They are excluded because
  cell-eval2's DE metrics cannot score them (they only produced "not scored"
  warnings). This changes the evaluated set, so filtered and unfiltered
  scores are not comparable.

### Step 6 — Restrict to shared genes, then tile

The training mean lives on K562's 8,563 genes; the test set uses RPE1's 8,749.
Evaluation is restricted to the intersection, in RPE1 gene order:

- 7,226 shared genes → evaluated; the training mean is aligned to them by name.
- 1,523 RPE1 genes never measured in K562 → **excluded from evaluation**
  (no model trained on K562 could predict them).
- 1,337 K562 genes absent from RPE1 → unused.

The aligned vector is then tiled (copied) once per test perturbation, giving a
`1213 × 7226` prediction matrix in which **every row is identical**. The model
never looks at which perturbation it is predicting — not the target gene, not
the cell count, nothing.

### Step 7 — Prepare cell-eval2 inputs (in `Evaluation`)

- Predicted profiles = `clip(RPE1 ctrl_mean_log + pred_delta, 0)`. Note the
  control mean is **RPE1's own** — the baseline predicts an effect, and the
  effect is applied on top of the test context's control.
- Predicted t-statistic: Welch t using the measured RPE1 control variance for
  both groups (a model predicts only a mean, so its noise is borrowed).
- Two DE tables (`de_table`): `log2_fold_change = delta / ln 2`,
  p-values from the t-statistics (normal approximation), BH-adjusted per
  perturbation. The real table uses the measured `delta`/`tstat` layers.

### Step 8 — Score and report

cell-eval2 runs its full metric profile except `pearson_edistance` and
`clustering_agreement` (not implemented upstream). DE tables are supplied by
us, so no DE is computed by the library (the `de_wilcoxon_*` names are just
its DE metric family). Inside, cell-eval2 additionally drops each
perturbation's own target gene from DE scoring (the knockdown of the target
itself is known from the label and would inflate scores).

Outputs, under `results/baseline_mean/ReplogleWeissman2022_rpe1/`:

- `results.csv` — one row per (perturbation, metric),
- `agg_results.csv` — per-metric means over scored perturbations,
- headline metrics logged: `delta_pearson`, `expr_mse`, `pds_cosine`,
  `de_wilcoxon_precision_top50`.

## Summary: what is kept, what is left out

| Stage | Kept | Left out |
|---|---|---|
| Training rows | 1,787 pass-QC K562 perturbations | 270 failed-QC + control row |
| Training data | `delta` layer only | `log2fc`, `tstat`, `frac_expr`, `counts`, cell counts |
| "Model" | one mean vector (8,563 values) | every per-perturbation difference; no weighting by cells/effect |
| Test rows | 1,579 pass-QC RPE1 → 1,213 after DE filter | failed-QC and weak-effect perturbations |
| Genes | 7,226 shared genes | 1,523 RPE1-only genes (excluded from evaluation), 1,337 K562-only genes (unused) |
| Prediction | same row for all perturbations | target gene identity, perturbation-specific signal |
| Scoring | full cell-eval2 profile minus e-distance/clustering | single-cell level evaluation; target-gene rows inside DE metrics |

## Expected behaviour

- `pds_cosine` ≈ 0.5 (chance): identical predictions cannot tell perturbations
  apart, by construction.
- `delta_pearson` > 0 (~0.2): the average knockdown response (e.g. generic
  stress/essential-gene effects) is partly shared across perturbations and
  cell lines, so the mean correlates with real effects.
- `de_wilcoxon_precision_top50` near 0 but above the no-change baseline:
  genes that respond to *most* knockdowns appear in many real top-50 lists.

## Limitations

- Generic response only: it can never predict anything specific to a
  perturbation, so it is a floor for discrimination metrics, not a competitor.
- Cross-context assumption: the K562 average effect is applied verbatim to
  RPE1; cell-line-specific effects are invisible to it.
- 17% of RPE1 genes (1,523) are excluded from evaluation entirely, so scores
  are not comparable to a full-panel (all-8,749-gene) run.
- Equal weighting: a 50-cell perturbation counts as much as a 3,000-cell one.

## Run

```bash
python -m src.model.baseline_mean              # full: train K562, test RPE1
python -m src.model.baseline_mean --limit 50   # quick run
```
