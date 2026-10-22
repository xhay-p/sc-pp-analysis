# Pseudobulking Perturb-seq Data

## Overview

In a Perturb-seq experiment, each cell gets one CRISPRi guide that knocks down one gene, and then we measure the expression of all the genes in that cell. So for every target we have hundreds of cells, and for the non-targeting guides we have thousands of control cells.

Single-cell data is very noisy and sparse, so we do not use individual cells for the downstream analysis. Instead we pseudobulk the data: we collapse all the cells of one perturbation into a single expression profile, and then compare it with the control profile. The output is one row per perturbation, with the logFC values, a t-statistic for every gene, and QC metrics for the knockdown.

These pseudobulk profiles are what we use to train the perturbation prediction models, and to get the Differentially Expressed genes and run gene set enrichment analysis.

**Input:** scPerturb `.h5ad` file (raw UMI counts, one row per cell, perturbation label in `obs`)
**Output:** one `.h5ad` file with one row per perturbation + one control row, and a QC table

---

## Step 1: Read the data in chunks

We open the `.h5ad` file in backed mode. The cell and gene metadata (`obs`, `var`) is loaded in memory, but the expression matrix stays on disk. We read it 20,000 cells at a time.

---

## Step 2: Select the cells

1. We read the perturbation label of every cell (`perturbation` column) and find the control cells (label `control`).
2. We remove the cells that got more than one guide (`nperts > 1`), because their expression is a mix of two perturbation effects and we cannot say which effect is from which gene.
3. We give every perturbation an integer code. Control is always code 0.
4. If we want batch matching, we also give every batch (for example the 10x lane, `gemgroup`) an integer code.

---

## Step 3: Normalize each cell

For every cell in the chunk:

1. We check (only in the first chunk) that the values are whole numbers, i.e. raw counts. If the data is already normalized, we skip this step.
2. We compute the library size, i.e. the total counts of the cell, and remove the cells with zero counts.
3. We compute CP10k (counts per 10,000):

$$\text{CP10k}_{g} = \frac{\text{count}_{g}}{\sum_{g'} \text{count}_{g'}} \times 10{,}000$$

4. We take the log:

$$\text{logX}_{g} = \log(1 + \text{CP10k}_{g})$$

**Statistical concept – library size normalization**

Two cells can have a very different number of total counts only because one was sequenced deeper or captured more mRNA. If we do not correct this, a gene looks "higher" in the deeper cell even when the biology is the same. Dividing by the total counts converts the raw counts into proportions, and multiplying by 10,000 just puts them on a readable scale.

**Statistical concept – log transformation and pseudocount**

Expression values are very skewed: a few genes have huge counts and most have small counts. In the raw scale the variance also grows with the mean, so highly expressed genes dominate everything. The log compresses the large values, makes the variance more similar across genes (variance stabilization), and turns fold changes into differences (log(a/b) = log(a) − log(b)).

We use log(1 + x) and not log(x) because most single-cell values are 0, and log(0) is undefined. The "+1" is the *pseudocount*: it keeps zeros at zero.

---

## Step 4: Add up the cells per perturbation

For each chunk we build a sparse membership matrix **G** with one row per perturbation and one column per cell:

$$G_{ij} = \begin{cases} 1 & \text{if cell } j \text{ has perturbation } i \\ 0 & \text{otherwise} \end{cases}$$

Then one matrix multiplication **G · X** gives the sum of all cells for every perturbation. We add these sums to the accumulators:

| Accumulator | What it holds | Used later for |
| --- | --- | --- |
| `n` | number of cells per perturbation | mean, variance, standard error |
| `Σ logX` | sum of log expression | mean log expression, delta |
| `Σ logX²` | sum of squared log expression | variance, t-statistic |
| `Σ CP10k` | sum of linear-scale expression | log2FC, knockdown efficiency |
| `nnz` | number of cells with expression > 0 | fraction of cells expressing |
| `Σ counts` (optional) | sum of raw counts | DE with pyDESeq2 |

If we use batch matching, we also keep the control sums per batch, and how many cells of each perturbation came from each batch.

**Statistical concept – why aggregate at all**

A single cell captures only a small part of its mRNA, so many genes that are actually expressed show 0 counts (*dropout*). Averaging over many cells of the same perturbation cancels most of this technical noise, and leaves the shared biological signal. This is the same reason bulk RNA-seq is less noisy than single-cell: the averaging happens in the tube instead of in the code.

We keep these sums in float64 (high precision). Adding hundreds of thousands of small numbers in low precision lets rounding errors grow, and this matters most for Σx² in the variance formula.

---

## Step 5: Compute the mean and the variance

After the last chunk, for every perturbation *p* and gene *g*:

$$\bar{x}_{p,g} = \frac{\sum \text{logX}}{n_p}$$

$$s^2_{p,g} = \frac{\sum \text{logX}^2 - n_p \, \bar{x}_{p,g}^2}{n_p - 1}$$

We also compute the mean CP10k and the fraction of cells expressing each gene (nnz / n).

**Statistical concept – sample variance and n − 1**

The variance tells us how much the cells of one perturbation disagree with each other. We divide by n − 1 and not by n (Bessel's correction). We estimate the mean from the same cells, and the cells are always a bit closer to their own mean than to the true mean, so dividing by n would underestimate the variance. Dividing by n − 1 corrects this. For groups with only one cell, we cannot estimate a variance, so we set it to 0.

---

## Step 6: Build the control reference

The effect of a perturbation is always relative to the control cells. We have two options:

1. **Global control (default):** the mean of all control cells.
2. **Batch-matched control (`--batch-col`):** a weighted mean of the controls from the same batches as the perturbation cells.

For the batch-matched reference, if perturbation *p* has a fraction *w<sub>p,b</sub>* of its cells in batch *b*:

$$\text{ref}_{p,g} = \sum_b w_{p,b} \cdot \bar{x}_{\text{ctrl},b,g}$$

If 70% of the KO-A cells came from batch 3, then the batch 3 controls get 70% of the weight.

**Statistical concept – confounding and batch effects**

Cells from different 10x lanes or sequencing runs can differ for technical reasons. If a perturbation happens to sit mostly in one batch, and we compare it with controls from all batches, part of the "effect" is actually the batch. This is *confounding*. When we compare each perturbation with controls that have the same batch composition, the batch effect cancels out and only the perturbation effect remains. This is most important for perturbations with a weak effect, where the batch difference can be as big as the real signal.

---

## Step 7: Compute the perturbation effects

For every perturbation and gene we compute three effect measures:

**1. Delta (log-space difference)**

$$\Delta_{p,g} = \bar{x}_{p,g} - \text{ref}_{p,g}$$

This is the target for the prediction models, and it is the "delta" used in most perturbation benchmarks (for example the Pearson delta).

**2. log2FC (log of the ratio of means)**

$$\text{log2FC}_{p,g} = \log_2 \frac{\overline{\text{CP10k}}_{p,g} + 0.1}{\overline{\text{CP10k}}_{\text{ref},g} + 0.1}$$

This is the closest to the logFC values we get from a standard DE tool, and it is easy to explain to the biology teams ("2-fold up").

**3. Welch t-statistic**

$$t_{p,g} = \frac{\Delta_{p,g}}{\sqrt{\dfrac{s^2_{p,g}}{n_p} + \dfrac{s^2_{\text{ctrl},g}}{n_{\text{ctrl}}}}}$$

We use t to rank the genes for the DE gene list and as the ranking for gene set enrichment analysis.

**Statistical concept – mean of logs vs. log of means**

Delta and log2FC usually agree in direction, but they are not the same. Delta is the difference of the *average of the logs*, and log2FC is the log of the *ratio of the averages*. Because the log is a concave function, the average of the logs is always less than or equal to the log of the average (Jensen's inequality). The average of logs is less pulled by a few cells with very high expression, so delta is more robust. log2FC is on the scale the biologists are used to.

**Statistical concept – the pseudocount 0.1 in log2FC**

For genes with very low expression, the ratio of two tiny numbers can be huge (0.002 / 0.0001 = 20-fold), even when the difference is pure noise. Adding 0.1 to both means shrinks these ratios towards 1, so low-expressed genes do not show fake large fold changes.

**Statistical concept – standard error and the Welch t-test**

A delta of 0.5 means a lot if the cells agree with each other, and very little if they are all over the place. The standard error (the denominator) measures how precisely we know the delta. It gets smaller with more cells and bigger with more variance. The t-statistic is the effect divided by its uncertainty, so it is a signal-to-noise ratio. We use Welch's version because it does not assume that the perturbed cells and the control cells have the same variance. A knockdown often changes the variance too, so this assumption would be wrong.

**Statistical concept – why we use t only for ranking (pseudoreplication)**

The t-test assumes every cell is an independent replicate. But cells from the same experiment share the same culture, the same lane and the same guide, so they are not truly independent. Because of this *pseudoreplication*, the p-values from a cell-level t-test are far too small: with 100 cells almost every gene looks "significant". So we use t only to order the genes, never to call significance. For real significance, we use the summed raw counts with pyDESeq2 on biological replicates.

The control row is compared with itself, so its delta, log2FC and t are set to 0.

---

## Step 8: Knockdown QC

For every perturbation:

1. We find its target gene in the gene list (matching the label, or a part of the label for older formats like `1_AARS_P1P2_ENSG...`).
2. We compute the knockdown efficiency:

$$\text{knockdown} = 1 - \frac{\overline{\text{CP10k}}_{p,\text{target}}}{\overline{\text{CP10k}}_{\text{ref},\text{target}}}$$

A value of 0.85 means the guide removed 85% of the target mRNA.

3. We count the genes with |t| > 5 as a quick measure of how strong the perturbation effect is.
4. We flag a perturbation as `pass_qc` if it has at least 25 cells and at least 30% knockdown. If the target gene is not in the gene list, we do not filter on knockdown.

**Statistical concept – why QC before modeling**

If the guide did not knock down its target, any "effect" we see is just noise, and training a model on it teaches the model noise. The minimum number of cells matters for the same reason: the standard error falls with √n, so a mean from 5 cells is much less reliable than a mean from 100 cells. These thresholds make sure every row we train and test on is a real, well-measured perturbation.

---

## Step 9: Save the output

We write one `.h5ad` file:

| Slot | Content |
| --- | --- |
| `X` | mean log1p(CP10k) per perturbation |
| `layers["delta"]` | mean log expression − control reference |
| `layers["log2fc"]` | log2 fold change of mean CP10k (pseudocount 0.1) |
| `layers["tstat"]` | Welch t-statistic |
| `layers["frac_expr"]` | fraction of cells expressing the gene |
| `layers["counts"]` | summed raw counts (optional) |
| `obs` | number of cells, target gene, knockdown efficiency, `pass_qc` |
| `var` | control mean and fraction of control cells expressing each gene |
| `uns["pseudobulk"]` | the settings used for the run |

We save the results as float32 to keep the file small; all the calculations are done in float64. The QC table is also saved as a CSV.

---

## How this connects to the downstream analysis

1. Use the `pass_qc` rows to make the train/test splits (random split and pathway-holdout split).
2. Train the perturbation prediction models on `delta`.
3. Rank the genes by `tstat` (or `log2fc`) to get the top Differentially Expressed genes for each perturbation.
4. Use the same ranking for gene set enrichment analysis, and compare the pathways from the predicted and the measured profiles.