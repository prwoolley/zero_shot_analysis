# Inherent limitations of zero-shot mutation prediction in protein engineering

Phillip R. Woolley, Aaron L. Feller, Andrew D. Ellington, and Claus O. Wilke  
The University of Texas at Austin

This repository contains the analysis code, curated data, and figures for our study of zero-shot mutation prediction in protein engineering. We evaluate protein foundation models using ProteinGym, FireProtDB, and curated datasets comparing experimental measurements, phenotypes, and new-to-nature functions.

## Repository organization

- [`data/`](data/): Dataset summary tables and curated new-to-nature data, sequences, and structures. See the [dataset reference guide](data/README.md) for field descriptions and source references.
- [`scripts/`](scripts/): Model inference scripts and utilities for preparing sequence indices, alignments, and structure inputs.
- [`figures_scripts_notebooks/`](figures_scripts_notebooks/): Analysis notebooks and scripts, with separate directories for Figures 1-5.
- [`outputs/csvs/`](outputs/csvs/): Processed results used to generate the figures.
- [`images/`](images/): Figure files and associated tables.

## Usage

The analyses use Python with NumPy, pandas, SciPy, scikit-learn, Matplotlib, and seaborn. Jupyter is needed for the notebooks. Model inference requires additional model-specific packages, such as PyTorch, Transformers, or the ESM package.

To work with the existing results, use the tables in [`outputs/csvs/`](outputs/csvs/) and the corresponding figure scripts. Run figure scripts from their own directories so that relative paths resolve correctly. For example, from the repository root:

```bash
cd figures_scripts_notebooks/figure1_scripts
python make_figure1_plots.py
```

For model inference, use the appropriate script in [`scripts/`](scripts/). Each script lists its arguments with `--help`, once its dependencies are installed. For example:

```bash
python scripts/ESM2.py --help
```

Recomputing the full analyses requires the raw ProteinGym and FireProtDB datasets and model outputs, which are not included here. Check the input paths at the top of each analysis script before running it, and obtain the relevant model weights and dependencies before running inference.

## Citation

If you use this code or the curated data, please cite:

Woolley, P. R., Feller, A. L., Ellington, A. D., and Wilke, C. O. (2026). *Inherent limitations of zero-shot mutation prediction in protein engineering*. bioRxiv. https://doi.org/10.64898/2026.06.04.730121

Please also cite the original datasets and models where applicable.
