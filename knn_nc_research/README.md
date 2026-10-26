# Revisiting KNN and Centroid-Based Classification

**How dataset structure influences comparative performance**

MSc in Computing thesis project, Atlantic Technological University (ATU), Galway.
Author: Md. Hossain Akash · Supervisor: Douglas Mota Dias

- **Thesis (PDF):** [thesis/thesis.pdf](thesis/thesis.pdf)
- **Screencast (5–10 minutes):** [link to the screencast]

## About the project

In 2022, Mladenova and Valova compared the classical k-nearest neighbour rule (KNN) with the nearest centroid neighbourhood rule (NCN) on 18 public datasets. They reported that NCN was significantly better on two of them. Their models were trained on raw, unscaled features.

This project asks whether that advantage belongs to the algorithm or to the preprocessing. It replicates the original comparison, repeats it with features standardised inside each training fold, and tests why the result changes. It then adds the nearest centroid classifier (NC) as a global reference model, to study how dataset structure affects local and global methods.

The main findings:

- On raw features, the original results replicate closely.
- After standardisation, most of the NCN advantage disappears. The size of the drop follows how unequal the feature scales are (Spearman ρ = 0.60, p = 0.019).
- A controlled experiment that changes only the scale of one feature reproduces the effect.
- Skewed classes hurt NC much more than KNN or NCN: measured against the Bayes-optimal accuracy, NC's shortfall is about twice that of KNN.

## Features

1. **NCN implementation** – the nearest centroid neighbourhood rule (Sánchez, Pla and Ferri, 1997), exact on real data, with an optional candidate pool for speed.
2. **Replication of the original study** – KNN vs NCN on 15 of its 18 datasets, using the original protocol (k = 5, raw features).
3. **Scaling experiment** – the same comparison with standardisation fitted on each training fold only, on identical folds.
4. **Sound statistics** – repeated stratified cross-validation, the corrected resampled t-test (Nadeau and Bengio, 2003), Bonferroni correction, Wilcoxon signed-rank test and Spearman/Pearson correlation.
5. **Structural profiling** – scale heterogeneity, skewness, kurtosis, separability, compactness and modality for every dataset.
6. **Controlled synthetic experiments** – a feature-scale sweep, a k sweep, a skewness sweep with separability held fixed (including the Bayes-optimal accuracy) and a separability sweep.
7. **Full reproducibility** – one command regenerates every number, table and figure in the thesis.

## Requirements

- Python 3.10 or later (developed and tested with Python 3.11)
- The packages in `requirements.txt`: NumPy, pandas, SciPy, scikit-learn and Matplotlib (seaborn is needed only for the notebook in `archive/`)

The project is a set of Python scripts. It needs no compilation and no deployment; it runs locally on any operating system that supports Python.

## Installation

```bash
git clone https://github.com/mdakash41/thesis-comparative-analysis-knn-nc
cd knn_nc_research
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the project

Run every command from the project root, so that the `src` package can be imported.

**Everything at once** (all experiments, then all figures):

```bash
python main.py
```

The full run can take half an hour or more, depending on the machine. Most of the time goes on the exact NCN for the Thyroid dataset (7,200 records) in experiments 1 and 3.

**One experiment at a time:**

```bash
python -m experiments.exp0_datasets                    # seconds
python -m experiments.exp1_scaling                     # slowest (Thyroid)
python -m experiments.exp2_mechanism
python -m experiments.exp3_structural_profile          # slow (Thyroid)
python -m experiments.exp4_skewness_fixed_separation   # under a minute
python -m experiments.exp5_separability_sweep          # under a minute
python -m experiments.make_figures                     # figures from existing CSV files
```

Each experiment prints a summary to the console and writes its results to `results/` as a CSV file. `make_figures` reads those files and writes the figures to `results/figures/`. All random seeds are fixed, so repeated runs give the same numbers.

## Project structure

```
knn_nc_research/
├── main.py                 Runs every experiment, then builds the figures
├── requirements.txt
├── thesis/                 The thesis as a PDF
├── data/                   14 UCI datasets as CSV files
├── src/
│   ├── paths.py            Locations of data/ and results/
│   ├── datasets.py         Dataset list, loading and preprocessing
│   ├── classifiers.py      NCN classifier
│   ├── evaluation.py       Repeated stratified CV and the corrected resampled t-test
│   └── measures.py         Structural measures
├── thesis/                 Thesis paper folder
│   ├── thesis.pdf          Thesis paper in pdf file
├── experiments/            One script per part of the thesis (see below)
├── results/                CSV outputs and results/figures/
└── archive/                Earlier exploratory work, not used for any thesis number
```

## Where each thesis result comes from

| Script | What it does | Thesis |
|---|---|---|
| `exp0_datasets` | Size of each dataset after cleaning, folds used, sizes in the original paper | Table 4.1 |
| `exp1_scaling` | KNN vs NCN on raw and standardised features, with significance tests and the heterogeneity correlation | Table 5.1, Figures 5.1–5.2 |
| `exp2_mechanism` | Synthetic sweep that scales one weak feature; k sweep on Wine | Figure 5.3 |
| `exp3_structural_profile` | Structural measures; KNN, NCN and NC on standardised features; correlations with the accuracy gap | Table 4.2, Figure 5.4 |
| `exp4_skewness_fixed_separation` | Skewness sweep with separability held fixed; Bayes-optimal accuracy at each level | Table 5.2, Figure 5.5 |
| `exp5_separability_sweep` | Symmetric Gaussian classes with the distance between them varied | Table 5.3 |
| `make_figures` | Builds every figure from the CSV files | All figures |

## Method notes

- k = 5 and Euclidean distance throughout, except in the k sweep.
- The scaler is always fitted on the training fold only, to avoid leaking information from the test fold.
- NCN is exact on real data. The synthetic experiments restrict its search to the 60 nearest candidates to keep run time down.
- When the smallest class has fewer than ten records, the number of folds is reduced to that size, with a minimum of two. This affects Glass (9 folds), Lung cancer (8), Lenses (4), Ecoli (2) and Absenteeism (2).
- Preprocessing: rows with missing values and non-numeric attributes are removed, and class labels are integer-encoded.

## Data

| File | Target column | Notes |
|---|---|---|
| `ecoli.csv`, `glass.csv`, `hepatitis.csv`, `ionosphere.csv`, `iris.csv`, `lenses.csv`, `lung_cancer.csv`, `soybean.csv`, `wine.csv` | `target` | UCI Machine Learning Repository |
| `absenteeism_at_work.csv` | `Absenteeism time in hours` | Semicolon-separated |
| `thyroid.csv` | `target` | UCI thyroid ANN data (`ann-train` and `ann-test` combined) |
| `cryotherapy.csv` | `Result_of_Treatment` | |
| `stars.csv` | `Type` | |
| `diabetes.csv` | `Outcome` | Pima Indians diabetes |
| loaded from scikit-learn | `target` | Breast Cancer Wisconsin (Diagnostic) |

Three datasets from the original study (SRBCT, Beavers and Algerian Forest Fires) are not included. Five datasets differ from the versions used in the original paper, as described in Section 4.1 of the thesis:

- **Thyroid** is the thyroid ANN data (7,200 records, 21 attributes); the original used new-thyroid (215 records, 5 attributes).
- **Absenteeism** uses absence time in hours as the target (19 classes); the original reports 28 classes, which matches the reason-for-absence attribute.
- **Lenses** has 3 of the 4 attributes.
- **Hepatitis** (80 of 155 records) and **Lung cancer** (27 of 32) contain only complete records.

## Archive

The `archive/` folder keeps earlier exploratory work for the record. It's archived after organizing every single experiment module wise.

- `controlled_dataset_analysis.ipynb` – the first synthetic KNN vs NC sweeps. Its skewness sweep changed class separation together with skewness, which is why it was redesigned as `exp4`.
- `verify_dataset.py` and `verify_*.png` – visual checks of those early synthetic datasets.

## Reference

T. Mladenova and I. Valova, "Comparative Analysis Between the Traditional and Nearest Centroid Methods in the K-Nearest Neighbor Algorithm," in *Proc. International Congress on Human-Computer Interaction, Optimization and Robotic Applications (HORA)*, 2022. doi: 10.1109/HORA55278.2022.9799808