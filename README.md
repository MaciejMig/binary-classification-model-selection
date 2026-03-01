# Binary Classification (Artificial Dataset) — model selection & CV

Course project for *Introduction to Machine Learning*.  
Goal: train a binary classifier and generate **probabilities for class 1** for the test set.

## Results (from cross-validation)

Using **5-fold Stratified CV** and selecting hyperparameters via **GridSearchCV** (scoring: balanced accuracy),
the best-performing model was **ExtraTrees**.

- Best CV balanced accuracy (after tuning): **0.857**
- Out-of-fold ROC AUC (class 1): **0.925**

These numbers are documented in the included project report: [`reports/report.pdf`](reports/report.pdf).

## Approach (high level)

1. Data checks: no missing values, roughly balanced classes.
2. Quick EDA + correlation scan (|corr| ≥ 0.95) to detect redundant features.
3. Model comparison with CV:
   - Logistic Regression (elastic-net, scaled)
   - Linear SVM + probability calibration (scaled)
   - HistGradientBoosting
   - ExtraTrees
4. Hyperparameter tuning with GridSearchCV.
5. Model analysis: ROC curve (out-of-fold), permutation importance on a validation split.
6. Train final model on full train data and export test probabilities.

## Repository structure

```
.
├─ notebooks/               # exploratory notebook
├─ src/                     # scripts (reproducible pipeline)
├─ reports/                 # PDF report
├─ docs/                    # short problem statement
├─ data/raw/                # place CSV files here (not tracked by git)
└─ outputs/                 # generated predictions
```

## Reproducibility

### 1) Create environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Put data in `data/raw/`

This repo does **not** include the dataset. Create `data/raw/` and place the following files there:

- `artifical_train_data.csv`
- `artifical_train_labels.csv`
- `artifical_test_data.csv`

### 3) Generate submission file

```bash
python src/make_submission.py --data-dir data/raw --out outputs/333077_artifical_prediction.txt
```

The output format is: one probability per line (10 decimal places).

## Notes

- Random seed used across experiments: `RANDOM_STATE = 67`.
- For linear models, scaling is done via `Pipeline(StandardScaler(), ...)` to avoid data leakage during CV.


A sample prediction file is provided in `examples/sample_prediction.txt`.
