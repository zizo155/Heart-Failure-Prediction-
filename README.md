# Heart Disease Prediction

Binary classification of heart disease from clinical measurements,
comparing Logistic Regression, Random Forest and SVM. Includes an
interactive Streamlit app that walks through each stage of the pipeline
and accepts an uploaded dataset.

## Dataset

[Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
from Kaggle — 918 patient records with 11 clinical features (age,
resting blood pressure, cholesterol, maximum heart rate, ST depression,
chest pain type, exercise-induced angina, ST slope, and others) and a
binary `HeartDisease` target.

After IQR-based outlier removal, 712 records remain, with a near-even
class split.

The CSV is not included in this repository. Download it from the link
above and place `heart.csv` in the project root.

## Pipeline

1. **Exploration** — distributions of numerical and categorical
   features, correlation matrix over factorised categoricals.
2. **Outlier removal** — IQR method (1.5×) applied to Age, RestingBP,
   Cholesterol, MaxHR and Oldpeak.
3. **Encoding** — categorical features one-hot encoded; numerical
   features standardised with `StandardScaler`.
4. **Feature engineering** — interaction term `Age_BP`, and a
   cholesterol ratio relative to the cohort mean.
5. **Modelling** — three classifiers evaluated with 5-fold
   cross-validated ROC AUC and a held-out 20% test set.

## Results

| Model | CV ROC AUC | Test ROC AUC | Test Accuracy |
|---|---|---|---|
| Random Forest | 0.926 ± 0.082 | 0.933 | 0.87 |
| Logistic Regression | 0.926 ± 0.069 | 0.921 | 0.85 |
| SVM | 0.927 ± 0.077 | 0.920 | 0.86 |

All three perform comparably, with Random Forest slightly ahead on the
test set. The narrow spread suggests the signal in this dataset is
largely linear — a simple logistic model captures nearly as much as
the ensemble.

Random Forest feature importances are plotted in the app; ST slope and
chest pain type dominate, consistent with their clinical role in
cardiac assessment.

## Streamlit app

`streamlit.py` presents the pipeline as five navigable sections — Data
Overview, Visualizations, Outlier Handling, Feature Engineering and
Modeling — with a sidebar uploader so a different dataset can be
substituted.

Run it with:

    streamlit run streamlit.py

## Setup

    pip install -r requirements.txt

## Files

| File | Purpose |
|---|---|
| `heart_failure.ipynb` | Full analysis, preprocessing and model comparison |
| `streamlit.py` | Interactive dashboard |
| `requirements.txt` | Dependencies |

## Notes

This is a coursework/portfolio project, not a clinical tool. The dataset
is an aggregation of several older cardiology studies and is not
representative of any current patient population.

IQR outlier removal discards roughly 22% of records. Some of these are
genuine data-quality problems (cholesterol recorded as 0), but others
are plausible extreme values that a clinician would want retained —
worth revisiting if the pipeline were extended.

---

**Zohreh Taghibakhshi** · [GitHub](https://github.com/zizo155)
