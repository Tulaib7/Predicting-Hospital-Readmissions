# Predicting Hospital Readmissions (MIMIC-IV)

Predicting 30-day hospital readmission for patients with kidney-related diagnoses, using the
MIMIC-IV clinical database. Covers cohort construction from raw admission records, feature
engineering, three classifiers, and a Streamlit dashboard for inspecting individual patients and
the cohort as a whole.

30-day readmission is a standard quality-of-care measure and a large cost driver, which gives it a
direct operational use: flagging patients who need more support behind their discharge plan.

---

## Results

Kidney cohort: 261,347 diagnosis rows across 111,891 stays and 47,919 patients.
Positive rate 25.8%. Threshold 0.30.

**As originally built**

| Model | Precision | ROC AUC | PR AUC | Flag rate |
|---|---|---|---|---|
| Logistic Regression | 0.464 | 0.701 | 0.446 | 22.3% |
| Random Forest | 0.416 | 0.777 | 0.521 | 46.7% |
| XGBoost + SMOTE | 0.375 | 0.767 | 0.493 | 64.1% |

**Corrected: leaky feature removed, patient-grouped split**

| Model | Precision | ROC AUC | PR AUC | Flag rate |
|---|---|---|---|---|
| Logistic Regression | 0.429 | 0.675 | 0.394 | 20.7% |
| Random Forest | 0.376 | 0.739 | 0.435 | 45.6% |
| XGBoost + SMOTE | 0.334 | 0.736 | 0.427 | 70.5% |

The original feature set included `readmit`, a count of prior admissions carrying the same
diagnosis code, and used a random train/test split. The cohort averages 2.3 stays per patient, so
that split placed the same patients on both sides. Correcting both costs 0.04 ROC AUC and 0.09 PR
AUC. **The corrected figures are the ones to cite.** Random Forest remains the best model. Both
configurations are reproducible via toggles in notebook 03. See [LIMITATIONS.md](LIMITATIONS.md).

Accuracy is deliberately absent. At a 25.8% positive rate, predicting "no readmission" for everyone
scores well on accuracy while being useless.

At the inherited 0.30 threshold the Random Forest flags 45.6% of the cohort, too many to be
operationally useful. The dashboard exposes the threshold as a slider so the recall / false-alarm
tradeoff can be inspected directly.

---

## Method

**Cohort.** `admissions`, `patients` and `diagnoses_icd` are joined on `subject_id` / `hadm_id`,
then restricted to diagnoses whose ICD long title matches kidney-related terms (kidney, renal,
nephr-, dialysis, glomerul-). Acute kidney failure was among the top diagnoses associated with
readmission, so the cohort widens from that single code to the disease area.

**Labels.** Admissions are sorted per patient and the gap between one discharge and the next
admission is computed. `readmission_under_30_days` is 1 when that gap is 30 days or less.

**Features.**

| Feature | Source |
|---|---|
| `anchor_age`, `gender` | demographics |
| `los_hours` | discharge minus admission |
| `admission_location`, `insurance` | administrative context |
| `hospital_expire_flag` | in-hospital death |
| `previous_admissions_count`, `visit_order` | patient history, derived from visit sequence |
| `severity` | rule-based score, 1 to 5, from the diagnosis text |
| `readmit` | prior visits carrying the same ICD code (leaky, see LIMITATIONS) |

MIMIC-IV shifts admission dates into the future to protect privacy, so `visit_order` encodes the
sequence of a patient's stays without relying on the absolute dates being real.

**Severity scoring.** A keyword lexicon maps diagnosis descriptions to a 1-5 score, walking from
most to least severe and returning on the first match, so "Chronic kidney disease, Stage IV
(severe)" scores 4 rather than 3. Unmatched text defaults to 1. It is a lookup, not a learned
model, and it is only as good as the ICD long titles.

**Models.** Each classifier sits in a pipeline: `ColumnTransformer` (one-hot for categoricals,
standardisation for numerics), then `SimpleImputer`, then the estimator. Fitting preprocessing
inside the pipeline keeps test-set statistics out of the training fold. XGBoost adds SMOTE, also
inside the pipeline, so synthetic samples are generated from training data only.

---

## Setup

```bash
git clone https://github.com/<your-username>/Predicting-Hospital-Readmissions.git
cd Predicting-Hospital-Readmissions
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Data

**The dataset is not in this repository and must not be.** MIMIC-IV is credentialed data under a
PhysioNet Data Use Agreement; redistributing any part of it, including a derived extract, violates
that agreement.

Get credentialed at [physionet.org/content/mimiciv](https://physionet.org/content/mimiciv/),
download the `hosp` module, and place these five here. Either `.csv` or `.csv.gz` works.

```
data/raw/hosp/
├── admissions.csv
├── patients.csv
├── diagnoses_icd.csv
├── d_icd_diagnoses.csv
└── transfers.csv
```

`data/` is gitignored.

### Run

```
notebooks/
├── 02_preprocessing_and_features.ipynb   run first: cohort, labels, severity scoring
├── 01_eda.ipynb                          demographics, care units, seasonality, correlations
└── 03_modeling.ipynb                     three classifiers, evaluation, saved artifact
```

Notebook 02 runs first: it writes `data/processed/kidney_cohort.parquet`, which the other two read.
Notebook 03 writes `models/readmission_rf.joblib`, which the dashboard reads.

---

## Dashboard

```bash
streamlit run app/streamlit_app.py
```

Runs entirely off local files. No database required.

**Patient lookup:** age, gender, visit count, diagnosis, severity, 30-day readmission probability,
a Yes/No call, and full admission history. Patients with `hospital_expire_flag = 1` short-circuit
the prediction, since someone who did not survive an admission cannot be readmitted.

**Cohort overview:** distribution of predicted risk across the cohort, plus breakdowns of flagged
patients by age band and severity.

The decision threshold is a slider rather than a hardcoded value, because trading recall against
false alarms is the actual decision a discharge team faces.

Lab trend charts (creatinine, hemoglobin, potassium, sodium, pH, WBC) appear only if
`data/raw/hosp/labevents.csv` is present. It is a large optional download; everything else works
without it.

---

## Disclaimer

Coursework. Not validated for clinical use and not a decision support tool. The model reflects the
biases of the data it was trained on, and a false negative here is a patient sent home without the
follow-up they needed.

## Licence

MIT (code only). MIMIC-IV is governed separately by the PhysioNet Data Use Agreement.