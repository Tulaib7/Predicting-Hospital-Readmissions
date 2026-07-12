# Limitations

Known problems with this analysis, written down rather than buried. The first two are reproduced
faithfully from the original project and each has a toggle in `notebooks/03_modeling.ipynb`, so
both configurations can be run and compared.

Cohort figures referenced below: 261,347 diagnosis rows across 111,891 stays and 47,919 patients.
Positive rate 25.8%.

## 1. `readmit` leaks the target

`readmit` counts how many separate admissions a patient carried a given ICD code on. A patient with
`readmit = 4` was, by definition, admitted four times. The 30-day label asks whether the patient
came back. These are not independent, and a model given `readmit` is partly reading the answer off
the feature rather than predicting it.

The correlation heatmap in notebook 01 shows it directly: `readmit` correlates with the target far
more strongly than any clinical variable does.

**Toggle:** `INCLUDE_READMIT_FEATURE = False` in notebook 03.

## 2. The same patient appears in train and test

The modelling table has one row per (admission x diagnosis code). The cohort averages 2.3 diagnosis
rows per stay and 2.3 stays per patient, so a single patient contributes several near-identical
rows sharing the same demographics and history features. A random `train_test_split` scatters those
rows across both sides, letting the model memorise a patient during training and be rewarded for
recognising them at test time.

Note that this does *not* meaningfully inflate the positive rate: the rebuilt cohort sits at 25.8%
at row level and 20.2% at admission level, so the duplication is roughly balanced across classes.
The damage is patient overlap, not label distortion.

**Toggle:** `GROUP_SPLIT_BY_PATIENT = True` in notebook 03 splits on `subject_id` so no patient
crosses the boundary. The cleaner fix, not implemented here in order to stay faithful to the
original, is to collapse the table to one row per admission and aggregate diagnoses into counts or
a max-severity column.

## 3. What the first two cost

Random Forest, before and after both corrections:

| | leaky + random split | corrected |
|---|---|---|
| ROC AUC | 0.777 | 0.739 |
| PR AUC | 0.521 | 0.435 |
| Precision | 0.416 | 0.376 |

0.04 ROC AUC and 0.09 PR AUC of the original performance was not real. The model did not collapse,
which means there is genuine signal in age, length of stay, prior admissions and severity. It is
simply weaker than the first run suggested. ROC AUC 0.739 is in line with the published MIMIC
readmission literature.

## 4. Accuracy is a misleading metric here

At a 25.8% positive rate, a model that never predicts readmission scores 74% accuracy while being
useless. The original report led with accuracy (52% / 70% / 72%) and picked XGBoost on that basis,
despite its recall for the positive class being 0.36. Notebook 03 reports precision, recall, ROC
AUC and PR AUC instead, with PR AUC as the primary comparison against a 0.258 baseline.

## 5. The flag rate is not operationally usable

At the inherited 0.30 threshold, the corrected Random Forest flags 45.6% of the cohort and XGBoost
flags 70.5%. Handing a discharge team a list containing half the ward guarantees the list gets
ignored. The threshold was inherited from the original project, not chosen from the data. In a real
deployment it would be set from the relative cost of a missed readmission versus an unnecessary
follow-up call. The Streamlit dashboard exposes it as a slider for exactly this reason.

## 6. Severe patients appear to readmit *less*

The original read this as evidence of better care for severe cases. A competing explanation is a
competing risk: the sickest patients are disproportionately the ones who die in hospital or move to
hospice, and a patient who does not survive cannot be readmitted. Death is being counted as a good
outcome. `hospital_expire_flag` is in the feature set so the model sees part of this, but any
causal reading of the severity gradient needs to model readmission and death jointly.

## 7. The severity score is a keyword lookup

It reads ICD long titles, not clinical notes or labs. It also matches greedily from the top, which
produces at least one known quirk: "Chronic kidney disease, unspecified" hits `unspecified`
(severity 2) before it can reach `chronic kidney disease` (severity 3). It adds signal, but it is
not a validated severity index. GFR-based CKD staging from the lab tables would be a real
improvement.

## 8. No temporal validation

Splits are random, not chronological. A model deployed in a hospital predicts the future from the
past, so a time-based split is the honest evaluation.

## 9. No fairness analysis

Readmission models are known to encode disparities in access to care. `insurance` and
`admission_location` are proxies for socioeconomic status and the model uses both. Performance was
never measured across demographic subgroups.