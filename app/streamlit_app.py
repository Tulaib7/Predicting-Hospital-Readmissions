"""Hospital Readmission Prediction dashboard.
 
Reads the kidney cohort produced by notebooks/02_preprocessing_and_features.ipynb and the model
saved by notebooks/03_modeling.ipynb. No database required.
 
    streamlit run app/streamlit_app.py
 
Prerequisites:
    data/processed/kidney_cohort.parquet   (notebook 02)
    models/readmission_rf.joblib           (notebook 03)
 
Lab trends are shown only if data/raw/hosp/labevents.csv is present. That file is a large optional
download; everything else in the dashboard works without it.
"""
 
from __future__ import annotations
 
from pathlib import Path
 
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
 
ROOT = Path(__file__).resolve().parents[1]
COHORT_PATH = ROOT / "data" / "processed" / "kidney_cohort.parquet"
MODEL_PATH = ROOT / "models" / "readmission_rf.joblib"
LABS_PATH = ROOT / "data" / "raw" / "hosp" / "labevents.csv"
 
# itemid -> (label, unit, reference low, reference high)
LAB_ITEMS = {
    50912: ("Creatinine", "mg/dL", 0.5, 1.2),
    51222: ("Hemoglobin", "g/dL", 12.0, 17.5),
    50971: ("Potassium", "mEq/L", 3.5, 5.0),
    50983: ("Sodium", "mEq/L", 135.0, 145.0),
    50820: ("pH", "units", 7.35, 7.45),
    51301: ("WBC", "K/uL", 4.0, 11.0),
}
 
AGE_BINS = [18, 34, 49, 64, 79, 120]
AGE_LABELS = ["Youth (19-34)", "Adults (35-49)", "Mature Adults (50-64)",
              "Seniors (65-79)", "Elders (80+)"]
SEVERITY_LABELS = {1: "1 (Low)", 2: "2 (Mild)", 3: "3 (Moderate)",
                   4: "4 (Severe)", 5: "5 (End Stage)"}
 
st.set_page_config(page_title="Hospital Readmission Prediction", layout="wide")
 
 
# ----------------------------------------------------------------------------- loading
@st.cache_resource
def load_model() -> dict:
    if not MODEL_PATH.exists():
        st.error("models/readmission_rf.joblib not found. Run notebooks/03_modeling.ipynb first.")
        st.stop()
    return joblib.load(MODEL_PATH)
 
 
@st.cache_data
def load_cohort() -> pd.DataFrame:
    if not COHORT_PATH.exists():
        st.error(
            "data/processed/kidney_cohort.parquet not found. "
            "Run notebooks/02_preprocessing_and_features.ipynb first."
        )
        st.stop()
    df = pd.read_parquet(COHORT_PATH)
    df["admittime"] = pd.to_datetime(df["admittime"])
    return df.sort_values(["subject_id", "admittime"])
 
 
@st.cache_data(show_spinner="Reading labevents (this file is large)...")
def load_labs(subject_id: int) -> pd.DataFrame:
    """Stream labevents in chunks and keep only this patient's rows of interest."""
    if not LABS_PATH.exists():
        return pd.DataFrame()
 
    keep = []
    reader = pd.read_csv(
        LABS_PATH,
        usecols=["subject_id", "itemid", "charttime", "valuenum"],
        parse_dates=["charttime"],
        chunksize=1_000_000,
    )
    for chunk in reader:
        hit = chunk[
            (chunk["subject_id"] == subject_id)
            & (chunk["itemid"].isin(LAB_ITEMS))
            & (chunk["valuenum"].notna())
        ]
        if not hit.empty:
            keep.append(hit)
 
    if not keep:
        return pd.DataFrame()
    return pd.concat(keep).sort_values("charttime")
 
 
# ----------------------------------------------------------------------------- scoring
def score(cohort: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Attach a readmission probability to every row of the cohort."""
    model, features = artifact["model"], artifact["features"]
    out = cohort.copy()
    out["probability"] = model.predict_proba(out[features])[:, 1]
    return out
 
 
def lab_chart(labs: pd.DataFrame, itemid: int) -> go.Figure:
    label, unit, low, high = LAB_ITEMS[itemid]
    series = labs[labs["itemid"] == itemid]
 
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=series["charttime"], y=series["valuenum"],
                   mode="lines+markers", name=label)
    )
    for bound in (low, high):
        fig.add_hline(y=bound, line_dash="dash", line_color="green", opacity=0.6)
 
    fig.update_layout(title=f"{label} ({unit})", height=250,
                      margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    return fig
 
 
# ----------------------------------------------------------------------------- app
artifact = load_model()
cohort = load_cohort()
 
st.title("Hospital Readmission Prediction")
st.caption(
    "30-day readmission risk for patients with kidney-related diagnoses (MIMIC-IV). "
    "Coursework project. Not a clinical decision support tool."
)
 
with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Decision threshold", 0.05, 0.95, float(artifact["threshold"]), 0.05,
        help="A patient is flagged when their predicted probability meets this cutoff. "
             "Lower catches more true readmissions at the cost of more false alarms.",
    )
    st.divider()
    st.caption("Model configuration used at training time:")
    st.write({
        "leaky `readmit` feature": artifact["include_readmit_feature"],
        "patient-grouped split": artifact["group_split_by_patient"],
    })
    if not artifact["group_split_by_patient"] or artifact["include_readmit_feature"]:
        st.warning("This model was trained with the settings that inflate performance. See LIMITATIONS.md.")
 
scored = score(cohort, artifact)
 
tab_patient, tab_cohort = st.tabs(["Patient lookup", "Cohort overview"])
 
# ----------------------------------------------------------------------------- patient tab
with tab_patient:
    ids = sorted(scored["subject_id"].unique())
    default = ids[0]
 
    col_a, col_b = st.columns([2, 1])
    subject_id = col_a.selectbox(
        "Patient ID", ids, index=ids.index(default),
        help=f"{len(ids):,} patients in the kidney cohort.",
    )
    col_b.write("")
    col_b.write("")
    generate = col_b.button("Generate", type="primary", use_container_width=True)
 
    if generate:
        history = scored[scored["subject_id"] == subject_id].sort_values(
            "admittime", ascending=False
        )
        latest = history.iloc[0]
 
        # A patient who died in hospital cannot be readmitted.
        if latest["hospital_expire_flag"] == 1:
            st.info(
                "This patient did not survive their most recent admission. "
                "No readmission prediction is generated."
            )
            st.stop()
 
        probability = float(latest["probability"])
        flagged = probability >= threshold
 
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Patient Age", int(latest["anchor_age"]))
        c2.metric("Gender", latest["gender"])
        c3.metric("Number of Visits", int(history["hadm_id"].nunique()))
        c4.metric("Readmission Probability", f"{probability * 100:.2f}%")
        c5.metric("Readmitted within 30 days", "Yes" if flagged else "No")
 
        st.markdown(f"**Primary diagnosis:** {latest['long_title']}")
        st.markdown(
            f"**Severity:** {SEVERITY_LABELS.get(int(latest['severity']), latest['severity'])} "
            f"&nbsp;&nbsp; **Length of stay:** {latest['los_hours']:.1f} hours"
        )
        st.progress(min(probability, 1.0))
        st.caption(f"Flagged when probability >= {threshold:.2f}.")
 
        st.subheader("Admission history")
        st.dataframe(
            history[[
                "hadm_id", "admittime", "los_hours", "icd_code", "long_title",
                "severity", "readmission_under_30_days", "probability",
            ]].rename(columns={"readmission_under_30_days": "readmitted_30d"}),
            use_container_width=True, hide_index=True,
        )
 
        st.subheader("Lab trends")
        if not LABS_PATH.exists():
            st.info(
                "Lab trends require data/raw/hosp/labevents.csv, which is not present. "
                "Everything else on this page works without it."
            )
        else:
            labs = load_labs(int(subject_id))
            if labs.empty:
                st.info("No lab results recorded for this patient.")
            else:
                items = [i for i in LAB_ITEMS if (labs["itemid"] == i).any()]
                for start in range(0, len(items), 3):
                    for col, itemid in zip(st.columns(3), items[start:start + 3]):
                        col.plotly_chart(lab_chart(labs, itemid), use_container_width=True)
 
# ----------------------------------------------------------------------------- cohort tab
with tab_cohort:
    flagged_mask = scored["probability"] >= threshold
 
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", f"{scored['subject_id'].nunique():,}")
    c2.metric("Admissions", f"{scored['hadm_id'].nunique():,}")
    c3.metric("Actual 30-day rate", f"{scored['readmission_under_30_days'].mean():.1%}")
    c4.metric("Flagged at threshold", f"{flagged_mask.mean():.1%}")
 
    st.subheader("Predicted risk distribution")
    fig = px.histogram(scored, x="probability", nbins=50,
                       labels={"probability": "Predicted readmission probability"})
    fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"threshold {threshold:.2f}")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
 
    left, right = st.columns(2)
 
    view = scored[flagged_mask].copy()
    view["age_group"] = pd.cut(view["anchor_age"], bins=AGE_BINS, labels=AGE_LABELS)
 
    age_pct = view["age_group"].value_counts().sort_index() / len(scored) * 100
    fig_age = px.bar(
        x=age_pct.index.astype(str), y=age_pct.values,
        labels={"x": "Age Group", "y": "% of cohort"},
        title="Age distribution of flagged patients",
    )
    fig_age.update_traces(marker_color="#6b5bd6", texttemplate="%{y:.1f}%", textposition="outside")
    fig_age.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
    left.plotly_chart(fig_age, use_container_width=True)
 
    sev_pct = view["severity"].value_counts().sort_index() / len(scored) * 100
    fig_sev = px.bar(
        x=[SEVERITY_LABELS.get(int(s), s) for s in sev_pct.index], y=sev_pct.values,
        labels={"x": "Severity", "y": "% of cohort"},
        title="Severity distribution of flagged patients",
    )
    fig_sev.update_traces(marker_color="#6b5bd6", texttemplate="%{y:.1f}%", textposition="outside")
    fig_sev.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
    right.plotly_chart(fig_sev, use_container_width=True)
 
    st.caption(
        "Severe cases appear under-represented among flagged patients. Part of this is a competing "
        "risk: the sickest patients are more likely to die in hospital, and a patient who does not "
        "survive cannot be readmitted. Read this descriptively, not causally."
    )