import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
engine = create_engine('postgresql://jacob:test123@localhost:5432/healthcare_db')
df = pd.read_sql('SELECT * FROM diabetes_hospital_data', engine)

# Engineer treatment_success if not present
median_stay = df['time_in_hospital'].median()
df['treatment_success'] = np.where(
    (df['discharge_disposition_id'] == 1) & (df['time_in_hospital'] < median_stay), 1, 0
)

st.set_page_config(layout="wide")
st.title("Patient Outcomes Dashboard")

# Distributions in two columns
col1, col2 = st.columns(2)
with col1:
    st.subheader("Readmission Risk Distribution")
    st.bar_chart(df['readmission_risk'].value_counts())
with col2:
    st.subheader("Treatment Success Distribution")
    st.bar_chart(df['treatment_success'].value_counts())

st.markdown("---")
col3, col4 = st.columns(2)
with col3:
    st.subheader("Feature Importances: Readmission Risk")
    # Paste or load your feature importances here; example below
    importances_risk = {
        "number_inpatient": 12.4,
        "height": 0.11,
        "metformin_No": 0.08,
        "number_emergency": 0.06,
        "pioglitazone_No": 0.05,
        "glimepiride_No": 0.05,
        "A1Cresult_>8": 0.05,
        "diabetesMed_Yes": 0.05,
        "pioglitazone_Steady": 0.05,
        "metformin_Steady": 0.05
    }
    plt.figure(figsize=(5,2.5))
    plt.barh(list(importances_risk.keys()), list(importances_risk.values()))
    plt.xlabel("Abs. Coefficient")
    st.pyplot(plt.gcf())
    plt.clf()

with col4:
    st.subheader("Feature Importances: Treatment Success")
    importances_success = {
        "discharge_disposition_id": 24.4,
        "time_in_hospital": 8.8,
        "height": 0.32,
        "max_glu_serum_Norm": 0.30,
        "medical_specialty_ObstetricsandGynecology": 0.24,
        "weight_[75-100)": 0.20,
        "medical_specialty_Cardiology": 0.18,
        "medical_specialty_Surgery-Cardiovascular/Thoracic": 0.17,
        "medical_specialty_PhysicalMedicineandRehabilitation": 0.16,
        "race_Asian": 0.16,
    }
    plt.figure(figsize=(5,2.5))
    plt.barh(list(importances_success.keys()), list(importances_success.values()))
    plt.xlabel("Abs. Coefficient")
    st.pyplot(plt.gcf())
    plt.clf()

st.markdown("---")

# Specialty drilldown, two columns again if you want
specialty = st.selectbox('Select Medical Specialty', df['medical_specialty'].unique())
filtered = df[df['medical_specialty'] == specialty]
st.subheader(f"Outcomes for {specialty}")
st.dataframe(filtered[['readmission_risk', 'treatment_success']].describe())

st.markdown("---")
st.subheader("Anomaly Alerts")
anomaly_specialties = df.groupby('medical_specialty')['readmission_risk'].mean().sort_values(ascending=False).head(5)
st.dataframe(anomaly_specialties)

