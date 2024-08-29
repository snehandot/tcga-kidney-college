import streamlit as st
import pandas as pd
import torch
import pickle

model_path = 'model_weights.pth'  # Update with your model path

model = torch.load(model_path)
model.eval()

feature_names = [
    'Diagnosis Age', 'Neoplasm Disease Stage American Joint Committee on Cancer Code',
    'American Joint Committee on Cancer Publication Version Type', 'Aneuploidy Score',
    'Buffa Hypoxia Score', 'Last Communication Contact from Initial Pathologic Diagnosis Date',
    'Birth from Initial Pathologic Diagnosis Date', 'Disease Free (Months)',
    'Disease Free Status', 'Months of disease-specific survival', 'Disease-specific Survival status',
    'Ethnicity Category', 'Form completion date', 'Fraction Genome Altered',
    'Genetic Ancestry Label', 'Neoplasm Histologic Grade', 'Neoadjuvant Therapy Type Administered Prior To Resection Text',
    'International Classification of Diseases for Oncology, Third Edition ICD-O-3 Histology Code',
    'In PanCan Pathway Analysis', 'MSI MANTIS Score', 'MSIsensor Score', 'Mutation Count',
    'New Neoplasm Event Post Initial Therapy Indicator', 'Overall Survival (Months)',
    'American Joint Committee on Cancer Metastasis Stage Code', 'Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code',
    'American Joint Committee on Cancer Tumor Stage Code', 'Person Neoplasm Cancer Status',
    'Progress Free Survival (Months)', 'Progression Free Status',
    'Primary Lymph Node Presentation Assessment', 'Prior Diagnosis', 'Race Category',
    'Radiation Therapy', 'Ragnum Hypoxia Score', 'Sex', 'Tissue Prospective Collection Indicator',
    'Tissue Retrospective Collection Indicator', 'Tissue Source Site', 'Tissue Source Site Code',
    'TMB (nonsynonymous)', 'Winter Hypoxia Score'
]

st.title("Survival Prediction")

inputs = []
for feature in feature_names:
    value = st.number_input(feature, value=0.0, format="%.6f")
    inputs.append(value)

input_df = pd.DataFrame([inputs], columns=feature_names)

input_tensor = torch.tensor(input_df.values, dtype=torch.float32)

with torch.no_grad():
    prediction = model(input_tensor).item()

st.write(f"Predicted Survival Probability: {prediction:.4f}")
