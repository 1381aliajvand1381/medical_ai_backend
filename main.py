import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import os

app = FastAPI()

# لود کردن فایل‌ها با هندل کردن خطا
try:
    model = joblib.load("final_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    drug_df = pd.read_csv("drugbank.csv")
    qa_df = pd.read_csv("medquad_clean_qa.csv")
except Exception as e:
    print(f"Error loading assets: {e}")

class SymptomInput(BaseModel):
    symptoms: str

@app.post("/analyze")
async def analyze(input_data: SymptomInput):
    try:
        text_vector = vectorizer.transform([input_data.symptoms])
        probabilities = model.predict_proba(text_vector)[0]
        
        top_indices = np.argsort(probabilities)[::-1]
        first_prob = probabilities[top_indices[0]]
        second_prob = probabilities[top_indices[1]]
        
        disease = label_encoder.inverse_transform([top_indices[0]])[0]
        
        # شرط تایید: احتمال بالای 70% و اختلاف 20% با نفر دوم
        is_confirmed = (first_prob >= 0.70) and (first_prob - second_prob >= 0.20)
        
        drug_info = ""
        if is_confirmed:
            # ستون‌های فایل شما: Name و Indication
            matching = drug_df[drug_df['Indication'].str.contains(disease, case=False, na=False)]
            drug_info = " / ".join(matching['Name'].tolist()[:5])

        qa_search = qa_df[qa_df['question'].str.contains(disease, case=False, na=False)]['answer'].tolist()
        qa_info = qa_search[0] if qa_search else ""

        return {
            "disease": disease if is_confirmed else "نامشخص",
            "is_confirmed": bool(is_confirmed),
            "drug_info": drug_info,
            "qa_info": qa_info,
            "confidence": float(first_prob)
        }
    except:
        return {"is_confirmed": False, "disease": "error"}

@app.get("/")
async def health():
    return {"status": "online"}