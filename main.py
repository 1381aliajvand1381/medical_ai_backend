import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from scipy.sparse import hstack
from gensim.utils import simple_preprocess

app = FastAPI()

# ۱. لود کردن تمام کامپوننت‌های ذخیره شده
try:
    model = joblib.load("final_model.pkl")
    vectorizer_tfidf = joblib.load("tfidf_vectorizer.pkl")
    w2v_model = joblib.load("word2vec_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    drug_df = pd.read_csv("drugbank.csv")
    qa_df = pd.read_csv("medquad_clean_qa.csv")
    print("✅ All assets loaded successfully!")
except Exception as e:
    print(f"❌ Error loading assets: {e}")

# ۲. توابع کمکی (باید دقیقاً مشابه کد آموزش باشند)
def get_w2v_vector(text, model, size=100):
    words = simple_preprocess(text)
    vectors = [model.wv[word] for word in words if word in model.wv]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(size)

def clean_text(text):
    # پاکسازی اولیه مشابه کد آموزش
    text = str(text).lower().replace('_', ' ').replace('|', ' ')
    return ' '.join(text.split())

class SymptomInput(BaseModel):
    symptoms: str

@app.post("/analyze")
async def analyze(input_data: SymptomInput):
    try:
        # ۳. پیش‌پردازش متن ورودی
        cleaned_text = clean_text(input_data.symptoms)
        
        # ۴. استخراج ویژگی TF-IDF
        tfidf_vec = vectorizer_tfidf.transform([cleaned_text])
        
        # ۵. استخراج ویژگی Word2Vec
        w2v_vec = np.array([get_w2v_vector(cleaned_text, w2v_model)])
        
        # ۶. ترکیب ویژگی‌ها (بسیار مهم: مدل روی این ساختار آموزش دیده)
        combined_vec = hstack([tfidf_vec, w2v_vec])
        
        # ۷. پیش‌بینی احتمالات
        probabilities = model.predict_proba(combined_vec)[0]
        top_indices = np.argsort(probabilities)[::-1]
        
        first_prob = probabilities[top_indices[0]]
        second_prob = probabilities[top_indices[1]]
        
        disease = label_encoder.inverse_transform([top_indices[0]])[0]
        
        # شرط تایید سخت‌گیرانه (می‌توانید برای تست اول 0.70 را کمتر کنید)
        is_confirmed = (first_prob >= 0.70) and (first_prob - second_prob >= 0.20)
        
        # ۸. جستجوی اطلاعات تکمیلی
        drug_info = ""
        if is_confirmed:
            matching = drug_df[drug_df['Indication'].str.contains(disease, case=False, na=False)]
            drug_info = " / ".join(matching['Name'].tolist()[:5])

        qa_search = qa_df[qa_df['question'].str.contains(disease, case=False, na=False)]['answer'].tolist()
        qa_info = qa_search[0] if qa_search else ""

        return {
            "disease": disease if is_confirmed else "نامشخص",
            "is_confirmed": bool(is_confirmed),
            "drug_info": drug_info,
            "qa_info": qa_info,
            "confidence": float(first_prob),
            "raw_disease": disease # اضافه شد برای دیباگ سمت اپلیکیشن
        }
    except Exception as e:
        print(f"Prediction Error: {e}")
        return {"is_confirmed": False, "disease": "error", "message": str(e)}

@app.get("/")
async def health():
    return {"status": "online"}