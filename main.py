import os
import re
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gensim.models import Word2Vec
from scipy.sparse import hstack
from fastapi.middleware.cors import CORSMiddleware

# پیدا کردن مسیر پوشه‌ای که این فایل در آن قرار دارد
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    return os.path.join(BASE_DIR, filename)

app = FastAPI(title="Medical AI Backend")

# حل مشکل دسترسی از مرورگرها یا فلاتر (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- بارگذاری مدل‌ها و فایل‌ها ---
try:
    print("⏳ در حال بارگذاری مدل‌ها...")
    rf_model = joblib.load(get_path('final_model.pkl'))
    tfidf = joblib.load(get_path('tfidf_vectorizer.pkl'))
    le = joblib.load(get_path('label_encoder.pkl'))
    # لود کردن مدل Word2Vec (اگر با joblib ذخیره کردید)
    w2v_model = joblib.load(get_path('word2vec_model.pkl')) 
    
    print("⏳ در حال بارگذاری دیتابیس‌ها...")
    df_drugs = pd.read_csv(get_path('drugbank.csv'))
    df_qa = pd.read_csv(get_path('medquad_clean_qa.csv'))
    
    print("✅ تمام فایل‌ها با موفقیت لود شدند.")
except Exception as e:
    print(f"❌ خطا در بارگذاری فایل‌ها: {e}")

# --- توابع پیش‌پردازش (مطابق فایل Jupyter شما) ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return ' '.join(text.split())

def get_w2v_vector(text, model):
    words = text.split()
    # دسترسی به بردارها بر اساس ساختار مدل شما
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# --- مدل داده ورودی ---
class ChatRequest(BaseModel):
    text: str

# --- نقطه دسترسی اصلی (Endpoint) ---
@app.post("/analyze")
async def analyze(request: ChatRequest):
    try:
        # ۱. تمیزکاری ورودی
        user_input = clean_text(request.text)
        
        # ۲. تبدیل به بردار (Vectorization)
        tfidf_vec = tfidf.transform([user_input])
        w2v_vec = np.array([get_w2v_vector(user_input, w2v_model)])
        combined_vec = hstack([tfidf_vec, w2v_vec])
        
        # ۳. پیش‌بینی بیماری و درصد اطمینان
        probs = rf_model.predict_proba(combined_vec)[0]
        top_idx = np.argsort(probs)[-1]
        disease = le.inverse_transform([top_idx])[0]
        confidence = float(probs[top_idx])
        
        # ۴. پیدا کردن دارو (DrugBank)
        # جستجوی نام بیماری در ستون Indication
        drugs = df_drugs[df_drugs['Indication'].str.contains(disease, case=False, na=False)]['Name'].head(3).tolist()
        
        # ۵. پیدا کردن اطلاعات علمی (MedQuad)
        qa_match = df_qa[df_qa['question'].str.contains(disease, case=False, na=False)].head(1)
        medical_info = qa_match['answer'].values[0] if not qa_match.empty else "توضیحات بیشتری یافت نشد."

        # ۶. بازگشت نتایج نهایی
        return {
            "status": "success",
            "disease": disease,
            "confidence": f"{confidence*100:.1f}%",
            "suggested_drugs": drugs,
            "medical_info": medical_info
        }

    except Exception as e:
        print(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Server is running! API is ready at /analyze"}

if __name__ == "__main__":
    import uvicorn
    # اجرا روی پورت ۸۰۰۰
    uvicorn.run(app, host="0.0.0.0", port=8000)