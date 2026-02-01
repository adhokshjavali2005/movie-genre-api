from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import re

# Load saved model and vectorizer
tfidf = joblib.load("tfidf_vectorizer.pkl")
svm_model = joblib.load("svm_genre_model.pkl")

app = FastAPI(title="Movie Genre Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class MovieInput(BaseModel):
    plot: str

def clean_text(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.post("/predict")
def predict_genre(data: MovieInput):
    cleaned_plot = clean_text(data.plot)
    vector = tfidf.transform([cleaned_plot])
    prediction = svm_model.predict(vector)[0]
    return {"predicted_genre": prediction}


