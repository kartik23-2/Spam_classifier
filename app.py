from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load model + vectorizer + threshold
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
threshold = pickle.load(open("threshold.pkl", "rb"))

app = FastAPI(title="Spam & Profanity Detection API")

# Request body model
class Comment(BaseModel):
    text: str

@app.post("/predict")
def predict(comment: Comment):
    vectorized = tfidf.transform([comment.text])
    prob = model.predict_proba(vectorized)[:, 1]
    prediction = int(prob >= threshold)

    return {
        "comment": comment.text,
        "prediction": prediction,
        "label": "SPAM / PROFANE" if prediction == 1 else "CLEAN COMMENT",
        "probability": float(prob)
    }
