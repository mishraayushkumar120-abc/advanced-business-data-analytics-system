from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import tensorflow as tf
import numpy as np

app = FastAPI(title="Federated Fraud Detection API", version="1.0.0")
model = tf.keras.models.load_model("/content/drive/MyDrive/federated_fraud_model.h5")

class Transaction(BaseModel):
    features: List[float]

class BatchRequest(BaseModel):
    transactions: List[List[float]]

@app.get("/")
def root():
    return {"name": "Federated Fraud Detection API", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/model/info")
def model_info():
    return {"model_type": "Neural Network", "f1": 0.8350, "precision": 0.7963, "recall": 0.8776}

@app.post("/predict")
def predict(t: Transaction):
    if len(t.features) != 30:
        raise HTTPException(400, "Expected 30 features")
    x = np.array(t.features, dtype="float32").reshape(1, -1)
    p = float(model.predict(x, verbose=0)[0][0])
    return {"fraud_probability": round(p, 4), "prediction": "FRAUD" if p > 0.5 else "NORMAL"}

@app.post("/predict/batch")
def predict_batch(b: BatchRequest):
    x = np.array(b.transactions, dtype="float32")
    if x.shape[1] != 30:
        raise HTTPException(400, "Each row needs 30 features")
    ps = model.predict(x, verbose=0).flatten()
    results = [{"fraud_probability": round(float(p), 4), "prediction": "FRAUD" if p > 0.5 else "NORMAL"} for p in ps]
    return {"total": len(results), "fraud_count": sum(1 for r in results if r["prediction"] == "FRAUD"), "results": results}
