# Federated Learning for Cross-Bank Fraud Detection

Privacy-preserving ML system for credit card fraud detection across 3 banks using Federated Learning (FedAvg). Raw data never shared.

## Results

| Model | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| Bank A | 0.9067 | 0.9444 | 0.8718 |
| Bank B | 0.8710 | 0.9643 | 0.7941 |
| Bank C | 0.7692 | 0.7407 | 0.8000 |
| Federated | 0.8350 | 0.7963 | 0.8776 |

Key Achievement: Bank C improved from 0.77 to 0.83 F1 without data sharing.

## Business Impact

- Money saved: $52,768
- Frauds caught: 431/492
- Recall: 87.76%

## Tech Stack

Python, TensorFlow, Scikit-learn, XGBoost, FastAPI, Streamlit, Matplotlib

## Structure

- api/ - FastAPI endpoints
- dashboard/ - Streamlit dashboard
- docs/ - Report and diagrams
- models/ - Trained model and samples

## How to Run

Install: pip install -r requirements.txt

API: uvicorn api.app:app --host 0.0.0.0 --port 8000

Dashboard: streamlit run dashboard/app.py

## API Endpoints

- GET / - API info
- GET /health - Health check
- GET /model/info - Model metrics
- POST /predict - Single prediction
- POST /predict/batch - Batch prediction

## Dataset

ULB Credit Card Fraud Detection (Kaggle)
- 284,807 transactions
- 492 fraud cases (0.172%)
- 30 features

## Federated Learning Process

1. Server creates global model
2. Sends to 3 banks
3. Each bank trains locally
4. Weights sent back (no data)
5. FedAvg aggregation
6. Repeat 14 rounds

## Privacy

Raw data never leaves banks. Only model weights shared. GDPR/PSD2 compliant.
