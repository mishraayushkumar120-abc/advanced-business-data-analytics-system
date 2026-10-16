# Project Report: Advanced Business Data Analytics System

## Overview
Privacy-preserving fraud detection using Federated Learning across 3 banks.

## Results
| Model | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| Bank A | 0.9067 | 0.9444 | 0.8718 |
| Bank B | 0.8710 | 0.9643 | 0.7941 |
| Bank C | 0.7692 | 0.7407 | 0.8000 |
| Federated | 0.8350 | 0.7963 | 0.8776 |

## Business Impact
- Money saved: $52,768
- Frauds caught: 431/492
- Recall: 87.76%

## Tech Stack
Python, TensorFlow, Scikit-learn, FastAPI, Streamlit, Matplotlib

## Federated Learning Process
1. Server creates global model
2. Sends to 3 banks
3. Each bank trains locally
4. Only weights sent back (no data)
5. FedAvg aggregation
6. Repeat 14 rounds

## Privacy Guarantee
Raw data never leaves banks. GDPR/PSD2 compliant.

## References
1. McMahan et al. (2017). FedAvg. AISTATS.
2. ULB ML Group. Credit Card Fraud Dataset.
3. J.P. Morgan & NVIDIA (2026). Federated Learning.
