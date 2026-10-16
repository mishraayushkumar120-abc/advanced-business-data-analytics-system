
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Fraud Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #f7f8fa; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { color: #1a1f36 !important; font-weight: 600; font-size: 28px; letter-spacing: -0.5px; margin-bottom: 0; }
    h2 { color: #1a1f36 !important; font-weight: 600; font-size: 20px; margin-top: 1.5rem; }
    h3 { color: #1a1f36 !important; font-weight: 500; font-size: 16px; }
    .subtitle { color: #697386; font-size: 14px; margin-top: 4px; margin-bottom: 24px; }
    .card { background: #ffffff; border: 1px solid #e3e8ee; border-radius: 8px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
    .metric-card { background: #ffffff; border: 1px solid #e3e8ee; border-radius: 8px; padding: 20px; height: 100%; }
    .metric-label { color: #697386; font-size: 12px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
    .metric-value { color: #1a1f36; font-size: 28px; font-weight: 600; line-height: 1; }
    .stNumberInput input, .stTextInput input { background-color: #ffffff !important; color: #1a1f36 !important; border: 1px solid #d5dbe1 !important; border-radius: 6px !important; padding: 8px 12px !important; }
    .stNumberInput label, .stTextInput label { color: #1a1f36 !important; font-size: 13px !important; font-weight: 500 !important; }
    .stButton > button { background: #1a1f36; color: #ffffff !important; border: none; border-radius: 6px; padding: 10px 20px; font-size: 14px; font-weight: 500; width: 100%; }
    .stButton > button:hover { background: #2d3548; }
    section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e3e8ee; }
    section[data-testid="stSidebar"] * { color: #1a1f36 !important; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    hr { border: none; border-top: 1px solid #e3e8ee; margin: 24px 0; }
    .badge-success { display: inline-block; background: #ecfdf3; color: #067647; font-size: 12px; font-weight: 500; padding: 4px 10px; border-radius: 12px; border: 1px solid #abefc6; }
    .badge-danger { display: inline-block; background: #fef3f2; color: #b42318; font-size: 12px; font-weight: 500; padding: 4px 10px; border-radius: 12px; border: 1px solid #fecdca; }
    .info-table { width: 100%; border-collapse: collapse; }
    .info-table td { padding: 12px 0; border-bottom: 1px solid #e3e8ee; font-size: 14px; color: #1a1f36; }
    .info-table td:first-child { color: #697386; width: 40%; }
    .info-table tr:last-child td { border-bottom: none; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("/content/drive/MyDrive/federated_fraud_model.h5")

@st.cache_data
def load_samples():
    fraud = np.load('/content/drive/MyDrive/fraud_samples.npy')
    normal = np.load('/content/drive/MyDrive/normal_samples.npy')
    return fraud, normal

model = load_model()
fraud_samples, normal_samples = load_samples()

with st.sidebar:
    st.markdown("### Fraud Detection")
    st.markdown("<p style='color:#697386; font-size:13px; margin-top:-10px;'>Federated Learning System</p>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("", ["Prediction", "Batch Analysis", "Model Performance", "System Architecture"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<p style='color:#697386; font-size:12px;'>Version 1.0</p>", unsafe_allow_html=True)

st.markdown("# Fraud Detection System")
st.markdown("<p class='subtitle'>Privacy-preserving machine learning across multiple financial institutions</p>", unsafe_allow_html=True)

if page == "Prediction":
    st.markdown("## Transaction Analysis")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Select Sample Transaction")

        sample_type = st.selectbox(
            "Choose a sample:",
            ["Fraud Sample #1", "Fraud Sample #2", "Fraud Sample #3",
             "Normal Sample #1", "Normal Sample #2", "Normal Sample #3"]
        )

        st.markdown("</div>", unsafe_allow_html=True)
        predict_btn = st.button("Analyze Transaction")

    with col2:
        if predict_btn:
            if "Fraud" in sample_type:
                idx = int(sample_type.split("#")[1]) - 1
                features = fraud_samples[idx].reshape(1, -1)
                true_label = "FRAUD"
            else:
                idx = int(sample_type.split("#")[1]) - 1
                features = normal_samples[idx].reshape(1, -1)
                true_label = "NORMAL"

            proba = float(model.predict(features, verbose=0)[0][0])
            is_fraud = proba > 0.5

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Analysis Result")

            st.markdown(f"""
            <table class='info-table'>
                <tr>
                    <td>Selected Sample</td>
                    <td><strong>{sample_type}</strong></td>
                </tr>
                <tr>
                    <td>Actual Label</td>
                    <td><strong>{true_label}</strong></td>
                </tr>
                <tr>
                    <td>Fraud Probability</td>
                    <td><strong>{proba:.2%}</strong></td>
                </tr>
                <tr>
                    <td>Model Prediction</td>
                    <td>{'<span class="badge-danger">FRAUD</span>' if is_fraud else '<span class="badge-success">NORMAL</span>'}</td>
                </tr>
                <tr>
                    <td>Result</td>
                    <td>{'Correct prediction' if (is_fraud and true_label == 'FRAUD') or (not is_fraud and true_label == 'NORMAL') else 'Incorrect prediction'}</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            if is_fraud:
                st.error(f"Fraud detected with {proba:.1%} confidence")
            else:
                st.success(f"Transaction appears legitimate ({proba:.1%} fraud probability)")
        else:
            st.markdown("<div class='card' style='text-align:center; padding: 60px 20px;'>", unsafe_allow_html=True)
            st.markdown("<p style='color:#697386; font-size:14px;'>Select a sample and click Analyze</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

elif page == "Batch Analysis":
    st.markdown("## Batch Transaction Analysis")
    st.markdown("<p class='subtitle'>Upload a CSV file with 30 features to analyze multiple transactions</p>", unsafe_allow_html=True)

    file = st.file_uploader("", type="csv", label_visibility="collapsed")

    if file:
        df = pd.read_csv(file)
        st.markdown("### Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        if st.button("Run Batch Analysis"):
            preds = model.predict(df.values.astype("float32"), verbose=0).flatten()
            df["Fraud_Probability"] = preds
            df["Prediction"] = (preds > 0.5).astype(int)

            st.markdown("### Results Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Total Transactions</div><div class='metric-value'>{len(df):,}</div></div>", unsafe_allow_html=True)
            with col2:
                fraud_count = int(df["Prediction"].sum())
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Flagged as Fraud</div><div class='metric-value' style='color:#b42318;'>{fraud_count:,}</div></div>", unsafe_allow_html=True)
            with col3:
                normal_count = len(df) - fraud_count
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Normal</div><div class='metric-value' style='color:#067647;'>{normal_count:,}</div></div>", unsafe_allow_html=True)

            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False)
            st.download_button("Download Results", csv, "fraud_results.csv", "text/csv")

elif page == "Model Performance":
    st.markdown("## Model Performance Metrics")
    st.markdown("<p class='subtitle'>Evaluated on combined test data</p>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    for col, (label, value) in zip([col1, col2, col3, col4], [("F1 Score", "0.83"), ("Precision", "0.78"), ("Recall", "0.89"), ("AUC", "0.97")]):
        with col:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Training Progress")
    history = pd.DataFrame({"Round": list(range(1, 10)), "F1 Score": [0.8342, 0.8077, 0.8058, 0.8190, 0.8230, 0.8173, 0.8246, 0.8365, 0.8286]})
    st.line_chart(history.set_index("Round"), use_container_width=True, height=300)

elif page == "System Architecture":
    st.markdown("## System Architecture")
    st.markdown("<p class='subtitle'>How federated learning works</p>", unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
    <p style='font-size:14px; line-height:1.7; color:#1a1f36;'>
    <strong>1. Initialization</strong><br>Server creates global model and distributes to all institutions.<br><br>
    <strong>2. Local Training</strong><br>Each institution trains on its own data. Raw data never leaves.<br><br>
    <strong>3. Weight Sharing</strong><br>Only model weights are sent back to server.<br><br>
    <strong>4. Aggregation (FedAvg)</strong><br>Server computes weighted average of updates.<br><br>
    <strong>5. Global Update</strong><br>Updated model is redistributed. Process repeats.
    </p>
    </div>
    """, unsafe_allow_html=True)
