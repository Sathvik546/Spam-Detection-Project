# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from datetime import datetime
import pandas as pd

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="📧",
    layout="wide"
)

# -----------------------------
# LOAD MODEL & TOKENIZER
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("spam_model.keras")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer_obj = pickle.load(handle)
    return tokenizer_obj

model = load_model()
tokenizer = load_tokenizer()
MAX_LEN = 100  # must match training

# -----------------------------
# LOAD DATASET (OPTIONAL, FOR INSIGHTS TAB)
# -----------------------------
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("SPAM text message 20170820 - Data.csv", encoding="latin-1")
        # Adjust column names if needed
        if {"Category", "Message"}.issubset(set(df.columns)):
            df = df[["Category", "Message"]].dropna()
        return df
    except FileNotFoundError:
        return None

dataset_df = load_dataset()

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def predict_spam(message: str) -> float:
    """Return spam probability between 0 and 1 for a single message."""
    seq = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    pred = model.predict(padded, verbose=0)
    return float(pred[0][0])


def bulk_predict(messages):
    """Takes a list of messages and returns a DataFrame with predictions."""
    seqs = tokenizer.texts_to_sequences(messages)
    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
    preds = model.predict(padded, verbose=0).reshape(-1)
    return preds


def init_history():
    if "history" not in st.session_state:
        st.session_state.history = []


def add_to_history(source: str, message: str, score: float, label: str):
    init_history()
    st.session_state.history.append(
        {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Source": source,
            "Message": message,
            "Spam Probability": round(score, 3),
            "Prediction": label,
        }
    )


def get_history_df():
    init_history()
    if not st.session_state.history:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.history)


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("⚙️ Settings")

threshold = st.sidebar.slider(
    "Spam Threshold",
    0.1, 0.9, 0.5, 0.05,
    help="If spam probability ≥ this value, the message is marked as SPAM."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📘 Model Details")
st.sidebar.markdown(
    """
**Architecture**
- Embedding (vocab size 10k, dim 64)  
- Bidirectional LSTM (64 units)  
- Dense (32 units, ReLU)  
- Output (sigmoid)

**Task:** SMS spam / ham classification  
**Framework:** TensorFlow / Keras  
**Test Accuracy:** ~98% (on held-out data)
"""
)

st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Quick Samples")
sample_spam_btn = st.sidebar.button("Insert sample SPAM")
sample_ham_btn = st.sidebar.button("Insert sample HAM")

st.sidebar.markdown("---")
# History download
history_df_for_download = get_history_df()
if not history_df_for_download.empty:
    csv_data = history_df_for_download.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        label="⬇️ Download Prediction History (CSV)",
        data=csv_data,
        file_name="spam_detection_history.csv",
        mime="text/csv",
    )

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    """
<div style="background-color:#0f4c75;padding:18px;border-radius:14px;margin-bottom:18px;">
  <h2 style="color:white;text-align:center;margin-bottom:4px;">📧 Spam Message Detection Dashboard</h2>
  <p style="color:#eeeeee;text-align:center;margin:0;">
    Analyze individual SMS messages or bulk data using a BiLSTM-based spam classifier.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Single Message", "📂 Bulk Analysis", "📊 Dataset Insights"])

# -----------------------------
# TAB 1: SINGLE MESSAGE
# -----------------------------
with tab1:
    st.subheader("🔍 Single Message Classification")

    default_text = ""
    if sample_spam_btn:
        default_text = "Congratulations! You have won a free lottery ticket. Click this link now to claim your prize."
    elif sample_ham_btn:
        default_text = "Hey, I will reach college by 10 AM. Wait near the main gate."

    user_input = st.text_area(
        "✍️ Enter your SMS message:",
        value=default_text,
        height=150,
        placeholder="Example: 'Dear user, your account has been selected for a special reward...'",
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        single_predict_btn = st.button("🔎 Analyze", use_container_width=True)
    with col2:
        clear_history_btn = st.button("🧹 Clear History", use_container_width=True)
    with col3:
        show_raw_history_btn = st.button("📜 Show History Table", use_container_width=True)

    if clear_history_btn:
        st.session_state.history = []
        st.success("History cleared.")

    if single_predict_btn:
        if not user_input.strip():
            st.warning("Please enter a message before analyzing.", icon="⚠️")
        else:
            score = predict_spam(user_input)
            label = "SPAM" if score >= threshold else "HAM"

            # Result box
            if label == "SPAM":
                st.error(
                    f"🚫 This looks like **SPAM**.\n\n"
                    f"- Spam probability: **{score:.2f}**\n"
                    f"- Threshold: **{threshold:.2f}**",
                    icon="🚫",
                )
            else:
                st.success(
                    f"✅ This looks like **HAM (Not Spam)**.\n\n"
                    f"- Spam probability: **{score:.2f}**\n"
                    f"- Threshold: **{threshold:.2f}**",
                    icon="✅",
                )

            # Probability bar chart
            st.subheader("📊 Probability")
            st.bar_chart(
                data={
                    "Spam": [score],
                    "Ham": [1 - score],
                }
            )

            add_to_history("Single", user_input, score, label)

    if show_raw_history_btn:
        hist_df = get_history_df()
        if hist_df.empty:
            st.info("No predictions yet to show in history.")
        else:
            st.subheader("📜 Prediction History (All Sources)")
            st.dataframe(hist_df, use_container_width=True)

# -----------------------------
# TAB 2: BULK ANALYSIS
# -----------------------------
with tab2:
    st.subheader("📂 Bulk Message Classification")

    st.write("You can either:")
    st.markdown("- Paste multiple messages (one per line), **or**")
    st.markdown("- Upload a CSV file with a **'Message'** column.")

    bulk_input_method = st.radio("Choose input method:", ["📝 Paste text", "📁 Upload CSV"], horizontal=True)

    messages = []

    if bulk_input_method == "📝 Paste text":
        bulk_text = st.text_area(
            "Paste messages (one per line):",
            height=200,
            placeholder="Line 1: message 1\nLine 2: message 2\nLine 3: message 3\n..."
        )
        if bulk_text.strip():
            messages = [line.strip() for line in bulk_text.split("\n") if line.strip()]
    else:
        uploaded_file = st.file_uploader("Upload CSV file with a 'Message' column", type=["csv"])
        if uploaded_file is not None:
            try:
                upload_df = pd.read_csv(uploaded_file)
                if "Message" in upload_df.columns:
                    messages = upload_df["Message"].astype(str).tolist()
                    st.success(f"Loaded {len(messages)} messages from CSV.")
                    st.dataframe(upload_df.head(), use_container_width=True)
                else:
                    st.error("CSV must contain a column named 'Message'.")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    if st.button("🚀 Run Bulk Classification", use_container_width=True):
        if not messages:
            st.warning("Please provide at least one message.", icon="⚠️")
        else:
            preds = bulk_predict(messages)
            labels = ["SPAM" if p >= threshold else "HAM" for p in preds]
            results_df = pd.DataFrame(
                {
                    "Message": messages,
                    "Spam Probability": preds,
                    "Prediction": labels,
                }
            )

            st.subheader("📊 Bulk Classification Results")
            st.dataframe(results_df, use_container_width=True)

            # Summary counts
            spam_count = (results_df["Prediction"] == "SPAM").sum()
            ham_count = (results_df["Prediction"] == "HAM").sum()
            st.write(f"**Summary:** SPAM: {spam_count} | HAM: {ham_count}")

            # Add to history
            for msg, score, lab in zip(messages, preds, labels):
                add_to_history("Bulk", msg, float(score), lab)

            # Download results
            csv_bulk = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv_bulk,
                file_name="bulk_spam_classification_results.csv",
                mime="text/csv",
            )

# -----------------------------
# TAB 3: DATASET INSIGHTS
# -----------------------------
with tab3:
    st.subheader("📊 Dataset Insights")

    if dataset_df is None:
        st.warning(
            "Could not find 'SPAM text message 20170820 - Data.csv' in the app folder. "
            "Place the dataset file next to app.py to see insights.",
            icon="⚠️",
        )
    else:
        st.write("Basic statistics about the dataset used to train the model:")

        total = len(dataset_df)
        spam_count = (dataset_df["Category"].str.lower() == "spam").sum()
        ham_count = (dataset_df["Category"].str.lower() == "ham").sum()
        spam_pct = (spam_count / total) * 100 if total > 0 else 0

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Messages", total)
        col_b.metric("Spam Messages", spam_count)
        col_c.metric("Spam Percentage", f"{spam_pct:.1f}%")

        st.markdown("### 📊 Spam vs Ham Count")
        count_df = pd.DataFrame(
            {"Label": ["HAM", "SPAM"], "Count": [ham_count, spam_count]}
        ).set_index("Label")
        st.bar_chart(count_df)

        st.markdown("### 🔎 Sample Messages")
        st.dataframe(dataset_df.sample(min(10, len(dataset_df))), use_container_width=True)

st.caption("🔒 All predictions are computed locally on your machine using your trained BiLSTM model.")
