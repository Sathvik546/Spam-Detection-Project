import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import re
import tldextract
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Phishing Defense AI",
    page_icon="🛡️",
    layout="centered"
)

# -----------------------------------------------------------------------------
# 2. MACHINE LEARNING ENGINE (Trained on your dataset)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_and_train_model():
    """Trains a Machine Learning model using your 10,000 message dataset."""
    try:
        # Tries to load the dataset you uploaded to GitHub
        if os.path.exists('sms_spam_10000_dataset.csv'):
            df = pd.read_csv('sms_spam_10000_dataset.csv')
            
            # Handle different common column names in datasets
            if 'Category' in df.columns and 'Message' in df.columns:
                X = df['Message']
                y = df['Category'].apply(lambda x: 1 if str(x).lower() == 'spam' else 0)
            elif 'v1' in df.columns and 'v2' in df.columns:
                X = df['v2']
                y = df['v1'].apply(lambda x: 1 if str(x).lower() == 'spam' else 0)
            else:
                return None, "Dataset columns not recognized. Ensure columns are 'Category' and 'Message'."

            # Create an AI Pipeline (TF-IDF + Logistic Regression for better probability calibration)
            model = Pipeline([
                # ngram_range=(1,2) helps the model learn phrases like "act now" or "bank account"
                ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))),
                ('clf', LogisticRegression(max_iter=1000)) 
            ])
            
            model.fit(X, y)
            return model, "✅ AI Model loaded and trained on 10,000 messages."
        else:
            return None, "Dataset file 'sms_spam_10000_dataset.csv' not found in repository."
    except Exception as e:
        return None, f"Error training model: {e}"

# Load the AI Brain
ai_model, system_status = load_and_train_model()

# -----------------------------------------------------------------------------
# 3. URL FORENSICS ENGINE
# -----------------------------------------------------------------------------
def analyze_url_accuracy(url):
    """A highly accurate, rule-based lexical analyzer for URLs."""
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
        
    ext = tldextract.extract(url)
    domain = ext.domain.lower()
    suffix = ext.suffix.lower()
    subdomain = ext.subdomain.lower()
    
    score = 0
    flags = []
    
    # 1. Whitelist Safe Domains (Prevents False Positives)
    safe_domains = ['google', 'youtube', 'facebook', 'github', 'linkedin', 'microsoft', 'apple', 'amazon', 'wikipedia']
    if domain in safe_domains:
        return 0, "SAFE", ["Verified Safe Domain (Whitelist)"]

    # 2. IP Address Check (Critical Phishing Trait)
    if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain):
        score += 85
        flags.append("🚨 Domain is a hidden IP Address (Very High Risk)")

    # 3. Subdomain Anomalies
    subdomain_parts = subdomain.split('.')
    if len(subdomain_parts) >= 3 and subdomain != '':
        score += 40
        flags.append(f"Excessive subdomains detected ({len(subdomain_parts)} parts)")
        
    if 'www' in subdomain_parts and len(subdomain_parts) > 1:
        if any(brand in subdomain for brand in safe_domains):
            score += 60
            flags.append("Brand impersonation in subdomain (e.g., google.login-secure.com)")

    # 4. Length and Characters
    if len(url) > 75:
        score += 20
        flags.append("Unusually long URL structure")
    if url.count('-') > 3:
        score += 25
        flags.append("Excessive hyphens (Common in phishing sites)")
    if '@' in url:
        score += 80
        flags.append("🚨 Contains '@' symbol (Browser credential spoofing)")

    # 5. Suspicious TLDs
    bad_tlds = ['xyz', 'top', 'club', 'online', 'vip', 'click', 'tk', 'ml']
    if suffix in bad_tlds:
        score += 45
        flags.append(f"Suspicious Top-Level Domain (.{suffix})")

    # 6. Sensitive Slugs
    if any(keyword in url.lower() for keyword in ['login', 'verify', 'update', 'secure', 'account', 'banking']):
        score += 20
        flags.append("URL contains sensitive action keywords")

    # Final Calculation
    score = min(score, 100)
    
    if score >= 60: level = "CRITICAL"
    elif score >= 35: level = "SUSPICIOUS"
    else: level = "SAFE"
    
    if score == 0: flags.append("URL structure is clean and standard.")
    
    return score, level, flags

# -----------------------------------------------------------------------------
# 4. CLAYMORPHISM CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
.stApp { background-color: #f0f4f8; font-family: 'Nunito', sans-serif; }
.minimal-card {
    background-color: #f0f4f8;
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 24px;
    box-shadow: 10px 10px 20px #d1d5db, -10px -10px 20px #ffffff;
}
.stTextInput > div > div > input, .stTextArea > div > div > textarea {
    background-color: #f0f4f8 !important;
    border: none !important;
    border-radius: 12px !important;
    box-shadow: inset 5px 5px 10px #d1d5db, inset -5px -5px 10px #ffffff !important;
    padding: 15px !important;
}
div.stButton > button {
    background-color: #f0f4f8 !important;
    color: #3182ce !important;
    font-weight: bold !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 12px 24px !important;
    box-shadow: 6px 6px 12px #d1d5db, -6px -6px 12px #ffffff !important;
    width: 100%;
}
div.stButton > button:hover {
    box-shadow: inset 4px 4px 8px #d1d5db, inset -4px -4px 8px #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. MAIN APP UI
# -----------------------------------------------------------------------------
st.markdown("<h1 style='text-align:center; color:#2d3748;'>🛡️ CyberSafe AI</h1>", unsafe_allow_html=True)
st.caption(f"<div style='text-align:center;'>{system_status}</div>", unsafe_allow_html=True)

selected = option_menu(
    menu_title=None,
    options=["Text/SMS Analyzer", "URL Analyzer"],
    icons=["chat-left-text", "globe"],
    orientation="horizontal",
    styles={
        "container": {"background-color": "#f0f4f8", "border-radius": "15px", "box-shadow": "5px 5px 10px #d1d5db, -5px -5px 10px #ffffff"},
        "nav-link-selected": {"background-color": "#f0f4f8", "color": "#3182ce", "box-shadow": "inset 3px 3px 6px #d1d5db, inset -3px -3px 6px #ffffff"},
    }
)

if selected == "Text/SMS Analyzer":
    st.markdown("<div class='minimal-card'><h3>💬 Phishing Message Detector</h3>", unsafe_allow_html=True)
    msg_input = st.text_area("Paste SMS, Email, or DM here:", height=150)
    
    if st.button("Analyze Message"):
        if not msg_input.strip():
            st.warning("Please enter a message.")
        elif ai_model is None:
            st.error("AI Model failed to load. Check if your dataset is in the repository.")
        else:
            with st.spinner("AI is analyzing context..."):
                # Predict probability
                prob = ai_model.predict_proba([msg_input])[0][1]
                risk_score = int(prob * 100)
                
                # Visuals (Adjusted thresholds for higher accuracy on safe messages)
                if risk_score >= 70:
                    color, level = "#e53e3e", "CRITICAL RISK"
                elif risk_score >= 45:
                    color, level = "#dd6b20", "SUSPICIOUS"
                else:
                    color, level = "#38a169", "SAFE"
                
                st.markdown(f"""
                <div style="text-align:center; padding:20px;">
                    <h1 style="color:{color}; font-size:4rem; margin:0;">{risk_score}%</h1>
                    <h3 style="color:{color}; margin:0;">{level}</h3>
                    <p style="color:#718096; margin-top:10px;">Powered by Machine Learning (10k Dataset)</p>
                </div>
                """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif selected == "URL Analyzer":
    st.markdown("<div class='minimal-card'><h3>🔗 Link Forensics</h3>", unsafe_allow_html=True)
    url_input = st.text_input("Paste web address here:")
    
    if st.button("Scan URL"):
        if not url_input.strip():
            st.warning("Please enter a URL.")
        else:
            with st.spinner("Analyzing domain structure..."):
                score, level, flags = analyze_url_accuracy(url_input)
                
                if score >= 60: color = "#e53e3e"
                elif score >= 35: color = "#dd6b20"
                else: color = "#38a169"
                
                st.markdown(f"""
                <div style="text-align:center; padding:10px;">
                    <h2 style="color:{color}; font-size:3rem; margin:0;">{score}%</h2>
                    <h4 style="color:{color}; margin:0;">{level}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.write("**Detection Flags:**")
                for f in flags:
                    if score == 0: st.success(f"✅ {f}")
                    else: st.error(f"🚩 {f}")
    st.markdown("</div>", unsafe_allow_html=True)
