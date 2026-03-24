import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import re
import urllib.parse
import ipaddress
import string
import cv2
import easyocr
import io
from PIL import Image

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Phishing Defense",
    page_icon="🛡️",
    layout="centered"
)

# -----------------------------------------------------------------------------
# 2. ADVANCED HEURISTIC ENGINE (Local Intelligence)
# -----------------------------------------------------------------------------
class PhishingAnalyzer:
    def __init__(self):
        # Dictionary of threat signatures
        self.threat_db = {
            "urgency": [
                "urgent", "immediate", "act now", "suspended", "locked", "restricted", 
                "unauthorized", "verify", "breach", "expires", "24 hours", "final notice",
                "compromised", "action required"
            ],
            "financial": [
                "wire", "transfer", "bank", "credit card", "debit", "irs", "tax", 
                "refund", "bitcoin", "crypto", "payment", "invoice", "balance", "withdrawal"
            ],
            "scam_apps": [
                "anydesk", "teamviewer", "zoho", "connectwise", "ultraviewer", 
                "logmein", "remote support"
            ],
            "authority": [
                "police", "fbi", "warrant", "arrest", "legal action", "law enforcement", 
                "federal", "bureau", "social security", "ssa"
            ],
            "social_red_flags": [
                "dm me", "direct message", "whatsapp", "telegram", "gift", "winner", 
                "congratulations", "invest", "forex", "mentor"
            ]
        }
        
        # Typosquatting targets and their common spoofs
        self.brands = {
            "paypal": ["paypa1", "pypal", "pay-pal", "paypal-secure", "security-paypal"],
            "google": ["goggle", "googie", "gooogle", "google-security", "drive-google"],
            "amazon": ["arnazon", "amaz0n", "amazon-prime-support", "amzon"],
            "facebook": ["faceb0ok", "face-book", "meta-security", "meta-support"],
            "apple": ["appie", "apple-id", "icloud-verify"],
            "netflix": ["net-flix", "netflix-update"],
            "bank": ["secure-banking", "account-update"]
        }

    def calculate_risk_level(self, score):
        if score >= 85: return "CRITICAL"
        if score >= 60: return "HIGH"
        if score >= 35: return "MEDIUM"
        if score > 0: return "LOW"
        return "SAFE"

    def analyze_url(self, url):
        score = 0
        flags = []
        
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.hostname.lower() if parsed.hostname else ""
            path = parsed.path.lower()
        except:
            return {"score": 0, "level": "Invalid", "flags": ["Invalid URL format"], "reason": "Could not parse URL."}

        try:
            ipaddress.ip_address(domain)
            score += 80
            flags.append("Domain is a raw IP address (Highly Suspicious)")
        except ValueError:
            pass

        if len(domain) > 50:
            score += 20
            flags.append("Extremely long domain name")
        
        if domain.count('-') > 2:
            score += 30
            flags.append("Excessive use of hyphens in domain")
            
        if '@' in url:
            score += 90
            flags.append("Contains '@' symbol (Authentication bypass attempt)")

        brand_detected = False
        for brand, variations in self.brands.items():
            if brand in path and brand not in domain:
                score += 40
                flags.append(f"Brand '{brand}' found in URL path, not domain")
            
            if brand in domain and not domain.endswith(f".{brand}.com") and domain != f"{brand}.com":
                score += 50
                flags.append(f"Suspicious use of '{brand}' in subdomain")

            for var in variations:
                if var in domain:
                    score += 85
                    flags.append(f"Typosquatting detected: '{var}' mimics '{brand}'")
                    brand_detected = True
        
        suspicious_tlds = ['.xyz', '.top', '.club', '.info', '.cn', '.ru', '.gq', '.ml']
        if any(domain.endswith(tld) for tld in suspicious_tlds):
            score += 20
            flags.append("Uses a TLD often associated with spam")

        if score == 0:
            reason = "URL structure appears standard. No obvious heuristic threats detected."
        else:
            reason = "Multiple structural anomalies detected."

        return {
            "score": min(score, 100),
            "level": self.calculate_risk_level(score),
            "flags": flags,
            "reason": reason
        }

    def analyze_text(self, text, context="general"):
        score = 0
        flags = []
        text_lower = text.lower()
        
        urgency_count = sum(1 for w in self.threat_db["urgency"] if w in text_lower)
        financial_count = sum(1 for w in self.threat_db["financial"] if w in text_lower)
        
        if urgency_count > 0:
            score += 25 + (urgency_count * 5)
            flags.append("Contains high-pressure urgency language")
            
        if financial_count > 0:
            score += 25 + (financial_count * 5)
            flags.append("Requests financial information or payment")

        has_link = bool(re.search(r'http[s]?://|www\.', text_lower))
        
        if context == "SMS":
            if has_link and urgency_count > 0:
                score += 40
                flags.append("Smishing Indicator: Urgency + Link")
            if "package" in text_lower or "delivery" in text_lower:
                if has_link:
                    score += 30
                    flags.append("Fake Delivery Scam pattern")

        if context == "SOCIAL":
            for phrase in self.threat_db["social_red_flags"]:
                if phrase in text_lower:
                    score += 30
                    flags.append(f"Suspicious social phrase: '{phrase}'")
            if "fill" in text_lower and "form" in text_lower:
                score += 30
                flags.append("Request to fill external form")

        if context == "VOICE":
            scam_apps = sum(1 for w in self.threat_db["scam_apps"] if w in text_lower)
            auth_apps = sum(1 for w in self.threat_db["authority"] if w in text_lower)
            
            if scam_apps > 0:
                score += 80
                flags.append("Tech Support Scam: Remote Access Software mentioned")
            if auth_apps > 0:
                score += 70
                flags.append("Authority Impersonation (Police/Gov)")
            if "gift card" in text_lower:
                score += 90
                flags.append("Demanding payment via Gift Cards")

        final_score = min(score, 100)
        
        reason = "Analysis complete based on behavioral heuristics."
        if final_score < 20: reason = "Content appears normal."
        elif final_score < 50: reason = "Some cautionary language detected."
        else: reason = "Significant indicators of social engineering present."

        return {
            "score": final_score,
            "level": self.calculate_risk_level(final_score),
            "flags": flags,
            "reason": reason
        }

    def analyze_social_profile(self, handle, message):
        score = 0
        flags = []
        handle_lower = handle.lower()
        
        if "support" in handle_lower or "help" in handle_lower or "service" in handle_lower:
            if re.search(r'\d{3,}$', handle_lower):
                score += 60
                flags.append("Fake Support Handle: Ends in random numbers")
            
            if handle_lower.count('_') > 1:
                score += 30
                flags.append("Suspicious handle formatting")
        
        text_result = self.analyze_text(message, context="SOCIAL")
        score += text_result["score"]
        flags.extend(text_result["flags"])
        
        return {
            "score": min(score, 100),
            "level": self.calculate_risk_level(min(score, 100)),
            "flags": flags,
            "reason": text_result["reason"]
        }

analyzer = PhishingAnalyzer()

# -----------------------------------------------------------------------------
# 3. CSS (Light Minimalist)
# -----------------------------------------------------------------------------
light_mode_css = """
<style>
.stApp {
    background-color: #ffffff;
    color: #111827;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.minimal-card {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 32px;
    margin-bottom: 24px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    transition: all 0.2s ease;
}
.minimal-card:hover {
    border-color: #d1d5db;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}
.minimal-card h3 {
    font-weight: 600;
    font-size: 1.125rem;
    color: #111827;
    margin-bottom: 8px;
}
.minimal-card p {
    font-size: 0.875rem;
    color: #6b7280;
    line-height: 1.5;
}
.stTextInput > div > div > input, 
.stTextArea > div > div > textarea {
    background-color: #f9fafb !important;
    border: 1px solid #d1d5db !important;
    border-radius: 6px !important;
    color: #111827 !important;
    padding: 10px 12px !important;
    font-size: 0.95rem !important;
}
.stTextInput > div > div > input:focus, 
.stTextArea > div > div > textarea:focus {
    border-color: #000000 !important;
    box-shadow: 0 0 0 1px #000000 !important;
}
div.stButton > button {
    background-color: #111827 !important;
    color: #ffffff !important;
    border: 1px solid #111827 !important;
    border-radius: 6px !important;
    padding: 10px 24px !important;
    font-weight: 500 !important;
}
div.stButton > button:hover {
    background-color: #374151 !important;
    border-color: #374151 !important;
}
div[data-testid="stMetricValue"] {
    font-weight: 600 !important;
    font-size: 2rem !important;
    color: #111827 !important;
}
div[data-testid="stMetricLabel"] {
    color: #6b7280 !important;
}
header {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(light_mode_css, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. UI HELPERS
# -----------------------------------------------------------------------------
def card_start(title=None):
    html = '<div class="minimal-card">'
    if title: html += f'<h3>{title}</h3>'
    st.markdown(html, unsafe_allow_html=True)

def card_end():
    st.markdown('</div>', unsafe_allow_html=True)

def display_results(result):
    if not result: return
    st.divider()
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Risk Score", f"{result['score']}")
        level = result['level']
        color = "#10b981" 
        if level in ["HIGH", "CRITICAL"]: color = "#ef4444"
        elif level == "MEDIUM": color = "#f97316"
        st.markdown(f"<span style='color:{color}; font-weight:700; letter-spacing:0.05em;'>{level} RISK</span>", unsafe_allow_html=True)
    with c2:
        st.subheader("Details")
        if not result['flags']:
            st.success("No specific threat patterns detected.")
        else:
            for f in result['flags']:
                st.markdown(f"🚩 {f}")
        st.caption(f"Reasoning: {result['reason']}")

# -----------------------------------------------------------------------------
# 5. MAIN APP
# -----------------------------------------------------------------------------
st.markdown("<h1 style='color:#111827; font-weight:700; letter-spacing:-0.03em;'>Phishing Defense</h1>", unsafe_allow_html=True)
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

selected = option_menu(
    menu_title=None,
    options=["Dashboard", "URL Scanner", "Smishing", "Social", "Vishing", "Forensics"],
    icons=["grid", "globe", "chat-text", "people", "mic", "camera"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "transparent"},
        "icon": {"color": "#6b7280", "font-size": "14px"}, 
        "nav-link": {"font-size": "13px", "text-align": "center", "margin": "0px 8px 0px 0px", "color": "#4b5563", "background-color": "transparent"},
        "nav-link-selected": {"background-color": "#f3f4f6", "color": "#111827", "font-weight": "600"},
    }
)
st.write("")

if selected == "Dashboard":
    st.markdown("<div class='minimal-card' style='text-align:center; padding:40px;'><h3 style='font-size:1.5rem;'>Unified Defense Suite</h3><p>Advanced Heuristic Detection • No API Keys Required</p></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        card_start("URL Scanner")
        st.write("Detects typosquatting, IP masking, and malicious structures.")
        card_end()
    with c2:
        card_start("Smishing")
        st.write("Analyzes SMS for urgency, financial keywords, and links.")
        card_end()
    with c3:
        card_start("Social Media")
        st.write("Identifies fake support handles and angler phishing.")
        card_end()
    c4, c5 = st.columns(2)
    with c4:
        card_start("Voice Analysis")
        st.write("Scans transcripts for Tech Support & IRS scam scripts.")
        card_end()
    with c5:
        card_start("Forensics")
        st.write("OCR & QR Code extraction with automated scanning.")
        card_end()

elif selected == "URL Scanner":
    card_start()
    st.markdown("### URL Scanner")
    url_input = st.text_input("Enter URL", placeholder="http://example.com", label_visibility="collapsed")
    if st.button("Scan Link"):
        if url_input:
            res = analyzer.analyze_url(url_input)
            display_results(res)
        else:
            st.warning("Input required.")
    card_end()

elif selected == "Smishing":
    card_start()
    st.markdown("### Smishing Detector")
    txt = st.text_area("Message", placeholder="Paste message...", height=120, label_visibility="collapsed")
    if st.button("Analyze SMS"):
        if txt:
            res = analyzer.analyze_text(txt, context="SMS")
            display_results(res)
        else:
            st.warning("Input required.")
    card_end()

elif selected == "Social":
    card_start()
    st.markdown("### Social Media Scanner")
    c1, c2 = st.columns(2)
    with c1:
        h = st.text_input("Handle", placeholder="@PayPal_Support_123")
    with c2:
        m = st.text_input("Message", placeholder="DM us for help...")
    if st.button("Scan Profile"):
        if h and m:
            res = analyzer.analyze_social_profile(h, m)
            display_results(res)
        else:
            st.warning("Both handle and message are required.")
    card_end()

elif selected == "Vishing":
    card_start()
    st.markdown("### Voice Transcript")
    v_txt = st.text_area("Transcript", placeholder="Transcript...", height=150, label_visibility="collapsed")
    if st.button("Scan Transcript"):
        if v_txt:
            res = analyzer.analyze_text(v_txt, context="VOICE")
            display_results(res)
        else:
            st.warning("Input required.")
    card_end()

elif selected == "Forensics":
    @st.cache_resource
    def load_ocr():
        return easyocr.Reader(['en'], gpu=False)

    card_start()
    st.markdown("### Forensics")
    uf = st.file_uploader("Upload", type=['png','jpg','jpeg'], label_visibility="collapsed")
    if uf:
        try:
            file_bytes = np.asarray(bytearray(uf.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            if img is not None:
                st.image(img, channels="BGR", width=300)
                st.divider()
                try:
                    det = cv2.QRCodeDetector()
                    data, _, _ = det.detectAndDecode(img)
                    if data:
                        st.info(f"QR Data: {data}")
                        q_res = analyzer.analyze_url(data)
                        display_results(q_res)
                    else:
                        st.caption("No QR Code.")
                except: pass
                st.markdown("---")
                try:
                    with st.spinner("Extracting text..."):
                        reader = load_ocr()
                        raw_res = reader.readtext(img, detail=0)
                        extracted = " ".join(raw_res)
                        if extracted.strip():
                            st.text_area("Extracted Text", extracted, height=100)
                            ocr_res = analyzer.analyze_text(extracted, context="SMS")
                            display_results(ocr_res)
                        else:
                            st.info("No readable text found.")
                except Exception as e:
                    st.error(f"OCR Error: {e}")
        except:
            st.error("File error.")
    card_end()
