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
from urllib.parse import urlparse

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PhishGuard",
    page_icon="🛡️",
    layout="centered"
)

# ─────────────────────────────────────────────
# MACHINE LEARNING ENGINE
# ─────────────────────────────────────────────
@st.cache_resource
def load_and_train_model():
    try:
        if os.path.exists('sms_spam_10000_dataset.csv'):
            df = pd.read_csv('sms_spam_10000_dataset.csv')
            if 'Category' in df.columns and 'Message' in df.columns:
                X, y = df['Message'], df['Category'].apply(lambda x: 1 if str(x).lower() == 'spam' else 0)
            elif 'v1' in df.columns and 'v2' in df.columns:
                X, y = df['v2'], df['v1'].apply(lambda x: 1 if str(x).lower() == 'spam' else 0)
            else:
                return None, "Dataset columns not recognized."
            model = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english', max_features=8000, ngram_range=(1, 3))),
                ('clf', LogisticRegression(max_iter=2000, C=1.5))
            ])
            model.fit(X, y)
            return model, "Model trained on dataset."
        else:
            # Fallback: built-in heuristic engine only
            return None, "No dataset found — using heuristic engine."
    except Exception as e:
        return None, f"Error: {e}"

ai_model, system_status = load_and_train_model()

# ─────────────────────────────────────────────
# SAFE DOMAIN ALTERNATIVES MAP
# ─────────────────────────────────────────────
SAFE_ALTERNATIVES = {
    "bank": ["Chase (chase.com)", "Bank of America (bankofamerica.com)", "Wells Fargo (wellsfargo.com)"],
    "paypal": ["paypal.com (official)"],
    "amazon": ["amazon.com (official)"],
    "netflix": ["netflix.com (official)"],
    "google": ["google.com (official)"],
    "apple": ["apple.com (official)"],
    "microsoft": ["microsoft.com (official)"],
    "facebook": ["facebook.com (official)"],
    "instagram": ["instagram.com (official)"],
    "twitter": ["twitter.com / x.com (official)"],
    "login": ["Visit the official site by typing it directly into your browser."],
    "verify": ["Contact the company via their official phone number or website."],
    "secure": ["Use the official app from the App Store or Google Play."],
    "crypto": ["Coinbase (coinbase.com)", "Binance (binance.com)", "Kraken (kraken.com)"],
    "default": [
        "Type the official website directly into your browser.",
        "Search on Google and look for the verified checkmark.",
        "Use the official app from the App Store or Google Play.",
        "Contact the company directly via their official support number."
    ]
}

def get_safe_alternatives(url, flags_text):
    """Return contextual safe alternatives based on the URL/flags."""
    url_lower = url.lower()
    combined = url_lower + " " + flags_text.lower()
    alts = []
    for keyword, suggestions in SAFE_ALTERNATIVES.items():
        if keyword != "default" and keyword in combined:
            alts.extend(suggestions)
    if not alts:
        alts = SAFE_ALTERNATIVES["default"]
    return list(dict.fromkeys(alts))  # deduplicate

# ─────────────────────────────────────────────
# URL FORENSICS ENGINE
# ─────────────────────────────────────────────
WHITELISTED = {
    'google', 'youtube', 'facebook', 'github', 'linkedin',
    'microsoft', 'apple', 'amazon', 'wikipedia', 'twitter',
    'instagram', 'netflix', 'spotify', 'adobe', 'dropbox',
    'paypal', 'reddit', 'stackoverflow', 'openai', 'anthropic'
}

SUSPICIOUS_TLDS = {'xyz', 'top', 'club', 'online', 'vip', 'click', 'tk', 'ml', 'cf', 'ga', 'gq', 'work', 'rest', 'date', 'download'}
SENSITIVE_KEYWORDS = ['login', 'verify', 'update', 'secure', 'account', 'banking', 'signin', 'password', 'confirm', 'validate', 'wallet', 'payment', 'credential']
BRAND_NAMES = ['paypal', 'amazon', 'apple', 'google', 'microsoft', 'facebook', 'netflix', 'instagram', 'twitter', 'bank', 'chase', 'wells']

def analyze_url(url):
    """Deep lexical and structural URL analysis. Returns (score 0-100, level, list of (reason, detail))."""
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url

    ext = tldextract.extract(url)
    domain = ext.domain.lower()
    suffix = ext.suffix.lower()
    subdomain = ext.subdomain.lower()
    parsed = urlparse(url)
    full_url = url.lower()

    score = 0
    flags = []  # list of (short_label, explanation)

    # 1. Whitelist check
    if domain in WHITELISTED and subdomain in ('', 'www', 'mail', 'm'):
        return 0, "SAFE", [("Verified Domain", "This domain is on the trusted whitelist.")]

    # 2. IP address as host (critical)
    if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain):
        score += 90
        flags.append(("IP Address Used as Domain", "Phishing sites often use raw IP addresses (e.g. 192.168.1.1) to hide their real identity instead of a proper domain name."))

    # 3. HTTPS check
    if parsed.scheme == 'http':
        score += 15
        flags.append(("No HTTPS Encryption", "The link uses HTTP, not HTTPS. Legitimate sites — especially those handling accounts or payments — always use HTTPS."))

    # 4. Subdomain abuse
    subdomain_parts = [p for p in subdomain.split('.') if p]
    if len(subdomain_parts) >= 3:
        score += 35
        flags.append(("Excessive Subdomains", f"Found {len(subdomain_parts)} subdomain levels (e.g. login.secure.pay.site.com). This is a common trick to make a fake URL appear official."))

    # 5. Brand name in subdomain (impersonation)
    for brand in BRAND_NAMES:
        if brand in subdomain and domain not in WHITELISTED:
            score += 65
            flags.append(("Brand Impersonation in Subdomain", f"The subdomain contains '{brand}' but the actual domain is '{domain}.{suffix}'. This is a classic impersonation technique where the real domain is at the end."))
            break

    # 6. Brand name in domain path (typosquatting)
    for brand in BRAND_NAMES:
        if brand in domain and domain not in WHITELISTED:
            score += 40
            flags.append(("Possible Typosquatting", f"The domain '{domain}' resembles a known brand name. Attackers register similar-looking domains to trick users."))
            break

    # 7. @ symbol in URL
    if '@' in full_url:
        score += 85
        flags.append(("@ Symbol in URL", "Browsers interpret everything before '@' as credentials. Attackers use this to display a fake-looking domain while redirecting to a malicious one."))

    # 8. Suspicious TLD
    if suffix in SUSPICIOUS_TLDS:
        score += 40
        flags.append(("Suspicious Top-Level Domain", f"The domain uses '.{suffix}' — a TLD commonly associated with free or throwaway domains frequently used in phishing campaigns."))

    # 9. Excessive hyphens
    hyphen_count = domain.count('-')
    if hyphen_count >= 2:
        score += 20 + (hyphen_count * 5)
        flags.append(("Excessive Hyphens in Domain", f"Found {hyphen_count} hyphens in '{domain}'. Phishing domains often use hyphens to mimic legitimate names (e.g. paypal-secure-login.com)."))

    # 10. Long URL
    if len(url) > 80:
        score += 20
        flags.append(("Unusually Long URL", f"The URL is {len(url)} characters long. Long URLs are used to bury the real destination or confuse users."))

    # 11. Sensitive keywords in path
    path = parsed.path.lower()
    found_keywords = [kw for kw in SENSITIVE_KEYWORDS if kw in path or kw in full_url]
    if found_keywords:
        score += 25
        flags.append(("Sensitive Keywords in URL", f"Contains: {', '.join(found_keywords)}. Legitimate sites rarely include these words directly in their URLs."))

    # 12. Numeric characters in domain
    digit_ratio = sum(c.isdigit() for c in domain) / max(len(domain), 1)
    if digit_ratio > 0.4:
        score += 30
        flags.append(("High Digit Ratio in Domain", f"About {int(digit_ratio*100)}% of the domain name consists of numbers. Randomly generated phishing domains often have this pattern."))

    # 13. Redirect indicators
    if 'redirect' in full_url or 'url=' in full_url or 'goto=' in full_url:
        score += 30
        flags.append(("Open Redirect Detected", "The URL contains a redirect parameter. Attackers chain URLs so the first link looks safe but redirects to a malicious page."))

    score = min(score, 100)

    if score >= 60:
        level = "PHISHING"
    elif score >= 30:
        level = "SUSPICIOUS"
    else:
        level = "SAFE"

    if not flags:
        flags.append(("Clean URL Structure", "No suspicious patterns were detected in this URL."))

    return score, level, flags


# ─────────────────────────────────────────────
# TEXT / MESSAGE ANALYSIS
# ─────────────────────────────────────────────
PHISHING_PATTERNS = [
    (r'\b(urgent|immediately|act now|limited time|expires? (today|soon|in \d+ hours?))\b', "Urgency / Pressure Tactics", "Phishing messages manufacture urgency to make you act before you think."),
    (r'\b(won|winner|prize|lottery|congratulations|reward|free gift|selected)\b', "Prize / Lottery Bait", "Fake reward or prize notifications are a classic phishing lure."),
    (r'\b(verify|confirm|validate|update) (your )?(account|identity|information|details|password|card)\b', "Account Verification Request", "Legitimate companies never ask you to verify sensitive data via SMS or email."),
    (r'\b(click|tap|open|follow) (this |the )?(link|url|below|here)\b', "Suspicious Call-to-Action Link", "Directing users to click unverified links is a primary phishing delivery method."),
    (r'\b(bank|credit card|debit card|account|wallet|password|ssn|social security|otp|pin)\b', "Sensitive Financial/Personal Data Request", "Phishing messages frequently ask for banking credentials or personal identifiers."),
    (r'https?://\S+', "URL Embedded in Message", "Links embedded in messages are often disguised to hide their real destination."),
    (r'\b(suspended|blocked|locked|restricted|compromised|hacked|unauthorized)\b', "Account Threat Language", "Threats of account suspension are used to trigger fear and hasty action."),
    (r'\$[\d,]+ ?(million|thousand|hundred)? ?(reward|transfer|prize|cash|check)', "Financial Reward Bait", "Promises of unexpected money transfers are hallmarks of advance-fee fraud."),
    (r'\b(nigerian?|prince|inheritance|diplomat|refugee|widow)\b', "Advance-Fee Fraud Indicators", "Classic advance-fee (419) fraud keywords detected."),
    (r'\b(re-?activate|re-?enable|re-?verify|restore access)\b', "Account Reactivation Scam", "Fake account reactivation requests pressure users to enter credentials on fake pages."),
]

def analyze_message(text):
    """Returns (ml_score, heuristic_score, final_score, level, list of (pattern_name, explanation))."""
    flags = []
    text_lower = text.lower()
    heuristic_score = 0

    for pattern, label, explanation in PHISHING_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            heuristic_score += 18
            flags.append((label, explanation))

    heuristic_score = min(heuristic_score, 100)

    ml_score = 0
    if ai_model:
        prob = ai_model.predict_proba([text])[0][1]
        ml_score = int(prob * 100)
        # Weighted blend: 60% ML, 40% heuristic
        final_score = int(0.60 * ml_score + 0.40 * heuristic_score)
    else:
        final_score = heuristic_score

    if final_score >= 65:
        level = "PHISHING"
    elif final_score >= 35:
        level = "SUSPICIOUS"
    else:
        level = "SAFE"

    return ml_score, heuristic_score, final_score, level, flags


# ─────────────────────────────────────────────
# MINIMAL CSS — Monochrome / Editorial
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background-color: #0d0d0d !important;
    color: #e8e8e8 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 1.5rem 4rem !important; max-width: 720px !important; }

/* Typography */
h1 { font-family: 'DM Mono', monospace !important; font-size: 1.6rem !important; font-weight: 500 !important; letter-spacing: -0.03em; color: #e8e8e8 !important; }
h3, h4 { font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; color: #e8e8e8 !important; }
p, label, .stMarkdown { color: #9a9a9a !important; font-size: 0.9rem !important; }

/* Nav menu */
div[data-testid="stHorizontalBlock"] { gap: 0 !important; }
.nav-container {
    display: flex;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    overflow: hidden;
    margin-bottom: 2rem;
}

/* Inputs */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background-color: #181818 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 6px !important;
    color: #e8e8e8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 14px !important;
    caret-color: #e8e8e8;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #555 !important;
    box-shadow: none !important;
    outline: none !important;
}

/* Button */
div.stButton > button {
    background-color: #e8e8e8 !important;
    color: #0d0d0d !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 10px 0 !important;
    width: 100% !important;
    transition: background 0.15s ease !important;
}
div.stButton > button:hover {
    background-color: #cfcfcf !important;
}

/* Option menu override */
.nav-link { font-family: 'DM Mono', monospace !important; font-size: 0.78rem !important; }

/* Cards */
.card {
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 24px;
    margin-bottom: 16px;
    background: #141414;
}

/* Score display */
.score-block {
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 28px 24px;
    margin: 20px 0;
    background: #111;
    text-align: center;
}
.score-number {
    font-family: 'DM Mono', monospace;
    font-size: 4.5rem;
    font-weight: 300;
    line-height: 1;
    letter-spacing: -0.04em;
}
.score-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 8px;
}

/* Flag items */
.flag-item {
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 14px 16px;
    margin-bottom: 10px;
    background: #131313;
}
.flag-name {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}
.flag-detail {
    font-size: 0.82rem;
    color: #6e6e6e;
    line-height: 1.5;
}

/* Safe alternative */
.alt-item {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #4ade80;
    padding: 8px 12px;
    border-left: 2px solid #4ade80;
    margin-bottom: 8px;
    background: #0d1f14;
    border-radius: 0 4px 4px 0;
}

/* Section label */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #555;
    margin: 20px 0 10px 0;
    border-bottom: 1px solid #222;
    padding-bottom: 6px;
}

/* Score bar */
.bar-outer {
    height: 4px;
    background: #222;
    border-radius: 2px;
    margin-top: 12px;
    overflow: hidden;
}
.bar-inner {
    height: 100%;
    border-radius: 2px;
    transition: width 1s ease;
}

/* Status pill */
.status-pill {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom: 2rem; padding-top: 0.5rem;'>
    <h1>⬡ phishguard</h1>
    <p style='font-family: DM Mono, monospace; font-size: 0.72rem; color: #444; margin-top: 4px; letter-spacing: 0.08em;'>
        PHISHING DETECTION ENGINE — URL · MESSAGE · EMAIL
    </p>
</div>
""", unsafe_allow_html=True)

selected = option_menu(
    menu_title=None,
    options=["Message / Email", "URL / Link"],
    icons=["envelope", "link-45deg"],
    orientation="horizontal",
    styles={
        "container": {"background-color": "#0d0d0d", "border": "1px solid #2a2a2a", "border-radius": "6px", "padding": "0"},
        "nav-link": {"font-family": "DM Mono, monospace", "font-size": "0.75rem", "color": "#555", "padding": "10px 20px", "border-radius": "0"},
        "nav-link-selected": {"background-color": "#1e1e1e", "color": "#e8e8e8", "font-weight": "500"},
        "icon": {"display": "none"},
    }
)

# ─────────────────────────────────────────────
# TAB: MESSAGE / EMAIL
# ─────────────────────────────────────────────
if selected == "Message / Email":
    st.markdown("<div class='section-label'>Input</div>", unsafe_allow_html=True)
    msg_input = st.text_area(
        label="message_input",
        label_visibility="collapsed",
        placeholder="Paste the SMS, email, or DM text here…",
        height=160
    )

    if st.button("ANALYZE →", key="msg_btn"):
        if not msg_input.strip():
            st.warning("Please paste a message first.")
        else:
            with st.spinner(""):
                ml_s, h_s, final_s, level, flags = analyze_message(msg_input)

                if level == "PHISHING":
                    s_color, bg_color, bar_color = "#ef4444", "#1a0808", "#ef4444"
                    pill_bg, pill_color = "#2d0b0b", "#ef4444"
                elif level == "SUSPICIOUS":
                    s_color, bg_color, bar_color = "#f59e0b", "#1a1408", "#f59e0b"
                    pill_bg, pill_color = "#2d2008", "#f59e0b"
                else:
                    s_color, bg_color, bar_color = "#4ade80", "#0d1a11", "#4ade80"
                    pill_bg, pill_color = "#0d1a11", "#4ade80"

                st.markdown(f"""
                <div class='score-block' style='background:{bg_color}; border-color: {s_color}22;'>
                    <div class='score-number' style='color:{s_color};'>{final_s}<span style='font-size:1.5rem; color:#444;'>%</span></div>
                    <div class='score-label' style='color:{s_color};'>{level}</div>
                    <div class='bar-outer'>
                        <div class='bar-inner' style='width:{final_s}%; background:{bar_color};'></div>
                    </div>
                    <div style='margin-top:14px; display:flex; justify-content:center; gap:16px;'>
                        <span style='font-family:DM Mono,monospace; font-size:0.7rem; color:#444;'>
                            ML <span style='color:#666;'>{ml_s}%</span>
                        </span>
                        <span style='font-family:DM Mono,monospace; font-size:0.7rem; color:#444;'>
                            HEURISTIC <span style='color:#666;'>{h_s}%</span>
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if flags:
                    st.markdown("<div class='section-label'>Detection Reasons</div>", unsafe_allow_html=True)
                    for name, detail in flags:
                        st.markdown(f"""
                        <div class='flag-item'>
                            <div class='flag-name' style='color: {s_color};'>⚑ {name}</div>
                            <div class='flag-detail'>{detail}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='flag-item'>
                        <div class='flag-name' style='color:#4ade80;'>✓ No suspicious patterns detected</div>
                        <div class='flag-detail'>This message does not match known phishing indicators.</div>
                    </div>
                    """, unsafe_allow_html=True)

                if level in ("PHISHING", "SUSPICIOUS"):
                    st.markdown("<div class='section-label'>Safe Alternatives</div>", unsafe_allow_html=True)
                    alts = get_safe_alternatives("", " ".join([n for n, _ in flags]))
                    for alt in alts:
                        st.markdown(f"<div class='alt-item'>→ {alt}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB: URL / LINK
# ─────────────────────────────────────────────
elif selected == "URL / Link":
    st.markdown("<div class='section-label'>Input</div>", unsafe_allow_html=True)
    url_input = st.text_input(
        label="url_input",
        label_visibility="collapsed",
        placeholder="Paste URL or link here…"
    )

    if st.button("SCAN →", key="url_btn"):
        if not url_input.strip():
            st.warning("Please paste a URL first.")
        else:
            with st.spinner(""):
                score, level, flags = analyze_url(url_input.strip())

                if level == "PHISHING":
                    s_color, bg_color, bar_color = "#ef4444", "#1a0808", "#ef4444"
                elif level == "SUSPICIOUS":
                    s_color, bg_color, bar_color = "#f59e0b", "#1a1408", "#f59e0b"
                else:
                    s_color, bg_color, bar_color = "#4ade80", "#0d1a11", "#4ade80"

                st.markdown(f"""
                <div class='score-block' style='background:{bg_color}; border-color:{s_color}22;'>
                    <div style='font-family:DM Mono,monospace; font-size:0.68rem; color:#444; letter-spacing:0.1em; margin-bottom:8px;'>RISK SCORE</div>
                    <div class='score-number' style='color:{s_color};'>{score}<span style='font-size:1.5rem; color:#444;'>%</span></div>
                    <div class='score-label' style='color:{s_color};'>{level}</div>
                    <div class='bar-outer'>
                        <div class='bar-inner' style='width:{score}%; background:{bar_color};'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div class='section-label'>Detection Reasons</div>", unsafe_allow_html=True)
                for name, detail in flags:
                    icon = "✓" if level == "SAFE" else "⚑"
                    name_color = "#4ade80" if level == "SAFE" else s_color
                    st.markdown(f"""
                    <div class='flag-item'>
                        <div class='flag-name' style='color:{name_color};'>{icon} {name}</div>
                        <div class='flag-detail'>{detail}</div>
                    </div>
                    """, unsafe_allow_html=True)

                if level in ("PHISHING", "SUSPICIOUS"):
                    st.markdown("<div class='section-label'>Safe Alternatives</div>", unsafe_allow_html=True)
                    flags_text = " ".join([n + " " + d for n, d in flags])
                    alts = get_safe_alternatives(url_input, flags_text)
                    for alt in alts:
                        st.markdown(f"<div class='alt-item'>→ {alt}</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(f"""
<div style='margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #1e1e1e;
     display:flex; justify-content:space-between; align-items:center;'>
    <span style='font-family:DM Mono,monospace; font-size:0.65rem; color:#333; letter-spacing:0.08em;'>PHISHGUARD v2.0</span>
    <span style='font-family:DM Mono,monospace; font-size:0.65rem; color:#333; letter-spacing:0.06em;'>{system_status}</span>
</div>
""", unsafe_allow_html=True)
