import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import re, os, math, base64, io
from urllib.parse import urlparse, unquote
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
import tldextract

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="PhishGuard AI", page_icon="🛡️", layout="centered")

# ══════════════════════════════════════════════════════════════════════════════
# 1.  KNOWLEDGE BASES
# ══════════════════════════════════════════════════════════════════════════════
WHITELIST = {
    'google','youtube','facebook','github','linkedin','microsoft','apple',
    'amazon','wikipedia','twitter','instagram','netflix','spotify','adobe',
    'dropbox','paypal','reddit','stackoverflow','openai','anthropic','stripe',
    'shopify','slack','zoom','notion','figma','vercel','cloudflare','aws',
    'azure','salesforce','hubspot','mailchimp','twilio','sendgrid','okta',
    'atlassian','jira','gitlab','docker','digitalocean','heroku','mongodb',
    'firebase','chase','wellsfargo','bankofamerica','citibank','hsbc','barclays',
}

BAD_TLDS = {
    'xyz','top','club','online','vip','click','tk','ml','cf','ga','gq',
    'work','rest','date','download','loan','win','bid','racing','review',
    'stream','party','trade','science','accountant','cricket','faith',
    'men','ninja','pw','kim','country','icu','buzz','monster','fyi','fit','cam',
}

BRAND_NAMES = [
    'paypal','amazon','apple','google','microsoft','facebook','netflix',
    'instagram','twitter','bank','chase','wells','citibank','hsbc','barclays',
    'ebay','alibaba','tiktok','whatsapp','telegram','discord','linkedin',
    'dropbox','adobe','spotify','walmart','fedex','ups','dhl','usps','irs',
    'binance','coinbase','metamask','ethereum','bitcoin','blockchain',
]

HOMOGLYPH = {'0':'o','1':'l','3':'e','4':'a','5':'s','6':'b','7':'t','@':'a','$':'s','|':'i'}

# ─── Safety database ──────────────────────────────────────────────────────────
SAFETY_DB = {
    'paypal':    ('paypal.com',          'PayPal official app',            '1-888-221-1161',
                  'PayPal will NEVER email you a link to log in. Always go to paypal.com directly.'),
    'amazon':    ('amazon.com',          'Amazon Shopping app',            '1-888-280-4331',
                  'Amazon never requests gift-card payments or passwords via email.'),
    'apple':     ('apple.com/support',   'Apple Support app',              '1-800-275-2273',
                  'Apple will never call you about an iCloud breach. Hang up immediately.'),
    'google':    ('myaccount.google.com','Google One app',                 'support.google.com',
                  'Google never sends a "click to verify" email with a login link.'),
    'microsoft': ('account.microsoft.com','Microsoft Authenticator app',   '1-800-642-7676',
                  'Microsoft support will never cold-call you about your computer.'),
    'facebook':  ('facebook.com',        'Facebook / Meta app',            'facebook.com/help',
                  'Facebook will never ask for your password over email or Messenger.'),
    'netflix':   ('netflix.com',         'Netflix app',                    '1-888-638-3549',
                  'Netflix never requests gift-card payments. Cancel at netflix.com directly.'),
    'instagram': ('instagram.com',       'Instagram app',                  'help.instagram.com',
                  'If told your account was hacked, go to instagram.com/hacked directly.'),
    'bank':      ('URL on back of your card','Your bank\'s official app',  'Number on your card',
                  'Your bank will NEVER ask for your full PIN or password via SMS or email.'),
    'irs':       ('irs.gov',             'IRS2Go app',                     '1-800-829-1040',
                  'The IRS contacts you by postal mail first — never by phone, text, or email.'),
    'crypto':    ('coinbase.com / binance.com','Official exchange app',    'support.coinbase.com',
                  'No legitimate crypto platform will ever ask for your seed phrase. Ever.'),
    'dhl':       ('dhl.com',             'DHL Express app',                '1-800-225-5345',
                  'DHL never asks for customs fees via SMS link. Check parcels at dhl.com.'),
    'fedex':     ('fedex.com',           'FedEx Mobile app',               '1-800-463-3339',
                  'FedEx delivery-fee SMS/email links are almost always phishing.'),
    'default':   ('Type the official address directly in your browser',
                  'Download the official app from App Store / Google Play',
                  'Find the contact number on the official website',
                  'When in doubt — do NOT click. Search for the official website on Google.'),
}

def get_safety(context: str) -> dict:
    c = context.lower()
    for key, vals in SAFETY_DB.items():
        if key != 'default' and key in c:
            return {'site': vals[0], 'app': vals[1], 'contact': vals[2], 'tip': vals[3]}
    d = SAFETY_DB['default']
    return {'site': d[0], 'app': d[1], 'contact': d[2], 'tip': d[3]}

# ══════════════════════════════════════════════════════════════════════════════
# 2.  ENSEMBLE ML MODEL
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def build_model():
    text_feats = FeatureUnion([
        ('wtf',  TfidfVectorizer(analyzer='word', stop_words='english',
                                 max_features=12000, ngram_range=(1,3), sublinear_tf=True)),
        ('ctf',  TfidfVectorizer(analyzer='char_wb', max_features=8000,
                                 ngram_range=(3,5), sublinear_tf=True)),
        ('bow',  CountVectorizer(analyzer='word', max_features=5000,
                                 ngram_range=(1,2), binary=True)),
    ])
    lr  = LogisticRegression(max_iter=2000, C=2.0, class_weight='balanced')
    cnb = ComplementNB(alpha=0.1)
    svm = CalibratedClassifierCV(LinearSVC(max_iter=3000, C=1.5, class_weight='balanced'))

    pipe = Pipeline([
        ('feat', text_feats),
        ('clf',  VotingClassifier(
            estimators=[('lr', lr), ('cnb', cnb), ('svm', svm)],
            voting='soft', weights=[3, 1, 2]))
    ])

    aug_phish = [
        "Your PayPal account has been limited. Verify now: http://paypal-secure-login.xyz/verify",
        "URGENT: Your bank account will be suspended. Click to verify: http://192.168.1.1/bank",
        "You have won $1,000,000 lottery prize. Send details to claim@lottery-win.tk",
        "Dear customer, unusual activity detected. Login: http://amaz0n-accounts.top",
        "IRS NOTICE: You owe back taxes. Call immediately or face arrest.",
        "Your Apple ID is locked. Verify: http://apple-id.secure-verify.click",
        "Nigerian prince needs help transferring $50 million. Reply urgently.",
        "Netflix subscription expired! Update payment: http://netflix-billing.online",
        "CONGRATULATIONS! Selected for $500 Amazon gift card. Claim now!",
        "Security alert: Someone logged into your Google account. Verify: http://google-security.xyz",
        "Your crypto wallet needs verification. Enter seed phrase at: http://metamask-verify.top",
        "Microsoft account expires today. Reset: http://ms-secure.click",
        "Package held at customs. Pay $2.99 to release: http://dhl-customs-fee.online",
        "Your SSN has been suspended. Call SSA fraud dept now.",
        "BANK ALERT: Card charged $847. Not you? Verify: http://chase-verify.xyz",
        "Verify your identity now or account will be permanently deleted within 24 hours.",
        "Click here to claim your free iPhone 14: http://free-prize.tk/iphone",
        "Your account will be closed unless you re-verify your credentials immediately.",
        "Dear valued customer, update your billing information to avoid service interruption.",
        "FedEx: Your package is on hold. Pay delivery fee: http://fedex-delivery.online",
    ]
    aug_safe = [
        "Hey, are we still on for dinner tonight?",
        "Meeting rescheduled to 3pm, conference room B.",
        "Thanks for your order! It will arrive in 3-5 business days.",
        "Your appointment is confirmed for Tuesday at 10am.",
        "Happy birthday! Hope you have a wonderful day.",
        "The report is ready for review. I've shared it in the drive.",
        "Can you pick up some milk on the way home?",
        "Great work on the presentation today!",
        "Your verification code is 483920. Do not share this with anyone.",
        "Reminder: Team standup in 15 minutes.",
        "The package was delivered to your front door at 2:34 PM.",
        "Your flight is confirmed. Check-in opens 24 hours before departure.",
        "Thanks for subscribing! Your first invoice is attached.",
        "The doctor will see you at 3pm on Thursday.",
    ]

    status = "Lightweight model (no dataset)"
    if os.path.exists('sms_spam_10000_dataset.csv'):
        try:
            df = pd.read_csv('sms_spam_10000_dataset.csv')
            if 'Category' in df.columns and 'Message' in df.columns:
                X_df, y_raw = df['Message'].astype(str), df['Category']
            elif 'v1' in df.columns and 'v2' in df.columns:
                X_df, y_raw = df['v2'].astype(str), df['v1']
            else:
                raise ValueError("Unknown columns")
            y_df = y_raw.apply(lambda v: 1 if str(v).strip().lower()=='spam' else 0)
            X_all = pd.concat([X_df, pd.Series(aug_phish+aug_safe)], ignore_index=True)
            y_all = pd.concat([y_df, pd.Series([1]*len(aug_phish)+[0]*len(aug_safe))], ignore_index=True)
            pipe.fit(X_all, y_all)
            status = f"Trained on {len(X_all):,} messages"
        except Exception as e:
            X_s = pd.Series(aug_phish+aug_safe)
            y_s = pd.Series([1]*len(aug_phish)+[0]*len(aug_safe))
            pipe.fit(X_s, y_s)
    else:
        X_s = pd.Series(aug_phish+aug_safe)
        y_s = pd.Series([1]*len(aug_phish)+[0]*len(aug_safe))
        pipe.fit(X_s, y_s)

    return pipe, status

model, model_status = build_model()

# ══════════════════════════════════════════════════════════════════════════════
# 3.  TEXT / MESSAGE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
TEXT_RULES = [
    (r'\b(urgent|act now|immediately|expires? (today|soon|now)|respond (now|immediately)|asap|don\'t delay)\b',
     4, "Urgency / Pressure Tactic",
     "Phishing messages manufacture a false sense of urgency to make you act before thinking. "
     "'Urgent', 'expires today', and 'act now' are psychological pressure tools."),

    (r'\b(your account (will be|has been|is|was) (suspended|blocked|locked|disabled|terminated|compromised|hacked))\b',
     5, "Account Threat Language",
     "Threatening account suspension is the #1 fear-based phishing tactic. "
     "Real companies send formal notices by post — not panicked SMS/emails demanding instant action."),

    (r'\b(verify|confirm|validate|update|re-?verify) (your )?(account|identity|information|details|password|card|billing)\b',
     4, "Fake Verification Request",
     "Legitimate services NEVER ask you to verify sensitive data via a link in a message. "
     "Always navigate to the official website yourself — never through a link."),

    (r'\b(won|winner|selected|awarded|prize|lottery|jackpot|congratulations.*you|you.*won|sweepstake)\b',
     4, "Prize / Lottery Bait",
     "You cannot win a lottery you never entered. Fake prize notifications are a classic "
     "lure to collect personal details or redirect you to malicious pages."),

    (r'\b(click|tap|open|follow|visit)\s+(this\s+)?(link|url|below|here|button)\b',
     3, "Suspicious Link Call-to-Action",
     "Directing you to click an unverified link is the primary phishing delivery method. "
     "The real destination is often hidden behind a redirect or URL shortener."),

    (r'\b(password|passwd|pin\b|otp|one.?time.?pass|verification code|security code|access code)\b',
     5, "OTP / Password Request",
     "No legitimate service will EVER ask for your password, PIN, or OTP via SMS or email. "
     "This is an absolute red line and always indicates fraud."),

    (r'\b(credit card|debit card|card number|cvv|expiry date|billing info|full card details)\b',
     5, "Card Details Request",
     "Requesting card number, CVV, or expiry via a message is always fraudulent. "
     "Real companies process payments only on secure, verified checkout pages."),

    (r'\b(social security|ssn|national insurance|passport number|driver.?s? licen[sc]e)\b',
     5, "Government ID Request",
     "Requesting government ID numbers via message is a strong indicator of identity theft fraud. "
     "No agency or service legitimately does this."),

    (r'\b(transfer|wire|send|pay).{0,20}(bitcoin|btc|ethereum|eth|crypto|gift card|itunes|google play|steam)\b',
     5, "Crypto / Gift Card Payment Demand",
     "Demanding payment in gift cards or crypto is the #1 scammer payment method — "
     "untraceable and irreversible. No government, utility, or bank ever asks for this."),

    (r'\b(nigerian?|prince|inheritance|diplomat|refugee|widow|dying|terminal).{0,40}(million|transfer|fund|help)\b',
     5, "Advance-Fee Fraud (419 Scam)",
     "Classic advance-fee fraud pattern. The goal is to extract small 'processing fees' "
     "with a promise of large returns that never materialise."),

    (r'\b(dear\s+(customer|user|member|account holder|client|valued|sir|madam|friend))\b',
     2, "Generic Impersonal Greeting",
     "Phishing messages use generic greetings ('Dear Customer') because they are blasted in bulk. "
     "Legitimate services address you by your real name."),

    (r'\b(legal action|arrest|sued|warrant|court order|fined|prosecuted|reported to police|law enforcement)\b',
     4, "Legal Threat / Intimidation",
     "Threatening arrest or legal action is a scare tactic. "
     "Government agencies always initiate contact via official postal mail, not SMS or email."),

    (r'\b(seed phrase|private key|wallet address|connect wallet|approve transaction|metamask)\b',
     5, "Crypto Wallet Credential Theft",
     "Asking for a seed phrase or private key will instantly drain your entire crypto wallet. "
     "No legitimate platform, wallet, or exchange ever needs this — not once, not ever."),

    (r'\b(invoice|receipt|billing statement|payment (due|failed|declined|overdue|required))\b',
     2, "Fake Invoice / Payment Alert",
     "Fake invoice or payment-failed alerts create urgency around non-existent debts "
     "to harvest payment details or redirect to phishing pages."),

    (r'\b(tracking (number|id|code)|your (order|package|parcel|shipment) (is|has|was))\b',
     2, "Fake Shipment / Delivery Notice",
     "Fake parcel delivery notifications are a top phishing lure, especially impersonating "
     "FedEx, DHL, UPS, and USPS. Always check deliveries on the carrier's official site."),

    (r'\b(re-?activate|re-?enable|restore access|recover your account|re-?verify)\b',
     3, "Account Reactivation Scam",
     "Fake reactivation requests drive you to credential-harvesting pages that look "
     "pixel-perfect to real login pages — but steal everything you type."),

    (r'\b(unsubscribe|opt.?out|stop receiving|manage (your )?preferences)\b',
     1, "Mass-Mail Footer",
     "An unsubscribe footer suggests bulk communication — commonly used in phishing emails "
     "to mimic legitimate newsletters and bypass spam filters."),

    (r'\b(download|install|update|run).{0,20}(app|software|tool|attachment|file|exe|apk)\b',
     4, "Malicious Download Prompt",
     "Prompting you to download or install something from a link in a message is a "
     "classic malware delivery technique. Only install software from official stores."),
]

def analyze_message(text: str):
    tl = text.lower()
    hits, tw = [], 0
    for pat, w, name, expl in TEXT_RULES:
        if re.search(pat, tl, re.I):
            hits.append((w, name, expl))
            tw += w
    max_w = sum(r[1] for r in TEXT_RULES)
    h_score = min(int((tw / max(max_w,1)) * 100 * 3.5), 100)

    ml_score = 0
    if model:
        try:
            ml_score = int(model.predict_proba([text])[0][1] * 100)
        except: pass

    final = int(0.55*ml_score + 0.45*h_score) if model else h_score
    crit = sum(1 for h in hits if h[0] == 5)
    final = min(final + 12*crit, 100)

    level = "PHISHING" if final>=65 else ("SUSPICIOUS" if final>=35 else "SAFE")
    return ml_score, h_score, final, level, hits

# ══════════════════════════════════════════════════════════════════════════════
# 4.  URL ANALYSIS  (17 signals)
# ══════════════════════════════════════════════════════════════════════════════
def _entropy(s):
    if not s: return 0.0
    c = Counter(s); n = len(s)
    return -sum((v/n)*math.log2(v/n) for v in c.values())

def _levenshtein(a, b):
    if len(a)<len(b): a,b = b,a
    prev = list(range(len(b)+1))
    for i,ca in enumerate(a):
        cur=[i+1]
        for j,cb in enumerate(b):
            cur.append(min(prev[j]+(ca!=cb), cur[-1]+1, prev[j+1]+1))
        prev=cur
    return prev[-1]

def analyze_url(raw: str):
    url = raw.strip()
    if not url.startswith(('http://','https://')):
        url = 'https://'+url
    ext    = tldextract.extract(url)
    domain = ext.domain.lower()
    suffix = ext.suffix.lower()
    sub    = ext.subdomain.lower()
    parsed = urlparse(url)
    path   = parsed.path.lower()
    query  = parsed.query.lower()
    full   = url.lower()
    try: decoded = unquote(url).lower()
    except: decoded = full

    score, flags = 0, []

    # Whitelist
    clean_sub = sub.replace('www','').replace('.','').strip()
    if domain in WHITELIST and not clean_sub:
        return 0, "SAFE", [(0,"Verified Trusted Domain","On the verified whitelist of major legitimate services.")]

    # 1. IP as host
    if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", parsed.netloc.split(':')[0]):
        score+=90; flags.append((5,"IP Address as Domain",
            f"Uses raw IP ({parsed.netloc}) instead of a domain name. "
            "Legitimate websites never use raw IPs. This is used to hide the server identity."))

    # 2. No HTTPS
    if parsed.scheme=='http':
        score+=20; flags.append((3,"No HTTPS Encryption",
            "Uses HTTP — unencrypted. Every site handling accounts or payments "
            "must use HTTPS. Absence of it is a strong warning sign."))

    # 3. Brand in subdomain
    for b in BRAND_NAMES:
        if b in sub and domain not in WHITELIST:
            score+=70; flags.append((5,f"'{b.title()}' Impersonated in Subdomain",
                f"Subdomain contains '{b}' but real domain is '{domain}.{suffix}'. "
                f"In 'paypal.evil.com' the real site is evil.com — NOT PayPal. "
                "This is the most common phishing technique.")); break

    # 4. Homoglyph substitution
    norm = ''.join(HOMOGLYPH.get(c,c) for c in domain)
    if norm != domain:
        for b in BRAND_NAMES:
            if b in norm and domain not in WHITELIST:
                score+=65; flags.append((5,"Homoglyph / Character Substitution",
                    f"'{domain}' uses lookalike characters (0→o, 1→l, @→a) to impersonate "
                    f"'{norm}'. Designed to fool visual inspection.")); break

    # 5. Typosquatting (Levenshtein)
    best, dist = None, 999
    for b in BRAND_NAMES:
        d = _levenshtein(domain, b)
        if d < dist: best, dist = b, d
    if 0 < dist <= 2 and domain not in WHITELIST and len(domain) > 3:
        score+=55; flags.append((4,f"Typosquatting — Close to '{best.title()}'",
            f"'{domain}' is only {dist} character(s) from '{best}'. "
            "Attackers register near-identical domains (amazzon.com, paypa1.com) to trick users."))

    # 6. Excess subdomains
    sub_parts = [p for p in sub.split('.') if p and p!='www']
    if len(sub_parts) >= 3:
        score+=35; flags.append((3,f"Excessive Subdomains ({len(sub_parts)} levels)",
            f"Subdomain: '{sub}'. 3+ levels of subdomains buries the real domain at the end, "
            "making fake URLs look official."))

    # 7. @ symbol
    if '@' in url:
        score+=90; flags.append((5,"'@' Symbol in URL",
            "Browsers ignore everything before '@'. So 'paypal.com@evil.com' actually loads evil.com. "
            "This is a direct browser credential-hijack exploit."))

    # 8. Bad TLD
    if suffix in BAD_TLDS:
        score+=45; flags.append((3,f"High-Risk TLD (.{suffix})",
            f"'.{suffix}' is widely used for free/throwaway domains and appears "
            "disproportionately in phishing, malware, and spam due to zero/low registration cost."))

    # 9. Hyphens
    h = domain.count('-')
    if h >= 2:
        score+=min(15+h*8,40); flags.append((2,f"Excessive Hyphens in Domain ({h})",
            f"'{domain}' has {h} hyphens. Phishing domains use hyphens to appear official "
            "(e.g. paypal-secure-verify-account.com)."))

    # 10. Long URL
    if len(url) > 85:
        score+=18; flags.append((2,f"Very Long URL ({len(url)} chars)",
            "Long URLs bury the real domain in walls of confusing parameters "
            "to prevent the recipient from seeing where the link actually goes."))

    # 11. Sensitive keywords
    kws = ['login','verify','account','update','secure','confirm','password',
           'banking','signin','credential','validate','wallet','payment','recover',
           'authenticate','billing','otp','pin','reset','access']
    found = [k for k in kws if k in path or k in query]
    if found:
        score+=25; flags.append((3,f"Sensitive Keywords in Path: {', '.join(found[:4])}",
            "Legitimate sites rarely place action words like 'login', 'verify', or 'password' "
            "directly in URL paths. Phishing pages do this to mimic real login/verification flows."))

    # 12. Digit ratio
    dr = sum(c.isdigit() for c in domain) / max(len(domain),1)
    if dr > 0.35:
        score+=30; flags.append((3,f"High Digit Ratio in Domain ({int(dr*100)}%)",
            f"'{domain}' is {int(dr*100)}% digits. Auto-generated phishing domains "
            "use random numbers to avoid detection and blocklists."))

    # 13. Open redirect
    if any(p in full for p in ['redirect=','url=','goto=','next=','return=','dest=','forward=']):
        score+=35; flags.append((4,"Open Redirect Parameter",
            "Contains a redirect parameter (?url=, ?goto=). This lets attackers chain URLs — "
            "the first link looks safe but silently forwards to a malicious page."))

    # 14. URL shortener
    shorteners = ['bit.ly','tinyurl','t.co','goo.gl','ow.ly','buff.ly',
                  'short.link','rb.gy','cutt.ly','tiny.cc','is.gd','v.gd']
    if any(s in full for s in shorteners):
        score+=30; flags.append((3,"URL Shortener Detected",
            "Shorteners hide the final destination. Phishing links are frequently disguised "
            "with bit.ly, tinyurl etc. to bypass link scanners."))

    # 15. Domain entropy (DGA)
    ent = _entropy(domain)
    if ent > 3.7 and len(domain) > 8:
        score+=25; flags.append((3,f"High Domain Entropy ({ent:.2f}) — Possible DGA",
            f"Entropy {ent:.2f} suggests algorithmically generated domain (Domain Generation Algorithm), "
            "common in phishing, botnet C2, and malware infrastructure."))

    # 16. Punycode / IDN
    if 'xn--' in domain or any(ord(c)>127 for c in raw):
        score+=50; flags.append((4,"Punycode / Unicode Domain (IDN Homograph Attack)",
            "Uses Unicode or punycode (xn--). Attackers register domains with Cyrillic/Greek "
            "characters that look identical to Latin letters — visually indistinguishable fakes."))

    # 17. URL encoding obfuscation
    if re.search(r'%[0-9a-fA-F]{2}', path):
        score+=20; flags.append((2,"URL Encoding in Path",
            "Percent-encoded characters (%2F, %40) in the path can hide malicious routes "
            "and evade content filters that scan for keywords."))

    score = min(score, 100)
    level = "PHISHING" if score>=60 else ("SUSPICIOUS" if score>=30 else "SAFE")
    if not flags:
        flags.append((0,"Clean URL Structure","No suspicious patterns detected. Appears structurally safe."))
    return score, level, flags

# ══════════════════════════════════════════════════════════════════════════════
# 5.  IMAGE PHISHING ANALYSIS
#     Works purely on heuristic extraction — no external API needed.
#     Detects: embedded URLs, suspicious text patterns, QR-code-like data URIs,
#     and image metadata anomalies.
# ══════════════════════════════════════════════════════════════════════════════
def analyze_image(uploaded_file) -> tuple:
    """
    Returns (score 0-100, level, list-of-(severity, name, explanation), extracted_urls).
    Pure heuristic — reads the raw bytes to extract embedded text/URLs.
    """
    raw = uploaded_file.read()
    uploaded_file.seek(0)
    score, flags, found_urls = 0, [], []

    # Decode bytes to string (ignore non-UTF chars) for pattern matching
    try:
        text_repr = raw.decode('latin-1', errors='replace')
    except:
        text_repr = str(raw)

    # ── a) Embedded URLs in image data ────────────────────────────────────────
    url_pat = re.compile(r'https?://[^\s\x00-\x1f\'"<>]{5,}', re.I)
    raw_urls = url_pat.findall(text_repr)
    # Clean and deduplicate
    for u in raw_urls:
        u_clean = re.sub(r'[^\x20-\x7e]', '', u)[:200]
        if u_clean and u_clean not in found_urls:
            found_urls.append(u_clean)

    if found_urls:
        score += 40
        flags.append((4, f"Embedded URLs Found ({len(found_urls)})",
            f"Image contains {len(found_urls)} embedded URL(s). Phishing images embed links "
            "in their binary data or metadata to bypass text-based spam filters. "
            f"First URL: {found_urls[0][:80]}"))

    # ── b) Suspicious embedded text patterns ──────────────────────────────────
    text_lower = text_repr.lower()
    phish_kw = {
        'verify': "Contains 'verify' keyword — common in phishing instructions.",
        'account': "Contains 'account' keyword — phishing images often ask to log in.",
        'password': "Contains 'password' keyword — embedded credential requests in images.",
        'click here': "Contains 'click here' — a call-to-action embedded in the image.",
        'urgent': "Contains 'urgent' — pressure tactic embedded in image text.",
        'login': "Contains 'login' — credential phishing instruction in image.",
        'suspended': "Contains 'suspended' — account-threat language in image.",
        'confirm': "Contains 'confirm' — verification request embedded in image.",
        'prize': "Contains 'prize' — prize bait embedded in image.",
        'winner': "Contains 'winner' — lottery/prize bait in image.",
        'gift card': "Contains 'gift card' — payment scam language in image.",
        'bitcoin': "Contains 'bitcoin' — crypto fraud language in image.",
        'seed phrase': "Contains 'seed phrase' — crypto wallet theft attempt in image.",
    }
    hit_kw = []
    for kw, reason in phish_kw.items():
        if kw in text_lower:
            hit_kw.append((kw, reason))
    if hit_kw:
        score += min(len(hit_kw) * 15, 45)
        flags.append((3, f"Suspicious Keywords Embedded in Image ({len(hit_kw)})",
            "Found phishing-related keywords in the image file: " +
            ", ".join(f"'{k}'" for k,_ in hit_kw[:5]) +
            ". Attackers embed text directly in images to hide it from text-based filters."))

    # ── c) QR-code-like high-entropy data blocks ──────────────────────────────
    # Large base64-like blocks can contain hidden payloads
    b64_blocks = re.findall(r'[A-Za-z0-9+/]{60,}={0,2}', text_repr)
    if len(b64_blocks) > 5:
        score += 20
        flags.append((2, f"Multiple Encoded Data Blocks ({len(b64_blocks)})",
            f"Image contains {len(b64_blocks)} large encoded data blocks. "
            "These can hide payloads, embedded scripts, or additional URLs "
            "that bypass visual inspection."))

    # ── d) Extremely small image (1x1 tracking pixel) ─────────────────────────
    if len(raw) < 500:
        score += 50
        flags.append((4, "Micro Image — Possible Tracking Pixel",
            f"File is only {len(raw)} bytes. 1×1 pixel tracking images are used "
            "to confirm email delivery, track when emails are opened, and map your IP address "
            "without your knowledge."))

    # ── e) File size anomaly (huge image with little visual content) ──────────
    if len(raw) > 5_000_000:
        score += 15
        flags.append((2, f"Unusually Large Image File ({len(raw)//1024//1024} MB)",
            "Extremely large images can contain hidden zip archives, executables, "
            "or steganographic payloads embedded inside the image data."))

    # ── f) JPEG/PNG magic byte mismatch (polyglot file) ──────────────────────
    ext_name = uploaded_file.name.lower().split('.')[-1] if '.' in uploaded_file.name else ''
    is_jpeg = raw[:2] == b'\xff\xd8'
    is_png  = raw[:8] == b'\x89PNG\r\n\x1a\n'
    is_gif  = raw[:6] in (b'GIF87a', b'GIF89a')
    claimed_img = ext_name in ('jpg','jpeg','png','gif','bmp','webp')
    actual_img  = is_jpeg or is_png or is_gif

    if claimed_img and not actual_img:
        score += 60
        flags.append((5, "File Format Mismatch — Possible Disguised Executable",
            f"File claims to be '{ext_name}' but does not start with a valid image signature. "
            "This is a classic technique to disguise malware as an image "
            "(e.g. malware.exe renamed to photo.jpg)."))

    # ── g) Scan any found URLs through URL engine ─────────────────────────────
    url_flags_summary = []
    for u in found_urls[:3]:
        u_score, u_level, u_flags = analyze_url(u)
        if u_level in ('PHISHING', 'SUSPICIOUS'):
            score += min(u_score // 3, 20)
            url_flags_summary.append(f"{u[:60]}… → {u_level} ({u_score}%)")

    if url_flags_summary:
        flags.append((5, "Embedded URLs Are Malicious",
            "URLs found inside this image were scanned and flagged as phishing or suspicious: "
            + " | ".join(url_flags_summary)))

    score = min(score, 100)
    level = "PHISHING" if score>=60 else ("SUSPICIOUS" if score>=30 else "SAFE")
    if not flags:
        flags.append((0,"No Threats Detected in Image",
            "No embedded URLs, suspicious keywords, tracking pixels, or format anomalies found."))
    return score, level, flags, found_urls

# ══════════════════════════════════════════════════════════════════════════════
# 6.  CSS — Clean White Minimal
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background-color: #ffffff !important;
    color: #111111 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 1.8rem 5rem !important; max-width: 760px !important; }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #fafafa !important;
    border: 1.5px solid #e5e5e5 !important;
    border-radius: 8px !important;
    color: #111 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    padding: 13px 14px !important;
    transition: border-color 0.18s;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #111 !important;
    box-shadow: none !important;
    outline: none !important;
    background: #fff !important;
}

/* ── Button ── */
div.stButton > button {
    background: #111 !important;
    color: #fff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 0 !important;
    width: 100% !important;
    transition: background 0.15s, transform 0.1s !important;
}
div.stButton > button:hover  { background: #333 !important; transform: translateY(-1px) !important; }
div.stButton > button:active { transform: translateY(0) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 1.5px dashed #d0d0d0 !important;
    border-radius: 8px !important;
    background: #fafafa !important;
    padding: 8px !important;
}
[data-testid="stFileUploader"]:hover { border-color: #aaa !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #111 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #f5f5f5; }
::-webkit-scrollbar-thumb { background: #ccc; border-radius: 2px; }

/* ── Alerts ── */
.stAlert { border-radius: 8px !important; font-size: 0.83rem !important; font-family: 'Inter',sans-serif !important; }

/* ── Option menu override ── */
nav[id="option-menu"] { font-family: 'Inter', sans-serif !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 7.  UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def c_score(score, level):
    return {
        "PHISHING":   ("#dc2626","#fef2f2","rgba(220,38,38,0.15)"),
        "SUSPICIOUS": ("#d97706","#fffbeb","rgba(217,119,6,0.12)"),
        "SAFE":       ("#16a34a","#f0fdf4","rgba(22,163,74,0.12)"),
    }.get(level, ("#555","#f9f9f9","rgba(0,0,0,0.05)"))

SEV_COLOR = {0:"#16a34a",1:"#6b7280",2:"#d97706",3:"#ea580c",4:"#dc2626",5:"#7f1d1d"}

def label(text):
    st.markdown(f"""
    <div style="font-family:'Inter',sans-serif; font-size:0.68rem; font-weight:600;
        letter-spacing:0.12em; text-transform:uppercase; color:#9ca3af;
        margin:22px 0 8px; padding-bottom:6px; border-bottom:1px solid #f0f0f0;">
        {text}
    </div>""", unsafe_allow_html=True)

def score_block(score, level, sub=""):
    tc, bg, br = c_score(score, level)
    icons = {"PHISHING":"🚨","SUSPICIOUS":"⚠️","SAFE":"✅"}
    st.markdown(f"""
    <div style="border:1.5px solid {br}; border-radius:12px; padding:28px 24px 22px;
        background:{bg}; margin:16px 0 22px; text-align:center;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:4.5rem;
            font-weight:400; letter-spacing:-0.04em; color:{tc}; line-height:1;">
            {score}<span style="font-size:1.6rem; color:#ccc;">%</span>
        </div>
        <div style="font-family:'Inter',sans-serif; font-size:0.8rem; font-weight:700;
            letter-spacing:0.12em; text-transform:uppercase; color:{tc}; margin:6px 0 12px;">
            {icons.get(level,'')} {level}
        </div>
        <div style="height:3px; border-radius:2px; margin-bottom:10px;
            background:linear-gradient(90deg,{tc} {score}%,#e5e7eb {score}%);"></div>
        {f'<div style="font-family:JetBrains Mono,monospace;font-size:0.67rem;color:#9ca3af;">{sub}</div>' if sub else ''}
    </div>""", unsafe_allow_html=True)

def flag_card(sev, name, expl, safe=False):
    color = "#16a34a" if safe else SEV_COLOR.get(sev,"#555")
    icon  = "✓" if safe else ("◉" if sev>=4 else "◎")
    st.markdown(f"""
    <div style="border:1px solid #f0f0f0; border-left:3px solid {color};
        border-radius:0 8px 8px 0; padding:13px 16px; margin-bottom:9px; background:#fafafa;">
        <div style="font-family:'Inter',sans-serif; font-size:0.78rem; font-weight:600;
            color:{color}; margin-bottom:5px;">{icon} {name}</div>
        <div style="font-family:'Inter',sans-serif; font-size:0.82rem;
            color:#6b7280; line-height:1.55;">{expl}</div>
    </div>""", unsafe_allow_html=True)

def safety_card(rec, level):
    if level=="SAFE": return
    st.markdown(f"""
    <div style="border:1.5px solid #bbf7d0; border-radius:12px; padding:20px;
        background:#f0fdf4; margin-top:4px;">
        <div style="font-family:'Inter',sans-serif; font-size:0.68rem; font-weight:700;
            letter-spacing:0.12em; text-transform:uppercase; color:#16a34a; margin-bottom:14px;">
            ✦ What to do instead
        </div>
        <div style="display:grid; gap:10px;">
            <div style="display:flex; gap:14px; align-items:flex-start;">
                <span style="font-family:JetBrains Mono,monospace; font-size:0.67rem; font-weight:500;
                    color:#9ca3af; min-width:68px; padding-top:2px;">WEBSITE</span>
                <span style="font-size:0.84rem; color:#111; font-weight:500;">{rec['site']}</span>
            </div>
            <div style="display:flex; gap:14px; align-items:flex-start;">
                <span style="font-family:JetBrains Mono,monospace; font-size:0.67rem; font-weight:500;
                    color:#9ca3af; min-width:68px; padding-top:2px;">APP</span>
                <span style="font-size:0.84rem; color:#111;">{rec['app']}</span>
            </div>
            <div style="display:flex; gap:14px; align-items:flex-start;">
                <span style="font-family:JetBrains Mono,monospace; font-size:0.67rem; font-weight:500;
                    color:#9ca3af; min-width:68px; padding-top:2px;">CONTACT</span>
                <span style="font-size:0.84rem; color:#111;">{rec['contact']}</span>
            </div>
            <div style="margin-top:6px; padding:10px 14px; background:#dcfce7;
                border-radius:6px; border-left:3px solid #16a34a;">
                <span style="font-family:JetBrains Mono,monospace; font-size:0.67rem;
                    font-weight:600; color:#15803d;">TIP  </span>
                <span style="font-size:0.82rem; color:#166534; line-height:1.5;">{rec['tip']}</span>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

def general_tips():
    tips = [
        ("🔗  Never click links in unexpected messages",
         "If a message creates urgency about your account, go directly to the website by typing it yourself. Never use links from SMS or emails."),
        ("🔍  Inspect the full URL before clicking",
         "Hover over a link and check the very last domain before the first slash — that's the real destination. Everything before it can be faked."),
        ("🔐  Enable two-factor authentication (2FA)",
         "Even if your password is stolen, 2FA blocks access. Use an authenticator app (Google Authenticator, Authy) — not SMS-based 2FA."),
        ("📞  Call official numbers — not the one they give you",
         "Your bank, IRS, Apple, and Amazon all have official numbers on their websites. Call those — never the number in the suspicious message."),
        ("🎁  Gift cards and crypto = scam, always",
         "No government agency, utility, police department, or legitimate business ever asks for payment via gift cards or cryptocurrency."),
        ("📧  Report phishing",
         "Forward phishing SMS to 7726 (SPAM). Report phishing emails to reportphishing@apwg.org or forward to your email provider as junk."),
        ("🧠  When in doubt — don't",
         "If something feels off, pause. Real emergencies give you time to verify. If you can't verify the sender independently, don't act."),
    ]
    st.markdown("""
    <div style="border:1.5px solid #f0f0f0; border-radius:12px; padding:20px; background:#fafafa; margin-top:4px;">
        <div style="font-family:'Inter',sans-serif; font-size:0.68rem; font-weight:700;
            letter-spacing:0.12em; text-transform:uppercase; color:#9ca3af; margin-bottom:14px;">
            How to Protect Yourself — General Rules
        </div>
    """, unsafe_allow_html=True)
    for title, detail in tips:
        st.markdown(f"""
        <div style="padding:10px 0; border-bottom:1px solid #f0f0f0;">
            <div style="font-size:0.83rem; font-weight:600; color:#374151; margin-bottom:3px;">{title}</div>
            <div style="font-size:0.8rem; color:#9ca3af; line-height:1.55;">{detail}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def image_phishing_guide():
    """Educational section: how to spot phishing images."""
    tips = [
        ("Blurry or low-res logos",
         "Phishing images often use stretched, pixelated, or slightly off-colour brand logos. Compare with the real brand's official assets."),
        ("QR codes in images",
         "Attackers embed QR codes in images knowing that QR scanners bypass email URL filters. Always inspect QR destinations before visiting."),
        ("Urgency text overlaid on images",
         "Text like 'Your account is suspended — scan now' overlaid on a fake brand logo is a hallmark of image-based phishing."),
        ("Tracking pixels (1×1 invisible images)",
         "These tiny images silently confirm your email is live and log your IP and device — allowing attackers to refine targeted attacks."),
        ("Attachments that look like images but aren't",
         "Files with .jpg or .png extensions can actually be executables or ZIP files. Always scan attachments before opening."),
    ]
    st.markdown("""
    <div style="border:1.5px solid #fef9c3; border-radius:12px; padding:20px;
        background:#fefce8; margin-top:4px;">
        <div style="font-family:'Inter',sans-serif; font-size:0.68rem; font-weight:700;
            letter-spacing:0.12em; text-transform:uppercase; color:#a16207; margin-bottom:14px;">
            How to Spot Phishing Images
        </div>
    """, unsafe_allow_html=True)
    for t, d in tips:
        st.markdown(f"""
        <div style="padding:8px 0; border-bottom:1px solid #fef08a;">
            <div style="font-size:0.82rem; font-weight:600; color:#713f12; margin-bottom:2px;">⚑ {t}</div>
            <div style="font-size:0.79rem; color:#92400e; line-height:1.5;">{d}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 8.  APP LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="padding:0.5rem 0 2rem; border-bottom:1.5px solid #f0f0f0; margin-bottom:2rem;">
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:5px;">
        <span style="font-family:'Inter',sans-serif; font-size:1.4rem;
            font-weight:700; color:#111; letter-spacing:-0.03em;">🛡️ PhishGuard AI</span>
        <span style="font-family:'JetBrains Mono',monospace; font-size:0.62rem; color:#9ca3af;
            padding:2px 10px; border:1px solid #e5e7eb; border-radius:20px; background:#f9fafb;">v3.1</span>
    </div>
    <p style="font-family:'Inter',sans-serif; font-size:0.8rem; color:#9ca3af; font-weight:400;">
        Phishing detection for URLs · Messages · Emails · Images
    </p>
</div>
""", unsafe_allow_html=True)

selected = option_menu(
    menu_title=None,
    options=["Message / Email", "URL / Link", "Image Scan"],
    icons=["envelope", "link-45deg", "image"],
    orientation="horizontal",
    styles={
        "container": {"background-color":"#f9fafb","border":"1.5px solid #f0f0f0",
                      "border-radius":"10px","padding":"3px"},
        "nav-link": {"font-family":"'Inter',sans-serif","font-size":"0.78rem","font-weight":"500",
                     "color":"#9ca3af","padding":"9px 18px","border-radius":"7px"},
        "nav-link-selected": {"background-color":"#111","color":"#fff","font-weight":"600"},
        "icon": {"display":"none"},
    }
)

# ─── TAB 1: Message / Email ───────────────────────────────────────────────────
if selected == "Message / Email":
    label("Paste message, email body, or SMS text")
    msg = st.text_area("msg", label_visibility="collapsed",
        placeholder="Paste the full message here…\n\nExample: 'URGENT: Your PayPal account is limited. Verify immediately at http://paypal-secure.xyz/verify'",
        height=170)

    if st.button("ANALYZE MESSAGE →", key="mb"):
        if not msg.strip():
            st.warning("Please paste a message to analyze.")
        else:
            with st.spinner("Running ensemble analysis…"):
                ml_s, h_s, final_s, level, hits = analyze_message(msg)
            score_block(final_s, level, f"ML model: {ml_s}%  ·  Heuristic: {h_s}%  ·  Final: {final_s}%")

            label(f"Detection reasons — {len(hits)} signal{'s' if len(hits)!=1 else ''} found")
            if hits:
                for w, name, expl in sorted(hits, key=lambda x:-x[0]):
                    flag_card(w, name, expl)
            else:
                flag_card(0,"No Suspicious Patterns Detected",
                    "This message does not match any known phishing indicators.", safe=True)

            if level in ("PHISHING","SUSPICIOUS"):
                label("What to do instead")
                ctx = " ".join(n for _,n,_ in hits)
                safety_card(get_safety(ctx), level)

            label("How to protect yourself")
            general_tips()

# ─── TAB 2: URL / Link ───────────────────────────────────────────────────────
elif selected == "URL / Link":
    label("Paste URL or link")
    url_in = st.text_input("url", label_visibility="collapsed",
        placeholder="https://example.com/login?verify=account")

    if st.button("SCAN URL →", key="ub"):
        if not url_in.strip():
            st.warning("Please paste a URL to scan.")
        else:
            with st.spinner("Running deep URL forensics…"):
                score, level, flags = analyze_url(url_in.strip())
            score_block(score, level)

            label(f"Detection reasons — {len(flags)} signal{'s' if len(flags)!=1 else ''} found")
            for sev, name, expl in sorted(flags, key=lambda x:-x[0]):
                flag_card(sev, name, expl, safe=(level=="SAFE" or sev==0))

            if level in ("PHISHING","SUSPICIOUS"):
                label("What to do instead")
                ctx = url_in + " " + " ".join(n for _,n,_ in flags)
                safety_card(get_safety(ctx), level)

            label("How to protect yourself")
            general_tips()

# ─── TAB 3: Image Scan ────────────────────────────────────────────────────────
elif selected == "Image Scan":
    label("Upload image to scan (JPG, PNG, GIF, WEBP)")

    st.markdown("""
    <div style="font-family:'Inter',sans-serif; font-size:0.82rem; color:#6b7280;
        background:#fafafa; border:1px solid #f0f0f0; border-radius:8px;
        padding:12px 16px; margin-bottom:16px; line-height:1.6;">
        <strong style="color:#374151;">What this scans for:</strong>
        embedded URLs · suspicious keyword injection · tracking pixels (1×1 spyware) ·
        file format mismatch (malware disguised as image) · encoded data blocks · QR-like payloads
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("img", label_visibility="collapsed",
        type=["jpg","jpeg","png","gif","webp","bmp"])

    if uploaded:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(uploaded, use_column_width=True, caption=uploaded.name)
        with col2:
            st.markdown(f"""
            <div style="font-size:0.8rem; color:#6b7280; line-height:1.9; padding:6px 0;">
                <b style="color:#374151;">File:</b> {uploaded.name}<br>
                <b style="color:#374151;">Size:</b> {len(uploaded.getvalue()):,} bytes ({len(uploaded.getvalue())/1024:.1f} KB)<br>
                <b style="color:#374151;">Type:</b> {uploaded.type or 'unknown'}
            </div>""", unsafe_allow_html=True)

        if st.button("SCAN IMAGE →", key="imgb"):
            with st.spinner("Analysing image for embedded threats…"):
                img_score, img_level, img_flags, found_urls = analyze_image(uploaded)

            score_block(img_score, img_level)

            label(f"Detection reasons — {len(img_flags)} finding{'s' if len(img_flags)!=1 else ''}")
            for sev, name, expl in sorted(img_flags, key=lambda x:-x[0]):
                flag_card(sev, name, expl, safe=(img_level=="SAFE" or sev==0))

            if found_urls:
                label(f"Embedded URLs found ({len(found_urls)}) — scanning each")
                for u in found_urls[:5]:
                    u_score, u_level, u_flags = analyze_url(u)
                    tc,_,_ = c_score(u_score, u_level)
                    st.markdown(f"""
                    <div style="border:1px solid #f0f0f0; border-radius:8px; padding:12px 14px;
                        margin-bottom:8px; background:#fafafa;">
                        <div style="font-family:'JetBrains Mono',monospace; font-size:0.73rem;
                            color:#374151; word-break:break-all; margin-bottom:6px;">{u[:100]}</div>
                        <div style="display:flex; gap:8px; align-items:center;">
                            <span style="font-size:1rem; font-weight:700; color:{tc};">{u_score}%</span>
                            <span style="font-size:0.72rem; font-weight:700; letter-spacing:0.1em;
                                text-transform:uppercase; color:{tc};">{u_level}</span>
                            <span style="font-size:0.75rem; color:#9ca3af;">
                                — {u_flags[0][1] if u_flags else 'No issues'}
                            </span>
                        </div>
                    </div>""", unsafe_allow_html=True)

            if img_level in ("PHISHING","SUSPICIOUS"):
                label("What to do instead")
                safety_card(get_safety(" ".join(n for _,n,_ in img_flags)), img_level)

            label("How to spot phishing images")
            image_phishing_guide()

    else:
        label("How to spot phishing images")
        image_phishing_guide()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:4rem; padding-top:1.2rem; border-top:1.5px solid #f0f0f0;
    display:flex; justify-content:space-between; flex-wrap:wrap; gap:6px;">
    <span style="font-family:'JetBrains Mono',monospace; font-size:0.62rem; color:#d1d5db;">
        PHISHGUARD AI  v3.1
    </span>
    <span style="font-family:'JetBrains Mono',monospace; font-size:0.62rem; color:#d1d5db;">
        {model_status}
    </span>
    <span style="font-family:'JetBrains Mono',monospace; font-size:0.62rem; color:#d1d5db;">
        17 URL signals · 18 text rules · ensemble ML · image scanner
    </span>
</div>
""", unsafe_allow_html=True)
