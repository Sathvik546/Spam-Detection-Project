import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import re
import tldextract
import os
import math
import hashlib
import unicodedata
from urllib.parse import urlparse, unquote, parse_qs
from collections import Counter

# ── sklearn ──────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="PhishGuard", page_icon="🛡️", layout="centered")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 ── KNOWLEDGE BASES
# ══════════════════════════════════════════════════════════════════════════════

WHITELIST = {
    'google','youtube','facebook','github','linkedin','microsoft','apple',
    'amazon','wikipedia','twitter','instagram','netflix','spotify','adobe',
    'dropbox','paypal','reddit','stackoverflow','openai','anthropic','stripe',
    'shopify','slack','zoom','notion','figma','vercel','cloudflare','aws',
    'azure','salesforce','hubspot','mailchimp','twilio','sendgrid','okta',
    'atlassian','jira','confluence','bitbucket','gitlab','docker','kubernetes',
    'digitalocean','heroku','mongodb','firebase','supabase','neon','planetscale',
    'chase','wellsfargo','bankofamerica','citibank','hsbc','barclays',
    'irs','gov','edu','mil'
}

BAD_TLDS = {
    'xyz','top','club','online','vip','click','tk','ml','cf','ga','gq',
    'work','rest','date','download','loan','win','bid','racing','review',
    'stream','party','trade','science','accountant','cricket','faith',
    'men','ninja','pw','kim','country','icu','buzz','live','monster',
    'hair','beauty','fyi','fit','cam','porn','adult'
}

GOOD_TLDS = {'com','org','net','edu','gov','mil','io','co','uk','de','fr','jp','ca','au'}

BRAND_NAMES = [
    'paypal','amazon','apple','google','microsoft','facebook','netflix',
    'instagram','twitter','bank','chase','wells','citibank','hsbc',
    'barclays','ebay','alibaba','tiktok','whatsapp','telegram','discord',
    'linkedin','dropbox','adobe','spotify','walmart','target','bestbuy',
    'fedex','ups','dhl','usps','irs','dmv','medicare','socialsecurity',
    'binance','coinbase','metamask','ethereum','bitcoin','blockchain'
]

HOMOGLYPH_MAP = {
    '0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's', '6': 'b',
    '7': 't', '8': 'b', '@': 'a', '$': 's', '|': 'i', '!': 'i'
}

# Safety alternatives database
SAFETY_DB = {
    'paypal':     {'site':'paypal.com',        'app':'PayPal (App Store / Google Play)',   'contact':'1-888-221-1161',   'tip':'Only ever log in at paypal.com directly — never from a link in email.'},
    'amazon':     {'site':'amazon.com',         'app':'Amazon Shopping App',               'contact':'1-888-280-4331',   'tip':'Amazon never asks for gift card payments or account passwords via email.'},
    'apple':      {'site':'apple.com/support',  'app':'Apple Support App',                 'contact':'1-800-275-2273',   'tip':'Apple will never call you about iCloud being hacked. Hang up.'},
    'google':     {'site':'myaccount.google.com','app':'Google One App',                   'contact':'support.google.com','tip':'Sign in at accounts.google.com — Google will never email you a login link.'},
    'microsoft':  {'site':'account.microsoft.com','app':'Microsoft Authenticator',         'contact':'1-800-642-7676',   'tip':'Microsoft support will never call you unsolicited about your computer.'},
    'facebook':   {'site':'facebook.com',        'app':'Facebook / Meta App',              'contact':'facebook.com/help','tip':'Facebook will never ask for your password over email or Messenger.'},
    'netflix':    {'site':'netflix.com',         'app':'Netflix App',                      'contact':'1-888-638-3549',   'tip':'Netflix will never ask for payment via gift cards.'},
    'instagram':  {'site':'instagram.com',       'app':'Instagram App',                    'contact':'help.instagram.com','tip':'If told your account was hacked, go directly to Instagram.com/hacked.'},
    'bank':       {'site':'Use the URL on the back of your card','app':'Your bank\'s official app','contact':'Number on your card/statement','tip':'Your bank will NEVER ask for your full PIN or password over email or SMS.'},
    'chase':      {'site':'chase.com',           'app':'Chase Mobile App',                 'contact':'1-800-935-9935',   'tip':'Chase will never send a link in an SMS asking you to verify your card.'},
    'irs':        {'site':'irs.gov',             'app':'IRS2Go App',                       'contact':'1-800-829-1040',   'tip':'The IRS only contacts you by postal mail first — never by phone, text or email.'},
    'crypto':     {'site':'coinbase.com or binance.com','app':'Coinbase / Binance official app','contact':'support.coinbase.com','tip':'No legitimate crypto platform will ever ask for your seed phrase. Ever.'},
    'login':      {'site':'Type the official URL directly in your browser','app':'Download the official app','contact':'Use the number on the official website','tip':'Never click login links from emails — type the URL yourself.'},
    'default':    {'site':'Type the official address directly in your browser','app':'Download the official app from App Store / Google Play','contact':'Find the contact on the official website','tip':'When in doubt, do NOT click — search for the official website on Google.'}
}

def get_safety_card(url_str: str, flags_text: str) -> dict:
    combined = (url_str + " " + flags_text).lower()
    for key, rec in SAFETY_DB.items():
        if key != 'default' and key in combined:
            return rec
    return SAFETY_DB['default']

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 ── FEATURE ENGINEERING (Custom sklearn transformers)
# ══════════════════════════════════════════════════════════════════════════════

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts 40+ hand-crafted numeric features from raw text."""

    URGENT_WORDS = re.compile(
        r'\b(urgent|immediately|act now|expires?|deadline|last chance|limited time|within \d+ hours?|'
        r'asap|right now|today only|do not ignore|respond now|don\'t delay|'
        r'your account (will be|has been)|suspended|blocked|locked|restricted|compromised|'
        r'verify now|confirm now|click (below|here|now)|call (now|immediately))\b', re.I)

    MONEY_WORDS = re.compile(
        r'(\$[\d,]+|\d+ dollars?|free money|won|winner|lottery|prize|jackpot|'
        r'inheritance|million|billion|unclaimed|reward|cash|voucher|gift card|coupon)', re.I)

    CREDENTIAL_WORDS = re.compile(
        r'\b(password|passwd|passw0rd|p@ssword|pin|otp|ssn|social security|'
        r'credit card|debit card|card number|cvv|expiry|bank account|routing|'
        r'username|user.?id|login|sign.?in|authenticate|verification code|'
        r'access code|security code|account number|billing)\b', re.I)

    IMPERSONATE_WORDS = re.compile(
        r'\b(amazon|paypal|apple|google|microsoft|facebook|instagram|netflix|'
        r'bank|chase|wells fargo|citibank|irs|fbi|police|government|'
        r'your (bank|provider|carrier|service)|customer service|support team|'
        r'technical support|it department|security team|fraud department)\b', re.I)

    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+', re.I)
    PHONE_PATTERN = re.compile(r'(\+?\d[\d\s\-\(\)]{7,}\d)')
    ALL_CAPS_PATTERN = re.compile(r'\b[A-Z]{4,}\b')

    def _entropy(self, text: str) -> float:
        if not text: return 0.0
        counts = Counter(text)
        total = len(text)
        return -sum((c/total)*math.log2(c/total) for c in counts.values())

    def _ratio_non_alpha(self, text: str) -> float:
        if not text: return 0.0
        return sum(1 for c in text if not c.isalpha()) / len(text)

    def transform(self, texts, y=None):
        out = []
        for text in texts:
            t = str(text)
            tl = t.lower()
            words = tl.split()
            sentences = re.split(r'[.!?]', t)
            urls = self.URL_PATTERN.findall(t)

            feats = [
                # Urgency
                len(self.URGENT_WORDS.findall(tl)),
                # Money
                len(self.MONEY_WORDS.findall(tl)),
                # Credentials
                len(self.CREDENTIAL_WORDS.findall(tl)),
                # Impersonation
                len(self.IMPERSONATE_WORDS.findall(tl)),
                # URL count
                len(urls),
                # Phone numbers
                len(self.PHONE_PATTERN.findall(t)),
                # ALL CAPS words
                len(self.ALL_CAPS_PATTERN.findall(t)),
                # Exclamation marks
                t.count('!'),
                # Question marks
                t.count('?'),
                # Dollar signs
                t.count('$'),
                # Message length
                len(t),
                # Word count
                len(words),
                # Avg word length
                np.mean([len(w) for w in words]) if words else 0,
                # Sentence count
                len([s for s in sentences if s.strip()]),
                # Ratio non-alpha chars
                self._ratio_non_alpha(t),
                # Entropy of text (random-looking text = high entropy)
                self._entropy(tl[:500]),
                # Contains "click here" or similar
                int(bool(re.search(r'click\s+(here|below|link|now|this)', tl))),
                # Contains unsubscribe (spam marker)
                int('unsubscribe' in tl),
                # Contains reply-to mismatch indicator
                int(bool(re.search(r'reply.?to\b', tl))),
                # Has suspicious URL pattern
                int(any(re.search(r'(login|verify|secure|account|update)', u.lower()) for u in urls)),
                # Has IP-based URL
                int(any(re.search(r'https?://\d{1,3}\.\d{1,3}', u) for u in urls)),
                # Multiple URLs
                int(len(urls) > 1),
                # Contains "dear customer/user/member"
                int(bool(re.search(r'\bdear\s+(customer|user|member|account holder|client|friend|sir|madam)\b', tl))),
                # Greeting with generic name
                int(bool(re.search(r'\b(hello|hi|dear|greetings)\b', tl[:50]))),
                # Contains threatening language
                int(bool(re.search(r'\b(legal action|arrest|police|court|terminate|close your account|permanent(ly)?)\b', tl))),
                # Contains "you have been selected"
                int(bool(re.search(r'\b(selected|chosen|eligible|qualified|awarded)\b', tl))),
                # Has base64-like content
                int(bool(re.search(r'[A-Za-z0-9+/]{30,}={0,2}', t))),
                # Ratio of digits
                sum(c.isdigit() for c in t) / max(len(t), 1),
                # Contains crypto keywords
                int(bool(re.search(r'\b(bitcoin|crypto|ethereum|wallet|seed phrase|private key|nft|blockchain)\b', tl))),
                # Contains fake invoice / shipment
                int(bool(re.search(r'\b(invoice|receipt|shipment|tracking|delivery|order #|package|parcel)\b', tl))),
            ]
            out.append(feats)
        return np.array(out, dtype=float)

    def fit(self, X, y=None):
        return self

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 ── ENSEMBLE ML MODEL
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def build_model():
    """
    Builds a stacked ensemble even without a dataset.
    If a dataset exists, trains on it. Otherwise returns a heuristic-only flag.
    """
    status_parts = []

    # ── Feature pipeline ──────────────────────────────────────────────────────
    text_features = FeatureUnion([
        ('word_tfidf', TfidfVectorizer(
            analyzer='word', stop_words='english',
            max_features=12000, ngram_range=(1, 3),
            sublinear_tf=True, min_df=1
        )),
        ('char_tfidf', TfidfVectorizer(
            analyzer='char_wb', max_features=8000,
            ngram_range=(3, 5), sublinear_tf=True, min_df=1
        )),
        ('word_counts', CountVectorizer(
            analyzer='word', max_features=5000,
            ngram_range=(1, 2), binary=True
        )),
    ])

    # ── Classifiers ───────────────────────────────────────────────────────────
    lr  = LogisticRegression(max_iter=2000, C=2.0, solver='lbfgs', class_weight='balanced')
    cnb = ComplementNB(alpha=0.1)
    svm = CalibratedClassifierCV(LinearSVC(max_iter=3000, C=1.5, class_weight='balanced'))
    gb  = GradientBoostingClassifier(n_estimators=120, max_depth=4, learning_rate=0.1, subsample=0.8)

    # Ensemble on TF-IDF + char n-gram features
    tfidf_ensemble = Pipeline([
        ('feat', text_features),
        ('vote', VotingClassifier(
            estimators=[('lr', lr), ('cnb', cnb), ('svm', svm)],
            voting='soft', weights=[3, 1, 2]
        ))
    ])

    if os.path.exists('sms_spam_10000_dataset.csv'):
        try:
            df = pd.read_csv('sms_spam_10000_dataset.csv')
            if 'Category' in df.columns and 'Message' in df.columns:
                X, y_raw = df['Message'].astype(str), df['Category']
            elif 'v1' in df.columns and 'v2' in df.columns:
                X, y_raw = df['v2'].astype(str), df['v1']
            else:
                return None, "Column names not recognised."

            y = y_raw.apply(lambda v: 1 if str(v).strip().lower() == 'spam' else 0)

            # Augment with hard phishing examples so model isn't just spam-trained
            aug_phishing = [
                "Your PayPal account has been limited. Verify now: http://paypal-secure-login.xyz/verify",
                "URGENT: Your bank account will be suspended. Click here to verify: http://192.168.1.1/bank",
                "You have won $1,000,000 lottery prize. Send your details to claim@lottery-win.tk",
                "Dear customer, unusual activity detected. Login here immediately: http://amaz0n-accounts.top",
                "IRS NOTICE: You owe back taxes. Call 1-800-xxx-xxxx immediately or face arrest.",
                "Your Apple ID is locked. Verify your information: http://apple-id.secure-verify.click",
                "Hi, I'm a Nigerian prince and need your help transferring $50 million. Contact me urgently.",
                "Your Netflix subscription has expired! Update payment: http://netflix-billing.online/update",
                "CONGRATULATIONS! You've been selected for a $500 Amazon gift card. Claim now!",
                "Security alert: Someone logged into your Google account. Verify: http://google-security.xyz",
                "Your crypto wallet needs verification. Enter seed phrase at: http://metamask-verify.top",
                "Final notice: Your Microsoft account password expires today. Reset: http://ms-secure.click",
                "Package held at customs. Pay $2.99 to release: http://dhl-customs-fee.online",
                "Your social security number has been suspended. Call SSA fraud dept now: 1-800-xxx",
                "BANK ALERT: Card ending 4242 charged $847. Not you? Click: http://chase-verify.xyz",
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
            ]

            X_aug = pd.concat([X, pd.Series(aug_phishing + aug_safe)], ignore_index=True)
            y_aug = pd.concat([y, pd.Series([1]*len(aug_phishing) + [0]*len(aug_safe))], ignore_index=True)

            tfidf_ensemble.fit(X_aug, y_aug)
            status_parts.append(f"Trained on {len(X_aug):,} messages")
        except Exception as e:
            return None, f"Training error: {e}"
    else:
        # Train on augmented synthetic data only
        synthetic_spam = [
            "URGENT: Your account has been compromised. Verify immediately at http://secure-login.xyz",
            "Congratulations! You won $500 Amazon gift card. Click to claim now!",
            "Your PayPal is limited. Confirm your details: http://paypal-verify.top",
            "ALERT: Suspicious login on your bank account. Verify now or account will be closed.",
            "You owe back taxes to the IRS. Call immediately or face legal action.",
            "Your Apple ID locked. Update your information: http://apple-id-verify.click",
            "Nigerian prince needs your help. $50 million transfer. Reply urgently.",
            "Netflix billing failed! Update payment info or lose access: http://netflix-update.online",
            "FREE iPhone 14! You've been selected. Claim your prize: http://free-iphone.tk",
            "Your crypto wallet detected unusual activity. Verify seed phrase at: http://metamask.top",
            "Microsoft account suspended. Login at http://microsoft-secure.xyz to restore.",
            "DHL: Package held. Pay customs fee $2.99: http://dhl-package.online",
            "Your SSN has been suspended. Call SSA fraud department immediately.",
            "Bank: Card charged $847. Dispute at http://chase-verify.club",
            "Winner! $1,000,000 sweepstakes prize. Send details to claim@prize.ml",
        ]
        synthetic_ham = [
            "See you at the meeting tomorrow at 9am",
            "Can you send me the report by Friday?",
            "Happy birthday! Hope you have a great day",
            "Your order has shipped and will arrive Thursday",
            "Reminder: Doctor appointment at 2pm on Wednesday",
            "Thanks for dinner last night, it was great!",
            "The project deadline has been moved to next Monday",
            "Please review the attached document and give feedback",
            "Your reservation is confirmed for Saturday, 7pm",
            "Call me when you get a chance, nothing urgent",
        ]
        X_syn = pd.Series(synthetic_spam + synthetic_ham)
        y_syn = pd.Series([1]*len(synthetic_spam) + [0]*len(synthetic_ham))
        tfidf_ensemble.fit(X_syn, y_syn)
        status_parts.append("Heuristic + lightweight model (no dataset found)")

    return tfidf_ensemble, " · ".join(status_parts)

model, model_status = build_model()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 ── HEURISTIC TEXT ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

TEXT_RULES = [
    # (regex, weight 1-5, short_name, explanation)
    (r'\b(urgent|act now|immediately|expires? (today|now|soon)|respond (now|immediately)|don\'t delay|asap)\b',
     4, "Urgency / Pressure",
     "Phishing relies on making you act before you think. Words like 'urgent' or 'expires today' are psychological pressure tactics designed to bypass your rational judgment."),

    (r'\b(your account (will be|has been|is|was) (suspended|blocked|locked|disabled|terminated|compromised|hacked))\b',
     5, "Account Threat Language",
     "Threatening account suspension or compromise is the #1 fear-based phishing tactic. Legitimate companies send formal notices by post — not threatening SMS/emails."),

    (r'\b(verify|confirm|validate|update|re-?verify) (your )?(account|identity|information|details|password|card|billing)\b',
     4, "Verification Request",
     "Legitimate services NEVER ask you to verify sensitive account details through a link in a message. Always go directly to the official website."),

    (r'\b(won|winner|selected|awarded|prize|lottery|jackpot|sweepstake|congratulations.*you|you.*won)\b',
     4, "Prize / Lottery Bait",
     "Fake prize notifications are a classic social engineering technique. Real lotteries require you to enter — you cannot win one you didn't enter."),

    (r'(\$[\d,]+\s*(million|thousand|hundred)?|\d{4,}[\s]*(dollars?|usd|gbp|eur))',
     3, "Large Financial Amount",
     "Unexpectedly large money amounts in messages are bait. Advance-fee fraud promises huge sums to steal small upfront 'processing fees'."),

    (r'\b(click|tap|open|follow|visit)\s+(this\s+)?(link|url|below|here|button)\b',
     3, "Suspicious CTA Link",
     "Directing you to click an unverified link is the primary phishing delivery mechanism. The real destination is hidden behind a redirect."),

    (r'\b(password|passwd|pin\b|otp|one.?time.?pass|verification code|security code|access code)\b',
     5, "Password / OTP Request",
     "No legitimate service will ever ask for your password, PIN, or OTP via SMS or email. This is an absolute red line."),

    (r'\b(credit card|debit card|card number|cvv|expiry date|billing info|full card)\b',
     5, "Card Details Request",
     "Requesting card details (number, CVV, expiry) in a message is always fraudulent. Legitimate companies process payments on secure, verified pages."),

    (r'\b(social security|ssn|national insurance|passport number|driver.?s? licen[sc]e)\b',
     5, "Government ID Request",
     "Requesting government ID numbers (SSN, passport) via message is a strong indicator of identity theft fraud."),

    (r'\b(transfer|wire|send)\s+.{0,20}(bitcoin|btc|ethereum|eth|crypto|gift card|itunes|google play|steam)\b',
     5, "Crypto / Gift Card Payment",
     "Asking for payment in gift cards or cryptocurrency is the #1 scam payment method because it is untraceable and irreversible. The IRS, police, and utilities will NEVER ask this."),

    (r'\b(nigerian?|prince|inheritance|diplomat|refugee|widow|dying|terminal illness).{0,40}(million|transfer|fund)\b',
     5, "Advance-Fee Fraud (419 Scam)",
     "This matches the classic advance-fee fraud pattern originating from Nigeria (419 scams). The goal is to extract small 'processing fees' with promise of large returns."),

    (r'\b(dear\s+(customer|user|member|account holder|client|valued|sir|madam|friend))\b',
     2, "Generic Impersonal Greeting",
     "Phishing messages use generic greetings because they are sent in bulk. Legitimate services address you by your actual name."),

    (r'\b(legal action|arrest|sued|warrant|court order|fined|prosecuted|reported to police)\b',
     4, "Legal Threat / Intimidation",
     "Threatening legal action or arrest is a scare tactic. Government agencies initiate contact by official post, not by SMS or email."),

    (r'\b(re-?activate|re-?enable|restore access|re-?verify|recover your account)\b',
     3, "Account Reactivation Scam",
     "Fake reactivation requests drive you to credential-harvesting pages that look identical to real login pages."),

    (r'\b(unsubscribe|opt.?out|stop receiving|manage preferences)\b',
     1, "Unsubscribe Footer",
     "Presence of an unsubscribe notice suggests bulk/marketing communication — often used in phishing emails to appear legitimate."),

    (r'\b(tracking (number|id|code)|your (order|package|parcel|shipment) (is|has))\b',
     2, "Fake Shipment Notification",
     "Fake parcel delivery notifications are common phishing lures, especially impersonating FedEx, UPS, USPS, and DHL."),

    (r'\b(seed phrase|private key|wallet address|connect wallet|approve transaction|metamask)\b',
     5, "Crypto Wallet Phishing",
     "Asking for a seed phrase or private key will instantly drain your entire crypto wallet. No platform ever needs this."),

    (r'\b(invoice|receipt|billing statement|payment (due|failed|declined|overdue))\b',
     2, "Fake Invoice / Payment Alert",
     "Fake invoice/payment alerts create urgency around non-existent debts to harvest payment details."),
]

def analyze_message_heuristic(text: str):
    tl = text.lower()
    hits = []
    total_weight = 0
    for pattern, weight, name, explanation in TEXT_RULES:
        if re.search(pattern, tl, re.I):
            hits.append((weight, name, explanation))
            total_weight += weight
    max_possible = sum(w for w, *_ in TEXT_RULES)
    score = min(int((total_weight / max(max_possible, 1)) * 100 * 3.2), 100)
    return score, hits

def analyze_message(text: str):
    h_score, hits = analyze_message_heuristic(text)

    ml_score = 0
    if model:
        try:
            prob = model.predict_proba([text])[0][1]
            ml_score = int(prob * 100)
        except:
            ml_score = 0

    # Weighted blend
    if model:
        final = int(0.55 * ml_score + 0.45 * h_score)
    else:
        final = h_score

    # Boost for critical patterns (password, OTP, SSN, crypto)
    critical = [h for h in hits if h[0] == 5]
    if critical:
        final = min(final + 15 * len(critical), 100)

    final = min(final, 100)

    if final >= 65:   level = "PHISHING"
    elif final >= 35: level = "SUSPICIOUS"
    else:             level = "SAFE"

    return ml_score, h_score, final, level, hits

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 ── URL ANALYSIS ENGINE (13 signals + homoglyph + entropy)
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_homoglyphs(domain: str) -> str:
    """Replace common digit/symbol lookalikes with letters."""
    result = domain
    for src, dst in HOMOGLYPH_MAP.items():
        result = result.replace(src, dst)
    return result

def _domain_entropy(domain: str) -> float:
    if not domain: return 0.0
    counts = Counter(domain)
    total = len(domain)
    return -sum((c/total)*math.log2(c/total) for c in counts.values())

def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a):
        curr = [i+1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j]+(ca!=cb), curr[-1]+1, prev[j+1]+1))
        prev = curr
    return prev[-1]

def _closest_brand(domain: str) -> tuple:
    """Returns (brand, distance) for the most similar brand name."""
    best, best_d = None, 999
    for brand in BRAND_NAMES:
        d = _levenshtein(domain.lower(), brand.lower())
        if d < best_d:
            best, best_d = brand, d
    return best, best_d

URL_RULES = []  # populated dynamically in analyze_url

def analyze_url(raw_url: str):
    url = raw_url.strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    ext     = tldextract.extract(url)
    domain  = ext.domain.lower()
    suffix  = ext.suffix.lower()
    sub     = ext.subdomain.lower()
    parsed  = urlparse(url)
    path    = parsed.path.lower()
    query   = parsed.query.lower()
    full    = url.lower()

    # Decode URL encoding
    try:
        decoded = unquote(url).lower()
    except:
        decoded = full

    score = 0
    flags = []  # list of (severity 1-5, name, explanation)

    # ── 0. Whitelist ─────────────────────────────────────────────────────────
    clean_sub = sub.replace('www', '').replace('.', '').strip()
    if domain in WHITELIST and not clean_sub:
        return 0, "SAFE", [(0, "Verified Trusted Domain", "This domain is on the verified whitelist of major legitimate services.")]

    # ── 1. IP Address as host ────────────────────────────────────────────────
    if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", parsed.netloc.split(':')[0]):
        score += 90
        flags.append((5, "IP Address Used as Domain",
            f"The URL uses a raw IP address ({parsed.netloc}) instead of a domain name. "
            "Legitimate websites always use proper domain names. IPs are used to hide the server's identity."))

    # ── 2. No HTTPS ──────────────────────────────────────────────────────────
    if parsed.scheme == 'http':
        score += 20
        flags.append((3, "Unencrypted HTTP Connection",
            "The link uses HTTP, not HTTPS. All legitimate websites handling accounts, "
            "logins, or payments exclusively use HTTPS to encrypt your data in transit."))

    # ── 3. Brand in subdomain (impersonation) ────────────────────────────────
    for brand in BRAND_NAMES:
        if brand in sub and domain not in WHITELIST:
            score += 70
            flags.append((5, f"Brand Name '{brand.title()}' Impersonated in Subdomain",
                f"The subdomain contains '{brand}' but the real domain is '{domain}.{suffix}'. "
                f"In a URL like 'paypal.evil.com', the actual website is evil.com — not PayPal. "
                "This is the most common phishing technique."))
            break

    # ── 4. Homoglyph / typosquatting detection ───────────────────────────────
    norm_domain = _normalize_homoglyphs(domain)
    if norm_domain != domain:
        for brand in BRAND_NAMES:
            if brand in norm_domain and domain not in WHITELIST:
                score += 65
                flags.append((5, "Homoglyph / Character Substitution Attack",
                    f"The domain '{domain}' uses character substitutions (e.g. '0' for 'o', '1' for 'l') "
                    f"to impersonate '{norm_domain}'. This is designed to fool visual inspection."))
                break

    # ── 5. Levenshtein typosquatting ─────────────────────────────────────────
    best_brand, dist = _closest_brand(domain)
    if 0 < dist <= 2 and domain not in WHITELIST and len(domain) > 3:
        score += 55
        flags.append((4, f"Typosquatting — Resembles '{best_brand.title()}'",
            f"The domain '{domain}' is only {dist} character(s) away from '{best_brand}'. "
            "Attackers register near-identical domains (e.g. 'amazzon.com', 'paypa1.com') to trick users."))

    # ── 6. Subdomain count ───────────────────────────────────────────────────
    sub_parts = [p for p in sub.split('.') if p and p != 'www']
    if len(sub_parts) >= 3:
        score += 35
        flags.append((3, f"Excessive Subdomains ({len(sub_parts)} levels)",
            f"Found subdomain: '{sub}'. Having 3+ levels of subdomains (e.g. login.secure.payment.site.com) "
            "is a tactic to make fake URLs appear official by burying the real domain at the end."))

    # ── 7. @ symbol ──────────────────────────────────────────────────────────
    if '@' in url:
        score += 90
        flags.append((5, "'@' Symbol in URL (Credential Hijack)",
            "Browsers ignore everything before '@' in a URL. "
            "So 'http://paypal.com@evil.com' actually loads evil.com. This is a direct browser exploit."))

    # ── 8. Suspicious TLD ────────────────────────────────────────────────────
    if suffix in BAD_TLDS:
        score += 45
        flags.append((3, f"High-Risk Top-Level Domain (.{suffix})",
            f"The '.{suffix}' TLD is widely used for free/throwaway domains. "
            "It appears disproportionately often in phishing, malware, and spam campaigns due to low/zero registration cost."))

    # ── 9. Excessive hyphens ─────────────────────────────────────────────────
    hyph = domain.count('-')
    if hyph >= 2:
        score += min(15 + hyph * 8, 40)
        flags.append((2, f"Excessive Hyphens in Domain ({hyph})",
            f"Domain: '{domain}' contains {hyph} hyphens. Phishing domains use hyphens to appear legitimate "
            "(e.g. 'paypal-secure-verify-account.com'). Legitimate brands rarely have hyphens."))

    # ── 10. Long URL ─────────────────────────────────────────────────────────
    if len(url) > 85:
        score += 18
        flags.append((2, f"Unusually Long URL ({len(url)} chars)",
            "Very long URLs are constructed to hide the real domain (usually at the start) "
            "in a wall of confusing parameters, making it hard to see where the link actually goes."))

    # ── 11. Sensitive keywords in path/query ─────────────────────────────────
    sensitive = ['login','verify','account','update','secure','confirm','password',
                 'banking','signin','credential','validate','wallet','payment','recover',
                 'authenticate','billing','otp','pin','reset','access']
    found_kw = [kw for kw in sensitive if kw in path or kw in query]
    if found_kw:
        score += 25
        flags.append((3, f"Sensitive Keywords in URL Path: {', '.join(found_kw[:4])}",
            "Legitimate sites do not typically place action keywords like 'login', 'verify', or 'password' "
            "directly in their URL paths. Phishing pages mimic real login/verification pages."))

    # ── 12. High digit ratio in domain ──────────────────────────────────────
    digit_r = sum(c.isdigit() for c in domain) / max(len(domain), 1)
    if digit_r > 0.35:
        score += 30
        flags.append((3, f"High Digit Ratio in Domain ({int(digit_r*100)}%)",
            f"Domain '{domain}' is {int(digit_r*100)}% digits. Auto-generated phishing domains "
            "often contain random numbers as filler to avoid detection."))

    # ── 13. Open redirect ────────────────────────────────────────────────────
    redirect_params = ['redirect', 'url=', 'goto=', 'next=', 'return=', 'returnurl', 'dest=', 'forward=']
    if any(p in full for p in redirect_params):
        score += 35
        flags.append((4, "Open Redirect Parameter Detected",
            "The URL contains a redirect parameter (e.g. '?url=' or '?goto='). This allows attackers "
            "to chain URLs — the first link appears safe but silently forwards you to a malicious page."))

    # ── 14. URL shortener ────────────────────────────────────────────────────
    shorteners = ['bit.ly','tinyurl','t.co','goo.gl','ow.ly','buff.ly','rebrand.ly',
                  'short.link','rb.gy','cutt.ly','tiny.cc','is.gd','v.gd','snip.ly']
    if any(s in full for s in shorteners):
        score += 30
        flags.append((3, "URL Shortener Detected",
            "URL shorteners hide the final destination. Phishing links are frequently disguised using "
            "services like bit.ly or tinyurl to bypass link scanners and fool recipients."))

    # ── 15. High domain entropy (randomness) ─────────────────────────────────
    ent = _domain_entropy(domain)
    if ent > 3.7 and len(domain) > 8:
        score += 25
        flags.append((3, f"High Domain Randomness (entropy {ent:.2f})",
            f"The domain '{domain}' has a high character entropy score of {ent:.2f}, indicating it may be "
            "algorithmically generated (DGA — Domain Generation Algorithm), common in phishing and malware."))

    # ── 16. Mixed unicode / punycode ─────────────────────────────────────────
    if 'xn--' in domain.lower() or any(ord(c) > 127 for c in raw_url):
        score += 50
        flags.append((4, "Punycode / Unicode Domain (IDN Homograph Attack)",
            "The domain uses Unicode or punycode (xn--...). Attackers register domains with characters from "
            "other alphabets that look identical to Latin letters (e.g. Cyrillic 'а' vs Latin 'a') "
            "to create visually indistinguishable fake domains."))

    # ── 17. Encoded characters in path ───────────────────────────────────────
    if re.search(r'%[0-9a-fA-F]{2}', path) and decoded != full:
        score += 20
        flags.append((2, "URL Encoding Used to Obfuscate Path",
            "The URL path contains percent-encoded characters (e.g. %2F, %40). "
            "While sometimes legitimate, this is frequently used to evade content filters and hide malicious paths."))

    score = min(score, 100)

    if score >= 60:   level = "PHISHING"
    elif score >= 30: level = "SUSPICIOUS"
    else:             level = "SAFE"

    if not flags:
        flags.append((0, "Clean URL Structure",
            "No suspicious patterns were found. The URL appears structurally clean and standard."))

    return score, level, flags

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 ── CSS (Dark Minimal — Monochrome Terminal Aesthetic)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;0,600;1,400&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background-color: #080808 !important;
    color: #d4d4d4 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 1.5rem 5rem !important; max-width: 740px !important; }

/* ── INPUTS ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background-color: #111 !important;
    border: 1px solid #252525 !important;
    border-radius: 5px !important;
    color: #d4d4d4 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    padding: 13px 14px !important;
    transition: border-color 0.2s;
    caret-color: #d4d4d4;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #404040 !important;
    box-shadow: none !important;
    outline: none !important;
}

/* ── BUTTON ── */
div.stButton > button {
    background-color: #d4d4d4 !important;
    color: #080808 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.76rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 11px 0 !important;
    width: 100% !important;
    transition: background 0.15s, transform 0.1s !important;
}
div.stButton > button:hover {
    background-color: #b8b8b8 !important;
    transform: translateY(-1px) !important;
}
div.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── SPINNER ── */
.stSpinner { color: #444 !important; }

/* ── OPTION MENU ── */
ul[data-testid="stHorizontalBlock"] { display: none !important; }

/* ── ALERT/WARNING ── */
.stAlert { border-radius: 5px !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.78rem !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #111; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 ── UI COMPONENTS (rendered via st.markdown)
# ══════════════════════════════════════════════════════════════════════════════

def render_score_block(score: int, level: str, sub_info: str = ""):
    if level == "PHISHING":
        color, bg = "#ff4444", "rgba(255,68,68,0.06)"
        border = "rgba(255,68,68,0.25)"
    elif level == "SUSPICIOUS":
        color, bg = "#ffaa00", "rgba(255,170,0,0.06)"
        border = "rgba(255,170,0,0.25)"
    else:
        color, bg = "#22dd77", "rgba(34,221,119,0.06)"
        border = "rgba(34,221,119,0.2)"

    bar_bg = f"linear-gradient(90deg, {color} {score}%, #1a1a1a {score}%)"

    st.markdown(f"""
    <div style="
        border: 1px solid {border};
        border-radius: 8px;
        padding: 28px 24px 22px;
        background: {bg};
        margin: 18px 0 22px;
        text-align: center;
    ">
        <div style="
            font-family: 'IBM Plex Mono', monospace;
            font-size: 5rem;
            font-weight: 300;
            letter-spacing: -0.05em;
            color: {color};
            line-height: 1;
        ">{score}<span style="font-size:1.8rem; color:#333;">%</span></div>

        <div style="
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.7rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: {color};
            margin-top: 6px;
            margin-bottom: 14px;
        ">{level}</div>

        <div style="height:3px; background:{bar_bg}; border-radius:2px; margin-bottom:12px;"></div>

        {f'<div style="font-family:IBM Plex Mono,monospace; font-size:0.68rem; color:#444; letter-spacing:0.05em;">{sub_info}</div>' if sub_info else ''}
    </div>
    """, unsafe_allow_html=True)


def render_section_label(text: str):
    st.markdown(f"""
    <div style="
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.64rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #3a3a3a;
        margin: 24px 0 10px;
        padding-bottom: 7px;
        border-bottom: 1px solid #181818;
    ">{text}</div>
    """, unsafe_allow_html=True)


SEVERITY_COLORS = {0: "#22dd77", 1: "#aaaaaa", 2: "#ffaa00", 3: "#ff8800", 4: "#ff5500", 5: "#ff4444"}

def render_flag(severity: int, name: str, explanation: str, is_safe: bool = False):
    color = SEVERITY_COLORS.get(severity, "#aaa") if not is_safe else "#22dd77"
    icon  = "✓" if is_safe else ("◉" if severity >= 4 else "◎")
    st.markdown(f"""
    <div style="
        border: 1px solid #1c1c1c;
        border-left: 3px solid {color};
        border-radius: 0 5px 5px 0;
        padding: 13px 15px 13px 16px;
        margin-bottom: 9px;
        background: #0e0e0e;
    ">
        <div style="
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.75rem;
            color: {color};
            letter-spacing: 0.03em;
            margin-bottom: 5px;
        ">{icon} {name}</div>
        <div style="
            font-family: 'IBM Plex Sans', sans-serif;
            font-size: 0.82rem;
            color: #666;
            line-height: 1.55;
        ">{explanation}</div>
    </div>
    """, unsafe_allow_html=True)


def render_safety_card(rec: dict, level: str):
    if level == "SAFE":
        return
    st.markdown(f"""
    <div style="
        border: 1px solid rgba(34,221,119,0.2);
        border-radius: 8px;
        padding: 20px 20px 16px;
        background: rgba(34,221,119,0.04);
        margin-top: 6px;
    ">
        <div style="
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.64rem;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            color: #22dd77;
            margin-bottom: 14px;
        ">⬡ Safety Recommendations</div>

        <div style="display:grid; gap:10px;">

            <div style="display:flex; gap:12px; align-items:flex-start;">
                <span style="font-family:IBM Plex Mono,monospace; font-size:0.68rem; color:#333; min-width:70px; padding-top:1px;">WEBSITE</span>
                <span style="font-family:IBM Plex Sans,sans-serif; font-size:0.84rem; color:#c8c8c8; line-height:1.4;">{rec['site']}</span>
            </div>

            <div style="display:flex; gap:12px; align-items:flex-start;">
                <span style="font-family:IBM Plex Mono,monospace; font-size:0.68rem; color:#333; min-width:70px; padding-top:1px;">APP</span>
                <span style="font-family:IBM Plex Sans,sans-serif; font-size:0.84rem; color:#c8c8c8; line-height:1.4;">{rec['app']}</span>
            </div>

            <div style="display:flex; gap:12px; align-items:flex-start;">
                <span style="font-family:IBM Plex Mono,monospace; font-size:0.68rem; color:#333; min-width:70px; padding-top:1px;">CONTACT</span>
                <span style="font-family:IBM Plex Sans,sans-serif; font-size:0.84rem; color:#c8c8c8; line-height:1.4;">{rec['contact']}</span>
            </div>

            <div style="
                margin-top:4px;
                padding: 10px 14px;
                background: rgba(34,221,119,0.07);
                border-radius: 4px;
                border-left: 2px solid rgba(34,221,119,0.5);
            ">
                <span style="font-family:IBM Plex Mono,monospace; font-size:0.68rem; color:#22dd77; letter-spacing:0.06em;">TIP  </span>
                <span style="font-family:IBM Plex Sans,sans-serif; font-size:0.82rem; color:#888; line-height:1.5;">{rec['tip']}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_general_safety_tips():
    tips = [
        ("Never click links in unexpected messages", "If a message creates urgency about your account, go directly to the website by typing it yourself — never through a link."),
        ("Check the full URL before clicking", "Hover over links. Look at the very last part before the first slash — that's the real domain. Everything before it can be faked."),
        ("Enable two-factor authentication (2FA)", "Even if your password is stolen, 2FA stops attackers from accessing your account. Use an app (Google Authenticator, Authy) not SMS."),
        ("Your bank / government will never ask via SMS", "No bank, IRS, Social Security, or government agency will demand urgent action via text or email. Call them on the official number."),
        ("Report phishing", "Forward phishing SMS to 7726 (SPAM). Report phishing emails to reportphishing@apwg.org or phishing-report@us-cert.gov."),
    ]
    st.markdown("""
    <div style="
        border: 1px solid #1c1c1c;
        border-radius: 8px;
        padding: 20px;
        background: #0b0b0b;
        margin-top: 6px;
    ">
        <div style="font-family:IBM Plex Mono,monospace; font-size:0.64rem; letter-spacing:0.15em; text-transform:uppercase; color:#3a3a3a; margin-bottom:14px;">
            General Safety Guidelines
        </div>
    """, unsafe_allow_html=True)
    for title, detail in tips:
        st.markdown(f"""
        <div style="margin-bottom:12px; padding-bottom:12px; border-bottom:1px solid #161616;">
            <div style="font-family:IBM Plex Mono,monospace; font-size:0.76rem; color:#888; margin-bottom:3px;">{title}</div>
            <div style="font-family:IBM Plex Sans,sans-serif; font-size:0.8rem; color:#484848; line-height:1.5;">{detail}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 ── APP LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div style="padding: 1rem 0 2rem; border-bottom: 1px solid #141414; margin-bottom: 2rem;">
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
        <span style="font-family:IBM Plex Mono,monospace; font-size:1.3rem; color:#d4d4d4; font-weight:600; letter-spacing:-0.02em;">⬡ PhishGuard</span>
        <span style="font-family:IBM Plex Mono,monospace; font-size:0.62rem; color:#2a2a2a; letter-spacing:0.1em; padding: 2px 8px; border:1px solid #1e1e1e; border-radius:20px;">AI · v3.0</span>
    </div>
    <p style="font-family:IBM Plex Mono,monospace; font-size:0.7rem; color:#2e2e2e; letter-spacing:0.08em;">
        PHISHING DETECTION — URL · MESSAGE · EMAIL · SMS
    </p>
</div>
""", unsafe_allow_html=True)

selected = option_menu(
    menu_title=None,
    options=["Message / Email / SMS", "URL / Link"],
    icons=["envelope-fill", "link-45deg"],
    orientation="horizontal",
    styles={
        "container": {
            "background-color": "#0d0d0d",
            "border": "1px solid #1e1e1e",
            "border-radius": "5px",
            "padding": "2px",
            "margin-bottom": "2rem"
        },
        "nav-link": {
            "font-family": "IBM Plex Mono, monospace",
            "font-size": "0.72rem",
            "letter-spacing": "0.06em",
            "color": "#383838",
            "padding": "9px 20px",
            "border-radius": "3px"
        },
        "nav-link-selected": {
            "background-color": "#1a1a1a",
            "color": "#d4d4d4",
            "font-weight": "500"
        },
        "icon": {"display": "none"},
    }
)

# ── TAB 1: Message / Email / SMS ─────────────────────────────────────────────
if selected == "Message / Email / SMS":
    render_section_label("Input — Paste message, email body, or SMS")
    msg_input = st.text_area(
        label="msg",
        label_visibility="collapsed",
        placeholder="Paste the full message here…\n\nExample: 'URGENT: Your PayPal account has been limited. Verify immediately at http://paypal-secure.xyz'",
        height=170
    )

    if st.button("ANALYZE MESSAGE →", key="msg_btn"):
        if not msg_input.strip():
            st.warning("Please paste a message to analyze.")
        else:
            with st.spinner("Analyzing with ensemble engine…"):
                ml_s, h_s, final_s, level, flags = analyze_message(msg_input)

            render_score_block(
                final_s, level,
                f"ML model: {ml_s}%  ·  Heuristic: {h_s}%  ·  Final: {final_s}%"
            )

            render_section_label(f"Detection Reasons — {len(flags)} signal(s) found")
            if flags:
                for w, name, expl in sorted(flags, key=lambda x: -x[0]):
                    render_flag(w, name, expl, is_safe=False)
            else:
                render_flag(0, "No Suspicious Patterns Detected",
                    "This message does not match any known phishing patterns in our rule database.", is_safe=True)

            if level in ("PHISHING", "SUSPICIOUS"):
                render_section_label("Safety Recommendations")
                flags_text = " ".join(n for _, n, _ in flags)
                rec = get_safety_card("", flags_text)
                render_safety_card(rec, level)
            else:
                render_section_label("General Safety Tips")
                render_general_safety_tips()

# ── TAB 2: URL / Link ────────────────────────────────────────────────────────
elif selected == "URL / Link":
    render_section_label("Input — Paste URL or link")
    url_input = st.text_input(
        label="url",
        label_visibility="collapsed",
        placeholder="https://example.com/login?verify=account"
    )

    if st.button("SCAN URL →", key="url_btn"):
        if not url_input.strip():
            st.warning("Please paste a URL to scan.")
        else:
            with st.spinner("Running deep URL forensics…"):
                score, level, flags = analyze_url(url_input.strip())

            render_score_block(score, level)

            render_section_label(f"Detection Reasons — {len(flags)} signal(s) found")
            for sev, name, expl in sorted(flags, key=lambda x: -x[0]):
                render_flag(sev, name, expl, is_safe=(level == "SAFE" or sev == 0))

            if level in ("PHISHING", "SUSPICIOUS"):
                render_section_label("Safety Recommendations")
                flags_text = " ".join(n for _, n, _ in flags)
                rec = get_safety_card(url_input, flags_text)
                render_safety_card(rec, level)
                render_section_label("General Safety Tips")
                render_general_safety_tips()


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="
    margin-top: 4rem;
    padding-top: 1rem;
    border-top: 1px solid #141414;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 6px;
">
    <span style="font-family:IBM Plex Mono,monospace; font-size:0.6rem; color:#222; letter-spacing:0.1em;">PHISHGUARD v3.0</span>
    <span style="font-family:IBM Plex Mono,monospace; font-size:0.6rem; color:#222; letter-spacing:0.06em;">{model_status}</span>
    <span style="font-family:IBM Plex Mono,monospace; font-size:0.6rem; color:#222; letter-spacing:0.06em;">17 URL SIGNALS · 18 TEXT RULES · ENSEMBLE ML</span>
</div>
""", unsafe_allow_html=True)
