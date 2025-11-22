# train_model.py
# Run this ONCE to train the model and save spam_model.keras + tokenizer.pickle

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# 1. LOAD THE DATASET
csv_path = "sms_spam_10000_dataset.csv"   # file must be in same folder
df = pd.read_csv(csv_path, encoding="latin-1")

# Adjust column names if needed
# Common Kaggle format: "Category" (ham/spam) and "Message" (text)
label_col = "Category"
text_col = "Message"

# Keep only needed columns and drop missing values
df = df[[label_col, text_col]].dropna()

# 2. ENCODE LABELS: ham=0, spam=1
le = LabelEncoder()
df["label"] = le.fit_transform(df[label_col])  # ham/spam -> 0/1
texts = df[text_col].astype(str).values
labels = df["label"].values

# 3. TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# 4. TOKENIZE TEXT
vocab_size = 10000
oov_token = "<OOV>"
max_len = 100  # we will use same value in app

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post", truncating="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post", truncating="post")

# 5. BUILD BiLSTM MODEL
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64, input_length=max_len),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

print("✅ Starting training...")
history = model.fit(
    X_train_pad,
    y_train,
    epochs=5,              # you can increase to 8–10 later
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 6. EVALUATE ON TEST SET
loss, acc = model.evaluate(X_test_pad, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")

# 7. SAVE MODEL AND TOKENIZER
model.save("spam_model.keras")
print("💾 Saved model as spam_model.keras")

with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("💾 Saved tokenizer as tokenizer.pickle")

print("\n🎉 Training complete!")
