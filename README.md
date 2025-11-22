# Spam Detection App (BiLSTM + Streamlit)

This project is an SMS **Spam Message Detection** system built using a **Bidirectional LSTM (BiLSTM)** model with **TensorFlow/Keras**, and a **Streamlit** web app for easy interaction.

The goal is to classify SMS messages as:

- **SPAM** (fraud / scam / promotional)
- **HAM** (normal / genuine messages)

---

## 🚀 Features

- 🔍 **Single Message Analysis** – Type a message and get instant SPAM/HAM prediction  
- 📂 **Bulk Analysis** – Paste multiple messages or upload a CSV file and classify them all at once  
- 🎚 **Adjustable Threshold** – Change spam sensitivity using a slider  
- 📊 **Dataset Insights** – See spam vs ham counts and random sample messages  
- 🧠 **BiLSTM Model** – Deep learning model trained on a 10,000-message dataset  
- 💾 **Prediction History** – View and download previous predictions

---

## 🧠 Model Details

- Text preprocessing with Tokenizer + Padding  
- **Embedding layer** (word embeddings)  
- **Bidirectional LSTM (BiLSTM)** with 64 units  
- **Dropout layer** to reduce overfitting  
- **Dense layer (ReLU)**  
- **Output layer (Sigmoid)** → returns spam probability between 0 and 1  

The model is trained using **binary cross-entropy loss** and **Adam optimizer**.

---

## 📚 Dataset

- Custom synthetic + realistic SMS dataset  
- **10,000 total messages**
  - 5,000 **HAM**
  - 5,000 **SPAM**
- Stored in file: `sms_spam_10000_dataset.csv`

Each row contains:

- `Category` → `ham` or `spam`  
- `Message` → SMS text

---

## 🛠 Tech Stack

- **Python**
- **TensorFlow / Keras**
- **Streamlit**
- **Pandas, NumPy**
- **scikit-learn**

---

## 🧪 How to Run the Project

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Sathvik546/Spam-Detection-Project.git
cd Spam-Detection-Project
