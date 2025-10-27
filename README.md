# Spam Message Detector 📧

## Introduction

This project implements a spam message classifier using a Bidirectional Long Short-Term Memory (BiLSTM) neural network. The model is built using TensorFlow/Keras in Python and trained on a publicly available SMS spam dataset. The goal is to accurately distinguish between legitimate messages ('ham') and spam messages.

A simple web application built with Streamlit (`app.py`) is also included to demonstrate the model's predictions in real-time.

---

## Dataset 📊

The dataset used is `SPAM text message 20170820 - Data.csv`. It contains SMS messages labeled as either 'ham' or 'spam'.

* **Source:** [Kaggle SMS Spam Collection Dataset](https://www.kaggle.com/datasets/team-ai/spam-text-message-classification) (You can link to the source if you like)
* **Preprocessing:** The dataset was downsampled to create a balanced set of ham and spam messages for training. Text messages were tokenized and padded to a fixed sequence length.

---

## Installation ⚙️

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/Spam-Detection-Project.git](https://github.com/YourUsername/Spam-Detection-Project.git)
    cd Spam-Detection-Project
    ```
2.  **Install dependencies:** It's recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You'll need to create the `requirements.txt` file as we discussed before)*

---

## Usage 🚀

### 1. Training the Model (Jupyter Notebook)

* Open the `Spam Detector.ipynb` notebook using Jupyter Notebook or Jupyter Lab.
* Run all the cells in the notebook. This will train the model and save the `spam_model.keras` and `tokenizer.pickle` files to your Desktop (or you can modify the save path in the notebook).

### 2. Running the Web Application (Streamlit)

* Make sure the saved `spam_model.keras` and `tokenizer.pickle` files are in the same directory as `app.py`.
* Open your terminal, navigate to the project directory, and run:
    ```bash
    streamlit run app.py
    ```
* This will open the web application in your browser. Enter a message and click "Check Message" to see the prediction.

---

## Model Architecture 🧠

The model uses the following layers:

1.  **Embedding Layer:** Converts word indices into dense vectors of fixed size (64 dimensions).
2.  **Bidirectional LSTM Layer:** Processes the sequence data forwards and backward to capture context (64 units).
3.  **Dropout Layer:** Helps prevent overfitting by randomly setting a fraction (50%) of input units to 0 during training.
4.  **Dense Layer:** A standard fully connected layer (32 units, ReLU activation).
5.  **Output Layer:** A single neuron with a sigmoid activation function, outputting a probability between 0 (ham) and 1 (spam).

---

## Results 🎯

The trained model achieves an accuracy of approximately **95-97%** on the held-out test set.

---

## Future Improvements ✨

* Train on the full, imbalanced dataset using class weights.
* Experiment with different model architectures (e.g., GRUs, Attention).
* Use pre-trained word embeddings (like GloVe or Word2Vec).
* Deploy the Streamlit app to a cloud platform.
