import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# --- Load the Model and Tokenizer ---

# Load the trained model
model = tf.keras.models.load_model('spam_model.keras')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# --- Preprocessing Function (same as in your notebook) ---
max_len = 100  # Use the same max_len as you did in training

def predict_spam(message):
    sequence = tokenizer.texts_to_sequences([message])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(padded_sequence)
    return prediction[0][0]

# --- Build the Streamlit Web App ---

st.set_page_config(page_title="Spam Detector", page_icon="📧")
st.title("📧 Spam Detection App")
st.write("Enter a message below to check if it's spam or ham (not spam).")

# Create a text input box
user_input = st.text_area("Your Message:")

# Create a button to run the prediction
if st.button("Check Message", use_container_width=True):
    if user_input:
        # Make a prediction
        prediction_score = predict_spam(user_input)
        
        # Display the result
        if prediction_score > 0.5:
            st.error(f"This looks like SPAM! (Score: {prediction_score:.2f})", icon="🚫")
        else:
            st.success(f"This looks like HAM! (Score: {prediction_score:.2f})", icon="✅")
    else:
        st.warning("Please enter a message to check.", icon="⚠️")