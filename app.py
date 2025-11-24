import string
import pickle
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# NLTK Downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download("punkt_tab")


ps = PorterStemmer()
tfidf = pickle.load(open('./artifacts/tfidf_vectorizer.pkl', 'rb'))
model = pickle.load(open('./artifacts/final_spam_classifier.pkl', 'rb'))

st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="centered")
st.title("📧 Email/SMS Spam Classifier")
st.write("Enter a message below to check if it is Spam or Not Spam.")

input_sms = st.text_area("Enter your message here", height=150)

# Text Preprocessing
def transform_text(text):
    """
    Function to preprocess text:
    - Lowercase conversion
    - Tokenization
    - Remove non-alphanumeric words
    - Remove stopwords & punctuation
    - Stemming
    """
    text = text.lower()
    words = nltk.word_tokenize(text)

    words = [word for word in words if word.isalnum()]

    words = [word for word in words if word not in stopwords.words('english') and word not in string.punctuation]

    words = [ps.stem(word) for word in words]

    return " ".join(words)

# Prediction
def predict_spam(message):
    transformed_sms = transform_text(message)
    vector_input = tfidf.transform([transformed_sms])
    prediction = model.predict(vector_input)[0]
    return prediction

# Button action
if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        result = predict_spam(input_sms)
        if result == 1:
            st.error("🚨 Prediction: Spam")
        else:
            st.success("✅ Prediction: Not Spam")

