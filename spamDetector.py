import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize NLTK resources
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)

download_nltk_resources()

# Load model and vectorizer
model = pickle.load(open('spam.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stopwords.words('english')])
    return text

def classify_email(email):
    cleaned_email = preprocess_text(email)
    processed_input = vectorizer.transform([cleaned_email]).toarray()
    prediction = model.predict(processed_input)[0]
    confidence = max(model.predict_proba(processed_input)[0]) * 100
    return prediction, confidence

def main():
    st.title("AI-Powered Spam Classification")
    st.markdown("""
    **Features**:
    - Real-time spam classification
    - Confidence levels
    - Advanced NLP backend
    """)

    user_input = st.text_area("Enter email content to classify", height=150)
    
    if st.button("Classify"):
        if user_input.strip():
            prediction, confidence = classify_email(user_input)
            if prediction == 1:
                st.error(f"🚨 This is classified as **SPAM** with a confidence of {confidence:.2f}%.")
            else:
                st.success(f"✅ This is classified as **NOT SPAM** with a confidence of {confidence:.2f}%.")
        else:
            st.warning("Please enter email content to classify.")
    
    st.sidebar.title("Example Emails")
    if st.sidebar.button("Show Examples"):
        st.sidebar.markdown("""
        **Ham Example**:
        - Hi there, are we still on for the meeting tomorrow?

        **Spam Example**:
        - Congratulations! You have won $1,000,000! Click here to claim.
        """)

if __name__ == "__main__":
    main()
