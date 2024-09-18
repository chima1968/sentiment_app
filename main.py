import nltk
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import joblib
import plotly.express as px

# Set NLTK data path to a writable directory
nltk.data.path.append(os.path.join(os.path.expanduser("~"), "nltk_data"))

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

print("NLTK data downloaded successfully!")



model=joblib.load('logistic_reg.pkl')
vectorizer=joblib.load('vectorizer.pkl')


def preprocess_text(text):
    if not text:
        return ""
    
    lemmatizer = WordNetLemmatizer()
    STOPWORDS = set(['a', 'an', 'the', 'in', 'on', 'at', 'is', 'it'])  # Keep negation words
    
    # Convert to lowercase and tokenize
    tokens = word_tokenize(text.lower())
    
    # Handle negations
    negation = False
    processed_tokens = []
    for token in tokens:
        if token in ['not', 'no', 'never']:
            negation = True
            processed_tokens.append('NOT_')
        elif token in ['.', '!', '?']:
            negation = False
        elif not token.isalnum():
            negation = False
        else:
            if negation:
                processed_tokens.append('NOT_' + token)
            else:
                processed_tokens.append(token)
    
    # Lemmatize and remove stopwords
    processed_tokens = [lemmatizer.lemmatize(token) for token in processed_tokens if token not in STOPWORDS]
    
    return ' '.join(processed_tokens)

st.title("Feedback Sentiment Classifier")


feedback = st.text_area("Enter feedback:")

if st.button("Predict Sentiment"):
  
    preprocessed_feedback = preprocess_text(feedback)
  
    feedback_vectorized = vectorizer.transform([preprocessed_feedback])
    
    # Make prediction
    prediction = model.predict(feedback_vectorized)
    
    # Display result
    if prediction == 1:
        st.write("The feedback is **Positive**!")
    else:
        st.write("The feedback is **Negative**.")
        
        # Add a section for sentiment visualization
    # Add sentiment visualization
st.subheader("Sentiment Visualization")
preprocessed_feedback = preprocess_text(feedback)
  
feedback_vectorized = vectorizer.transform([preprocessed_feedback])
    
    # Make prediction
prediction = model.predict(feedback_vectorized)
    
    # Get prediction probabilities
proba = model.predict_proba(feedback_vectorized)[0]
neg_proba, pos_proba = proba[0], proba[1]
    
    # Create a bar chart for sentiment probabilities
st.bar_chart({
    'Negative': neg_proba,
    'Positive': pos_proba
})
    
    # Add confidence level
confidence = max(neg_proba, pos_proba)
st.write(f"Confidence: {confidence:.2%}")
    
    # Add feedback length analysis
st.subheader("Feedback Analysis")
word_count = len(feedback.split())
st.write(f"Word count: {word_count}")
    
    # Add a button to clear the input
if st.button("Clear Input"):
    st.experimental_rerun()
    
    # Add a section for multiple feedback analysis
st.subheader("Analyze Multiple Feedbacks")
uploaded_file = st.file_uploader("Choose a CSV file with feedbacks", type="csv")
if uploaded_file is not None:
    import pandas as pd
    df = pd.read_csv(uploaded_file)
    if st.button("Analyze Bulk Feedbacks"):
        # Check the column names
        st.write("Columns in the uploaded file:", df.columns.tolist())
        print(df.columns)
        # Assuming the column name is 'feedback'
        if 'feedback ' in df.columns:
            df['preprocessed'] = df['feedback '].apply(preprocess_text)
            df_vectorized = vectorizer.transform(df['preprocessed'])
            df['sentiment'] = model.predict(df_vectorized)
            df['sentiment'] = df['sentiment'].map({0: 'Negative', 1: 'Positive'})
            st.write(df[['feedback ', 'sentiment']])
            
            # Display sentiment distribution
            sentiment_counts = df['sentiment'].value_counts()
            fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title='Sentiment Distribution')
            st.plotly_chart(fig)
        else:
            st.error("The uploaded CSV file does not contain a 'feedback' column. Please check your file and try again.")

