import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from textblob import TextBlob
import re

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

def analyze_sentiment(text):
    # Preprocess the text
    text = preprocess_text(text)
    
    # Create a TextBlob object
    blob = TextBlob(text)
    
    # Get the sentiment polarity (-1 to 1)
    polarity = blob.sentiment.polarity
    
    # Classify sentiment based on polarity
    if polarity > 0.1:
        return "Positive", polarity
    elif polarity < -0.1:
        return "Negative", polarity
    else:
        return "Neutral", polarity

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english')) - {'not', 'no', 'nor'}
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Handle negations
    negation = False
    processed_tokens = []
    for token in tokens:
        if token in ['not', 'no', 'never']:
            negation = True
            continue
        if negation:
            token = 'not_' + token
            negation = False
        processed_tokens.append(token)
    
    return ' '.join(processed_tokens)

# Example usage in your Streamlit app
import streamlit as st

st.title("Feedback Sentiment Analyzer")

feedback = st.text_area("Enter your feedback:")

if st.button("Analyze Sentiment"):
    if feedback:
        sentiment, polarity = analyze_sentiment(feedback)
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Polarity: {polarity:.2f}")
        
        # Visualize the sentiment
        st.progress((polarity + 1) / 2)  # Convert -1 to 1 range to 0 to 1
        
        if sentiment == "Positive":
            st.success("This feedback is positive!")
        elif sentiment == "Negative":
            st.error("This feedback is negative.")
        else:
            st.info("This feedback is neutral.")
    else:
        st.warning("Please enter some feedback to analyze.")

# Add a section for multiple feedback analysis
st.subheader("Analyze Multiple Feedbacks")
uploaded_file = st.file_uploader("Choose a CSV file with feedbacks", type="csv")
if uploaded_file is not None:
    import pandas as pd
    df = pd.read_csv(uploaded_file)
    if st.button("Analyze Bulk Feedbacks"):
        # Check the column names
        st.write("Columns in the uploaded file:", df.columns.tolist())
        
        # Assuming the column name is 'feedback'
        if 'feedback' in df.columns:
            df['sentiment'], df['polarity'] = zip(*df['feedback'].apply(analyze_sentiment))
            st.write(df[['feedback', 'sentiment', 'polarity']])
            
            # Display sentiment distribution
            sentiment_counts = df['sentiment'].value_counts()
            st.bar_chart(sentiment_counts)
        else:
            st.error("The uploaded CSV file does not contain a 'feedback' column. Please check your file and try again.")