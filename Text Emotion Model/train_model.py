import pandas as pd
import numpy as np
import joblib
import re
import nltk
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLP resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load Dataset
df = pd.read_csv("project/text/new/emotion_dataset.csv")  # Replace with your actual dataset path

# Define Text Cleaning Function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    cleaned_text = " ".join([lemmatizer.lemmatize(word) for word in words if word not in stop_words])
    return cleaned_text

# Apply Cleaning Function
df['Clean_Text'] = df['Text'].apply(clean_text)  # Assuming column "text" contains raw data

# Split Data
X_train, X_test, y_train, y_test = train_test_split(df['Clean_text'], df['Emotion'], test_size=0.2, random_state=42)

# Create a Pipeline for Training
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),  # Convert text into numerical vectors
    ('classifier', LogisticRegression(max_iter=500))  # Train a Logistic Regression model
])

# Train the Model
pipeline.fit(X_train, y_train)

# Evaluate Model Performance
accuracy = pipeline.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save Model Pipeline
joblib.dump(pipeline, "project/text/new/emotion_classifier_pipe_lrr.pkl")

print("Model training complete. Pipeline saved as 'emotion_classifier_pipe_lr.pkl'.")
