import pandas as pd
import re
import nltk
import spacy
import contractions
from textblob import TextBlob
from transformers import pipeline
import re
import nltk
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Optional: Load paraphrase model (if using sarcasm handling)
paraphrase_pipeline = pipeline("text2text-generation", model="t5-base", tokenizer="t5-base")

# Load raw dataset
raw_data = pd.read_csv("emotion_dataset_raw.csv")  # Update with your file name

# Define a function to clean text
def clean_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = contractions.fix(text)  # Expand contractions (e.g., "don't" -> "do not")
    text = re.sub(r'\s+', ' ', text)  # Remove excessive spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers

    # Tokenize and remove stopwords
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]

    # Lemmatization using spaCy
    doc = nlp(" ".join(words))
    words = [token.lemma_ for token in doc]

    # Correct spelling using TextBlob
    corrected_text = " ".join(words)
    corrected_text = str(TextBlob(corrected_text).correct())

    return corrected_text

# Optional: Sarcasm handling with paraphrasing
def handle_sarcasm(text):
    if "sarcasm" in text.lower():  # Just a basic rule, you can enhance it
        return paraphrase_pipeline(f"paraphrase: {text}", max_length=50, do_sample=True)[0]['generated_text']
    return text

# Apply cleaning function
raw_data["clean_text"] = raw_data["text"].apply(clean_text)

# Apply sarcasm handling (Optional)
raw_data["clean_text"] = raw_data["clean_text"].apply(handle_sarcasm)

# Save cleaned dataset
raw_data.to_csv("emotion_dataset.csv", index=False)

print("Cleaning complete! Cleaned dataset saved as 'cleaned_emotion_text_dataset.csv'. 🎉")
