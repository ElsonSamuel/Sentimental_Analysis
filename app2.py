# SSL and NLTK setup
import ssl
import nltk
from datetime import datetime
import string
from deep_translator import GoogleTranslator
from flask_sqlalchemy import SQLAlchemy
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Disable SSL verification
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('punkt')

# All other imports
import os
import re
from flask import Flask, request, jsonify, render_template
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import json
from difflib import get_close_matches

# Initialize Flask app with template folder
app = Flask(__name__, template_folder='template')

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sentiment_analysis_hybrid.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Initialize VADER sentiment analyzer and translator
sia = SentimentIntensityAnalyzer()
translator = GoogleTranslator(source='auto', target='en')

# Load the pre-trained Logistic Regression model and TF-IDF vectorizer
def load_model_and_vectorizer():
    """Try loading model and vectorizer from different possible locations and evaluate performance"""
    try:
        # First try the original files
        print("Attempting to load original model files...")
        model = joblib.load('trained_model.sav')
        vectorizer = joblib.load('vectorizer.pkl')
        print("Original model files loaded successfully!")
        
        # Evaluate model performance
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        import pandas as pd
        
        try:
            # Load your dataset
            data = pd.read_csv('combined_training_data.csv')
            X = data['text']
            y = data['target']
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Transform the data
            X_train_vectorized = vectorizer.transform(X_train)
            X_test_vectorized = vectorizer.transform(X_test)
            
            # Get predictions
            train_predictions = model.predict(X_train_vectorized)
            test_predictions = model.predict(X_test_vectorized)
            
            # Calculate accuracy scores
            train_accuracy = accuracy_score(y_train, train_predictions)
            test_accuracy = accuracy_score(y_test, test_predictions)
            
            print("\nModel Performance Metrics:")
            print(f"Training Accuracy: {train_accuracy:.4f}")
            print(f"Testing Accuracy: {test_accuracy:.4f}")
            print("\nDetailed Classification Report:")
            print(classification_report(y_test, test_predictions))
            
        except Exception as e:
            print(f"Note: Could not evaluate model performance: {str(e)}")
        
        return model, vectorizer
    except:
        try:
            # Try the models directory
            print("Attempting to load from models directory...")
            model = joblib.load('models/sentiment_model.joblib')
            vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
            print("Model files loaded from models directory successfully!")
            return model, vectorizer
        except Exception as e:
            print(f"Error loading model files: {str(e)}")
            return None, None

# Load the model and vectorizer
model, vectorizer = load_model_and_vectorizer()

if model is None or vectorizer is None:
    print("No model files found. Please ensure either:")
    print("1. trained_model.sav and vectorizer.pkl are in the root directory, or")
    print("2. sentiment_model.joblib and tfidf_vectorizer.joblib are in the models directory")

# Define profanity patterns with improved special character handling
PROFANITY_PATTERNS = [
    r'f+[^\w]*u+[^\w]*c*k+',  # Matches variations of 'fuck'
    r'f+[^\w]*c*k+',          # Matches 'fck', 'f..ck', etc.
    r'f+[^\w]*u+[^\w]*k+',    # Matches 'fuk', 'f..uk', etc.
    r'f+[^\w]*[@\*\.\s]+k',   # Matches 'f**k', 'f..k', 'f@k' etc.
    r'w+[^\w]*[o0]+[^\w]*r+[^\w]*s+[^\w]*t+',  # Matches variations of 'worst'
    r'w+[^\w]*[@\*\.\s]+s+[^\w]*t+',  # Matches 'w.st', 'w@st', etc.
]

# Define offensive word patterns
OFFENSIVE_PATTERNS = [
    # Basic offensive words
    r't+[^\w]*r+[^\w]*a+[^\w]*s+[^\w]*h+',  # Matches variations of 'trash'
    r'g+[^\w]*a+[^\w]*r+[^\w]*b+[^\w]*a+[^\w]*g+[^\w]*e+',  # Matches variations of 'garbage'
    r'w+[^\w]*o+[^\w]*r+[^\w]*t+[^\w]*h+[^\w]*l+[^\w]*e+[^\w]*s+[^\w]*s+',  # Matches variations of 'worthless'
    r'u+[^\w]*s+[^\w]*e+[^\w]*l+[^\w]*e+[^\w]*s+[^\w]*s+',  # Matches variations of 'useless'
    r's+[^\w]*t+[^\w]*u+[^\w]*p+[^\w]*i+[^\w]*d+',  # Matches variations of 'stupid'
    r'i+[^\w]*d+[^\w]*i+[^\w]*o+[^\w]*t+',  # Matches variations of 'idiot'
    r'd+[^\w]*u+[^\w]*m+[^\w]*b+',  # Matches variations of 'dumb'
    r'h+[^\w]*o+[^\w]*r+[^\w]*r+[^\w]*i+[^\w]*b+[^\w]*l+[^\w]*e+',  # Matches variations of 'horrible'
    r't+[^\w]*e+[^\w]*r+[^\w]*r+[^\w]*i+[^\w]*b+[^\w]*l+[^\w]*e+',  # Matches variations of 'terrible'
    r'a+[^\w]*w+[^\w]*f+[^\w]*u+[^\w]*l+',  # Matches variations of 'awful'
    
    # Phrases with "just"
    r'just\s+a?\s*(trash|garbage|waste|idiot|stupid|dumb|useless)',
    r'you\s+are\s+just\s+(trash|garbage|waste|idiot|stupid|dumb|useless)',
    
    # Phrases with "like"
    r'like\s+a?\s*(trash|garbage|waste|idiot|stupid|dumb|useless)',
    
    # Direct insults
    r'you\s+(are|look|seem|sound)\s*(trash|garbage|waste|idiot|stupid|dumb|useless)',
    
    # Negative qualities
    r'w+[^\w]*o+[^\w]*r+[^\w]*t+[^\w]*h+[^\w]*l+[^\w]*e+[^\w]*s+[^\w]*s+',
    r'p+[^\w]*a+[^\w]*t+[^\w]*h+[^\w]*e+[^\w]*t+[^\w]*i+[^\w]*c+',
    r'r+[^\w]*i+[^\w]*d+[^\w]*i+[^\w]*c+[^\w]*u+[^\w]*l+[^\w]*o+[^\w]*u+[^\w]*s+',
    
    # Negative actions
    r'h+[^\w]*a+[^\w]*t+[^\w]*e+',
    r's+[^\w]*u+[^\w]*c+[^\w]*k+[^\w]*s*',
    
    # Common variations
    r'tr[@\*\.\s]+sh',  # t.r.a.s.h, tr@sh, etc.
    r'g[@\*\.\s]+rb[@\*\.\s]+ge',  # g@rb@ge, etc.
    r'st[@\*\.\s]+p[@\*\.\s]+d',  # st@p@d, etc.
]

# Define animal insult patterns
ANIMAL_INSULT_PATTERNS = [
    r'similar\s+to\s+a\s+(pig|donkey|ass|horse|dog|monkey|rat|snake|worm)',
    r'like\s+a\s+(pig|donkey|ass|horse|dog|monkey|rat|snake|worm)',
    r'you\s+are\s+a\s+(pig|donkey|ass|horse|dog|monkey|rat|snake|worm)',
]

# Database Model
class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    vader_sentiment = db.Column(db.String(20), nullable=False)
    ml_sentiment = db.Column(db.String(20), nullable=False)
    final_sentiment = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float)
    language = db.Column(db.String(10))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'text': self.text,
            'vader_sentiment': self.vader_sentiment,
            'ml_sentiment': self.ml_sentiment,
            'final_sentiment': self.final_sentiment,
            'confidence': self.confidence,
            'language': self.language,
            'timestamp': self.timestamp.isoformat()
        }

# Define global word sets at the module level
POSITIVE_WORDS = {
    'good', 'great', 'excellent', 'awesome', 'amazing', 'wonderful', 'perfect', 'best',
    'nice', 'fine', 'beautiful', 'brilliant', 'fantastic', 'outstanding', 'incredible',
    'phenomenal', 'spectacular', 'superb', 'terrific', 'love', 'lovely', 'enjoyed',
    'recommend', 'recommended', 'worth', 'favorite', 'favourite', 'masterpiece'
}

NEGATIVE_WORDS = {
    # Profanity
    'fuck', 'shit', 'crap', 'damn', 'hell', 'ass', 'bitch', 'dick', 'piss', 'cunt', 
    'bastard', 'whore', 'slut', 'twat', 'prick', 'asshole', 'dammit', 'fucking', 'shitty',
    
    # Insults and offensive terms
    'trash', 'garbage', 'worthless', 'useless', 'stupid', 'idiot', 'dumb', 'retard', 
    'moron', 'fool', 'jerk', 'loser', 'scum', 'waste', 'pathetic', 'ridiculous', 'nonsense',
    
    # Negative descriptors
    'bad', 'terrible', 'horrible', 'awful', 'poor', 'worst', 'disgusting', 'revolting',
    'repulsive', 'vile', 'nasty', 'ugly', 'hideous', 'repugnant', 'abhorrent', 'atrocious',
    
    # Negative actions
    'hate', 'loathe', 'despise', 'abhor', 'detest', 'abominate', 'execrate', 'revile',
    
    # Negative emotions
    'angry', 'furious', 'enraged', 'irate', 'outraged', 'frustrated', 'annoyed', 'irritated',
    
    # Negative qualities
    'incompetent', 'inadequate', 'inept', 'inefficient', 'ineffective', 'inferior', 'substandard',
    
    # Negative outcomes
    'failure', 'disaster', 'catastrophe', 'fiasco', 'debacle', 'calamity', 'mishap', 'misfortune',
    
    # Negative comparisons
    'worse', 'worst', 'inferior', 'subpar', 'substandard', 'below average', 'below par'
}

# Define common obfuscation patterns
OBFUSCATION_PATTERNS = {
    # Profanity patterns
    r'f[\W_]*u[\W_]*c[\W_]*k': 'fuck',
    r'f[\W_]*[@*.\s]*k': 'fuck',
    r'f[\W_]*c[\W_]*k': 'fuck',
    r'sh[\W_]*[i!][\W_]*t': 'shit',
    r'b[\W_]*i[\W_]*t[\W_]*ch': 'bitch',
    r'a[\W_]*s[\W_]*s': 'ass',
    
    # Negative word patterns
    r'wo[\W_]*[@*.\s]*st': 'worst',
    r'w[\W_]*[@*.\s]*st': 'worst',
    r'b[\W_]*a[\W_]*d': 'bad',
    r'tr[\W_]*[@*.\s]*sh': 'trash',
    r'cr[\W_]*[@*.\s]*p': 'crap',
    r'h[\W_]*[@*.\s]*te': 'hate',
    r'st[\W_]*[@*.\s]*p[\W_]*d': 'stupid'
}

def deobfuscate(text):
    """
    Preprocess text to normalize obfuscated words before sentiment analysis
    """
    text_lower = text.lower()
    
    # Apply each pattern replacement
    for pattern, replacement in OBFUSCATION_PATTERNS.items():
        text_lower = re.sub(pattern, replacement, text_lower, flags=re.IGNORECASE)
    
    return text_lower

def preprocess_text(text):
    """Preprocess text for ML model"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep spaces between words
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def contains_positive_sentiment(text):
    """Check if text contains positive sentiment markers"""
    text_lower = text.lower()
    words = text_lower.split()
    
    # Special case for "not bad" - this is a positive sentiment
    if text_lower == "not bad" or text_lower == "not bad!" or text_lower == "not bad.":
        return True
    
    # Check for negations first
    negation_words = {'not', 'no', 'never', 'none', 'nobody', 'nowhere', 'neither', 'nor', 'cannot', "can't", "don't", "doesn't", "didn't", "won't", "wouldn't", "shouldn't", "couldn't", "mightn't", "mustn't", "shan't"}
    
    # Special case: "not bad" is positive
    if len(words) == 2 and words[0] in negation_words and words[1] == 'bad':
        return True
    
    # Check if the text contains a negation followed by a positive word
    for i in range(len(words) - 1):
        if words[i] in negation_words and words[i+1] in POSITIVE_WORDS:  # Use global POSITIVE_WORDS
            # Special case: "not bad" is positive
            if words[i+1] == 'bad':
                return True
            return False
    
    # Check individual words
    for word in words:
        if word in POSITIVE_WORDS:  # Use global POSITIVE_WORDS
            return True
    
    # Positive phrases
    positive_phrases = [
        'beautiful film', 'great film', 'good film', 'nice film',
        'worth watching', 'must watch', 'must see',
        'highly recommend', 'really good', 'very good',
        'enjoyed it', 'loved it', 'recommend it',
        'best movie', 'best film', 'favorite film', 'favourite film'
    ]
    
    # Check phrases
    for phrase in positive_phrases:
        if phrase in text_lower:
            return True
            
    return False

def contains_negative_sentiment(text):
    """Check if text contains negative sentiment markers"""
    text_lower = text.lower()
    words = text_lower.split()
    
    # Check individual words against global NEGATIVE_WORDS
    for word in words:
        if word in NEGATIVE_WORDS:
            return True
    
    # Check for negations followed by positive words
    negation_words = {'not', 'no', 'never', 'none', 'nobody', 'nowhere', 'neither', 'nor', 'cannot', "can't", "don't", "doesn't", "didn't", "won't", "wouldn't", "shouldn't", "couldn't", "mightn't", "mustn't", "shan't"}
    
    for i in range(len(words) - 1):
        if words[i] in negation_words and words[i+1] in POSITIVE_WORDS:
            # Special case: "not bad" is positive
            if words[i+1] != 'bad':
                return True
    
    # Negative phrases
    negative_phrases = [
        'not good', 'not great', 'not excellent', 'not nice', 'not fine',
        'not worth', 'not recommended', 'not worth watching',
        'do not recommend', 'would not recommend', 'cannot recommend',
        'do not watch', 'would not watch', 'cannot watch',
        'do not see', 'would not see', 'cannot see',
        'do not like', 'would not like', 'cannot like',
        'do not enjoy', 'would not enjoy', 'cannot enjoy',
        'do not love', 'would not love', 'cannot love'
    ]
    
    # Check phrases
    for phrase in negative_phrases:
        if phrase in text_lower:
            return True
            
    return False

# Add Tamil negative words/phrases with their variations
TAMIL_NEGATIVE_WORDS = {
    'mokka': True,      # மொக்க
    'waste': True,      # வேஸ்ட்
    'kodumai': True,    # கொடுமை
    'boring': True,     # போரிங்
    'worst': True,      # வோர்ஸ்ட்
    'mosam': True,      # மோசம்
    'garbage': True,    # கார்பேஜ்
    'cheap': True,      # சீப்
}

def is_tamil_negative(text):
    """Check if the Tamil text contains negative sentiment markers"""
    text_lower = text.lower()
    words = text_lower.split()
    
    # Check for common negative word combinations
    negative_combinations = [
        ('mokka', 'padam'),
        ('waste', 'padam'),
        ('kodumai', 'padam'),
        ('boring', 'padam'),
        ('worst', 'padam'),
        ('mosam', 'padam'),
    ]
    
    # Check for word combinations
    for word1, word2 in negative_combinations:
        if word1 in text_lower and word2 in text_lower:
            return True
    
    # Count negative words
    negative_count = sum(1 for word in words if word in TAMIL_NEGATIVE_WORDS)
    return negative_count > 0

def get_ml_sentiment(text):
    """Get sentiment using the pre-trained Logistic Regression model"""
    if model is None or vectorizer is None:
        return None, 0.0
    
    text_lower = text.lower()
    
    # Check for Tamil negative sentiment first
    if is_tamil_negative(text_lower):
        return "Negative 😞", 1.0
        
    # Special handling for Tamil positive phrases
    if any(phrase in text_lower for phrase in ['sema padam', 'semma padam', 'semaya irunthuthu']):
        return "Positive 😊", 1.0
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Transform text using the same vectorizer used during training
    text_vectorized = vectorizer.transform([processed_text])
    
    # Get prediction and probability
    prediction = model.predict(text_vectorized)[0]
    probabilities = model.predict_proba(text_vectorized)[0]
    confidence = max(probabilities)
    
    # For Tamil text, adjust sentiment based on known words
    if any(word in text_lower for word in TAMIL_NEGATIVE_WORDS):
        prediction = 0  # Force negative for known negative words
        confidence = max(confidence, 0.8)
    elif any(word in text_lower for word in TAMIL_POSITIVE_WORDS):
        if confidence > 0.4:  # Lower threshold for Tamil positive words
            prediction = 1
            confidence = max(confidence, 0.8)
    
    # Convert numeric prediction to sentiment label
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    
    return sentiment, confidence

def contains_profanity(text):
    """Check if text contains profanity using regex patterns"""
    text = text.lower()
    
    # First check for exact matches of profanity patterns
    for pattern in PROFANITY_PATTERNS:
        if re.search(pattern, text):
            return True
    
    # Then check for profanity with special characters between letters
    profanity_words = ['fuck', 'shit', 'crap', 'damn', 'hell', 'ass', 'bitch', 'dick', 'piss', 'cunt', 'worst']
    
    # Create patterns that match words with any special characters between letters
    for word in profanity_words:
        # Pattern that matches the word with any non-letter characters between letters
        pattern = ''.join([f"{c}[^a-zA-Z]*" for c in word])
        if re.search(pattern, text):
            return True
        
        # Additional pattern for special character substitutions
        pattern = ''.join([f"{c}[^a-zA-Z0-9]*" for c in word])
        if re.search(pattern, text):
            return True
        
        # Check for cleaned version of the word
        cleaned_text = ''.join(c for c in text if c.isalnum())
        if word in cleaned_text:
            return True
    
    return False

def contains_offensive(text):
    """Check if text contains offensive language using regex patterns"""
    text = text.lower()
    for pattern in OFFENSIVE_PATTERNS:
        if re.search(pattern, text):
            return True
    return False

def contains_animal_insult(text):
    """Check if text contains animal-based insults"""
    text = text.lower()
    for pattern in ANIMAL_INSULT_PATTERNS:
        if re.search(pattern, text):
            return True
    return False

def normalize_text(text):
    """
    Normalize text by:
    1. Removing special characters and dots between letters
    2. Finding closest matching real words
    """
    # Step 1: Check for special patterns first
    text_lower = text.lower()
    
    # Check for profanity with special characters
    for pattern in PROFANITY_PATTERNS:
        if re.search(pattern, text_lower):
            return text  # Return as is to preserve the profanity for detection
    
    # Check for words with dots or special characters that might be trying to evade detection
    words = text_lower.split()
    normalized_words = []
    
    for word in words:
        # Remove all special characters between letters
        cleaned_word = ''.join(c for c in word if c.isalnum())
        
        # Check if the cleaned word matches any profanity or negative words
        if cleaned_word in NEGATIVE_WORDS or any(re.search(pattern, cleaned_word) for pattern in PROFANITY_PATTERNS):
            return text  # Return as is to preserve the negative/profane content
        
        # If it's not profanity or negative, normalize it
        if cleaned_word in NEUTRAL_WORDS:
            normalized_words.append(cleaned_word)
        else:
            normalized_words.append(word)
    
    return ' '.join(normalized_words)

def get_hybrid_analysis(text):
    """Combine VADER and ML model predictions with improved positive sentiment detection"""
    # First deobfuscate the text
    deobfuscated_text = deobfuscate(text)
    text_lower = deobfuscated_text.lower()
    
    # First check for profanity or offensive content with special characters
    if contains_profanity(text_lower):
        return {
            'final_sentiment': "Negative 😞",
            'vader_sentiment': "Negative 😞",
            'ml_sentiment': "Negative 😞",
            'confidence': 1.0,
            'details': {
                'vader_scores': {'pos': 0.0, 'neg': 1.0, 'neu': 0.0, 'compound': -1.0},
                'ml_confidence': 1.0,
                'reason': 'profanity_detected'
            }
        }
    
    # Then normalize the text for further analysis
    normalized_text = normalize_text(text_lower)
    
    # Special case for "not bad" - this is a positive sentiment
    if text_lower == "not bad" or text_lower == "not bad!" or text_lower == "not bad.":
        return {
            'final_sentiment': "Positive 😊",
            'vader_sentiment': "Positive 😊",
            'ml_sentiment': "Positive 😊",
            'confidence': 1.0,
            'details': {
                'vader_scores': {'pos': 1.0, 'neg': 0.0, 'neu': 0.0, 'compound': 1.0},
                'ml_confidence': 1.0,
                'reason': 'special_case_not_bad'
            }
        }
    
    # Check for words with dots that might be neutral
    words_with_dots = [word for word in text_lower.split() if '.' in word]
    for word in words_with_dots:
        # Remove dots and check if it's a neutral word
        cleaned_word = word.replace('.', '')
        if cleaned_word in NEUTRAL_WORDS:
            return {
                'final_sentiment': "Neutral 😐",
                'vader_sentiment': "Neutral 😐",
                'ml_sentiment': "Neutral 😐",
                'confidence': 1.0,
                'details': {
                    'vader_scores': {'pos': 0.0, 'neg': 0.0, 'neu': 1.0, 'compound': 0.0},
                    'ml_confidence': 1.0,
                    'reason': 'neutral_word_with_dots'
                }
            }
    
    # Check for profanity with dots
    for word in words_with_dots:
        cleaned_word = word.replace('.', '')
        if cleaned_word in ['fuck', 'shit', 'crap', 'damn', 'hell', 'ass', 'bitch', 'dick', 'piss', 'cunt']:
            return {
                'final_sentiment': "Negative 😞",
                'vader_sentiment': "Negative 😞",
                'ml_sentiment': "Negative 😞",
                'confidence': 1.0,
                'details': {
                    'vader_scores': {'pos': 0.0, 'neg': 1.0, 'neu': 0.0, 'compound': -1.0},
                    'ml_confidence': 1.0,
                    'reason': 'profanity_detected'
                }
            }
    
    # Movie-specific positive phrases that should always be positive
    movie_positive_phrases = [
        'beautiful film', 'great film', 'good film', 'nice film',
        'beautiful movie', 'great movie', 'good movie', 'nice movie',
        'worth watching', 'must watch', 'must see',
        'enjoyed watching', 'loved watching', 'amazing film',
        'best movie', 'best film', 'favorite film', 'favourite film'
    ]
    
    # Check for movie-specific positive phrases
    for phrase in movie_positive_phrases:
        if phrase in text_lower:
            return {
                'final_sentiment': "Positive 😊",
                'vader_sentiment': "Positive 😊",
                'ml_sentiment': "Positive 😊",
                'confidence': 1.0,
                'details': {
                    'vader_scores': {'pos': 1.0, 'neg': 0.0, 'neu': 0.0, 'compound': 1.0},
                    'ml_confidence': 1.0,
                    'reason': 'movie_positive_phrase'
                }
            }
    
    # Check for explicit negative sentiment
    if contains_negative_sentiment(text):
        return {
            'final_sentiment': "Negative 😞",
            'vader_sentiment': "Negative 😞",
            'ml_sentiment': "Negative 😞",
            'confidence': 1.0,
            'details': {
                'vader_scores': {'pos': 0.0, 'neg': 1.0, 'neu': 0.0, 'compound': -1.0},
                'ml_confidence': 1.0,
                'reason': 'explicit_negative'
            }
        }
    
    # Check for general positive sentiment
    if contains_positive_sentiment(text):
        return {
            'final_sentiment': "Positive 😊",
            'vader_sentiment': "Positive 😊",
            'ml_sentiment': "Positive 😊",
            'confidence': 1.0,
            'details': {
                'vader_scores': {'pos': 1.0, 'neg': 0.0, 'neu': 0.0, 'compound': 1.0},
                'ml_confidence': 1.0,
                'reason': 'explicit_positive'
            }
        }
    
    # Get VADER sentiment scores
    vader_scores = sia.polarity_scores(normalized_text)
    
    # Get ML model prediction
    ml_sentiment, ml_confidence = get_ml_sentiment(normalized_text)
    
    # Check if the word is in our neutral words list
    if text_lower.strip() in NEUTRAL_WORDS:
        return {
            'final_sentiment': "Neutral 😐",
            'vader_sentiment': "Neutral 😐",
            'ml_sentiment': "Neutral 😐",
            'confidence': 1.0,
            'details': {
                'vader_scores': {'pos': 0.0, 'neg': 0.0, 'neu': 1.0, 'compound': 0.0},
                'ml_confidence': 1.0,
                'reason': 'neutral_word'
            }
        }
    
    # Combine predictions with adjusted weights
    vader_weight = 0.4
    ml_weight = 0.6
    
    # Calculate weighted sentiment with adjusted thresholds
    vader_value = 1 if vader_scores['compound'] >= 0 else -1  # More lenient threshold for positive
    ml_value = 1 if ml_sentiment == "Positive 😊" else -1
    
    # If either model is very confident about positive sentiment, trust it
    if vader_scores['pos'] > 0.5 or ml_confidence >= 0.8:
        weighted_score = 1.0
    else:
        weighted_score = (vader_value * vader_weight) + (ml_value * ml_weight)
    
    # Check if the sentiment is close to neutral
    if abs(weighted_score) < 0.2:
        return {
            'final_sentiment': "Neutral 😐",
            'vader_sentiment': "Neutral 😐",
            'ml_sentiment': "Neutral 😐",
            'confidence': abs(weighted_score),
            'details': {
                'vader_scores': vader_scores,
                'ml_confidence': ml_confidence,
                'reason': 'neutral_sentiment'
            }
        }
    
    return {
        'final_sentiment': "Positive 😊" if weighted_score > 0 else "Negative 😞",
        'vader_sentiment': "Positive 😊" if vader_scores['compound'] >= 0 else "Negative 😞",
        'ml_sentiment': ml_sentiment,
        'confidence': abs(weighted_score),
        'details': {
            'vader_scores': vader_scores,
            'ml_confidence': ml_confidence,
            'reason': 'hybrid_analysis'
        }
    }

# Routes
@app.route('/')
def index():
    return render_template('index.html')

# Define the set of neutral words at the module level
NEUTRAL_WORDS = {
    # Weather terms
    'frost', 'cloud', 'rain', 'snow', 'wind', 'storm', 'weather', 'climate', 
    'temperature', 'cold', 'hot', 'warm', 'cool', 'mild', 'severe', 'extreme',
    
    # Time terms
    'time', 'day', 'night', 'morning', 'evening', 'afternoon', 'week', 'month', 'year',
    
    # Location terms
    'place', 'location', 'area', 'region', 'city', 'town', 'country', 'world',
    
    # Object terms
    'object', 'thing', 'item', 'stuff', 'material', 'substance', 'element',
    
    # Action terms
    'action', 'activity', 'process', 'procedure', 'method', 'technique', 'approach',
    
    # Descriptive terms
    'normal', 'regular', 'standard', 'typical', 'usual', 'common', 'ordinary',
    
    # Quantity terms
    'some', 'many', 'few', 'several', 'multiple', 'various', 'different',
    
    # Other neutral terms
    'data', 'information', 'fact', 'truth', 'reality', 'existence', 'being', 'state',
}

# Add Tamil positive words/phrases with their variations
TAMIL_POSITIVE_WORDS = {
    'sema': True,      # செம
    'semma': True,     # செம்ம
    'semaya': True,    # செமய
    'super': True,     # சூப்பர்
    'nalla': True,     # நல்ல
    'mass': True,      # மாஸ்
    'thara': True,     # தர
    'vera': True,      # வேற
    'level': True,     # லெவல்
    'padam': True,     # படம்
    'gethu': True,     # கெத்து
    'adipoli': True,   # அடிபோலி
    'kalakkal': True,  # கலக்கல்
}

def is_tamil_positive(text):
    """Check if the Tamil text contains positive sentiment markers"""
    text_lower = text.lower()
    words = text_lower.split()
    
    # Check for common positive word combinations
    positive_combinations = [
        ('sema', 'padam'),
        ('semma', 'padam'),
        ('semaya', 'irunthuthu'),
        ('vera', 'level'),
        ('super', 'padam'),
        ('nalla', 'padam'),
    ]
    
    # Check for word combinations
    for word1, word2 in positive_combinations:
        if word1 in text_lower and word2 in text_lower:
            return True
    
    # Check for variations of "sema/semma" followed by any word
    for i in range(len(words) - 1):
        if words[i] in ['sema', 'semma', 'semaya'] and words[i + 1] in ['padam', 'irunthuthu', 'iruku']:
            return True
    
    # Count positive words
    positive_count = sum(1 for word in words if word in TAMIL_POSITIVE_WORDS)
    
    # Return True if 2 or more positive words are found
    return positive_count >= 2

def analyze_multilingual(text, source_lang='auto'):
    """Handle multiple languages with improved translation handling"""
    try:
        # First deobfuscate the text
        deobfuscated_text = deobfuscate(text)
        text_lower = deobfuscated_text.lower()
        
        # Initialize translator
        translator = GoogleTranslator(source='auto', target='en')
        
        try:
            # Get translation regardless of language for display purposes
            try:
                translated_text = translator.translate(deobfuscated_text)
            except Exception as e:
                print(f"Translation error: {str(e)}")
                translated_text = None
            
            # Check for Tamil negative patterns first
            if any(word in text_lower.split() for word in TAMIL_NEGATIVE_WORDS.keys()):
                return {
                    'original_text': text,
                    'language': 'ta',
                    'translated_text': translated_text,  # Include translation
                    'analysis': {
                        'final_sentiment': "Negative 😞",
                        'vader_sentiment': "Negative 😞",
                        'ml_sentiment': "Negative 😞",
                        'confidence': 1.0,
                        'details': {
                            'vader_scores': None,
                            'ml_confidence': 1.0,
                            'reason': 'tamil_negative_word'
                        }
                    }
                }
            
            # Check specifically for "mokka padam" pattern
            if 'mokka' in text_lower and 'padam' in text_lower:
                return {
                    'original_text': text,
                    'language': 'ta',
                    'translated_text': translated_text,  # Include translation
                    'analysis': {
                        'final_sentiment': "Negative 😞",
                        'vader_sentiment': "Negative 😞",
                        'ml_sentiment': "Negative 😞",
                        'confidence': 1.0,
                        'details': {
                            'vader_scores': None,
                            'ml_confidence': 1.0,
                            'reason': 'mokka_padam_pattern'
                        }
                    }
                }
            
            # Detect language
            if source_lang == 'auto':
                source_lang = translator.detect(deobfuscated_text)
            
            # Special handling for Tamil text
            if source_lang == 'ta':
                # Check for Tamil positive sentiment
                if is_tamil_positive(deobfuscated_text):
                    return {
                        'original_text': text,
                        'language': source_lang,
                        'translated_text': translated_text,  # Include translation
                        'analysis': {
                            'final_sentiment': "Positive 😊",
                            'vader_sentiment': "Positive 😊",
                            'ml_sentiment': "Positive 😊",
                            'confidence': 1.0,
                            'details': {
                                'vader_scores': None,
                                'ml_confidence': 1.0,
                                'reason': 'tamil_positive_phrase'
                            }
                        }
                    }
                
                # For Tamil text, use ML model directly
                ml_sentiment, ml_confidence = get_ml_sentiment(deobfuscated_text)
                
                return {
                    'original_text': text,
                    'language': source_lang,
                    'translated_text': translated_text,  # Include translation
                    'analysis': {
                        'final_sentiment': ml_sentiment,
                        'vader_sentiment': ml_sentiment,
                        'ml_sentiment': ml_sentiment,
                        'confidence': ml_confidence,
                        'details': {
                            'vader_scores': None,
                            'ml_confidence': ml_confidence,
                            'reason': 'tamil_ml_analysis'
                        }
                    }
                }
            
            # For non-Tamil languages, proceed with regular analysis
            if source_lang != 'en':
                text_to_analyze = translated_text if translated_text else deobfuscated_text
            else:
                text_to_analyze = deobfuscated_text
            
            # Get the analysis using the hybrid approach for non-Tamil text
            analysis = get_hybrid_analysis(text_to_analyze)
            
            return {
                'original_text': text,
                'language': source_lang,
                'translated_text': translated_text if source_lang != 'en' else None,
                'analysis': analysis
            }
            
        except Exception as e:
            print(f"Error in translation: {str(e)}")
            # Fallback to analyzing original text
            analysis = get_hybrid_analysis(deobfuscated_text)
            return {
                'original_text': text,
                'language': 'en',
                'translated_text': None,
                'analysis': analysis
            }
    except Exception as e:
        print(f"Error in analyze_multilingual: {str(e)}")
        return {'error': str(e)}

@app.route('/process_comment', methods=['POST'])
def process_comment():
    """Process a single comment and return sentiment analysis"""
    try:
        data = request.json
        comment = data.get('comment', '')
        
        if not comment:
            return jsonify({'error': 'Comment is required'}), 400
            
        # Get multilingual analysis with hybrid sentiment
        analysis_result = analyze_multilingual(comment)
        
        # Check if there was an error in analysis
        if 'error' in analysis_result:
            return jsonify({'error': analysis_result['error']}), 500
            
        # Save to database
        try:
            analysis_record = Analysis(
                text=comment,
                vader_sentiment=analysis_result['analysis']['vader_sentiment'],
                ml_sentiment=analysis_result['analysis']['ml_sentiment'],
                final_sentiment=analysis_result['analysis']['final_sentiment'],
                confidence=analysis_result['analysis']['confidence'],
                language=analysis_result['language']
            )
            db.session.add(analysis_record)
            db.session.commit()
        except Exception as db_error:
            print(f"Database error: {str(db_error)}")
            # Continue even if database save fails
        
        # Return the complete response including the comment and translation
        return jsonify({
            'comment': comment,
            'translated_text': analysis_result.get('translated_text'),  # Include translation in response
            'sentiment': analysis_result['analysis']['final_sentiment'],
            'details': {
                'vader_sentiment': analysis_result['analysis']['vader_sentiment'],
                'ml_sentiment': analysis_result['analysis']['ml_sentiment'],
                'confidence': analysis_result['analysis']['confidence']
            }
        })
        
    except Exception as e:
        print(f"Error in process_comment: {str(e)}")
        return jsonify({'error': 'An error occurred while processing your comment'}), 500

# Create the database tables
def init_db():
    with app.app_context():
        db.create_all()

# Run the Flask app
if __name__ == '__main__':
    init_db()  # Initialize database
    app.run(debug=True, port=5004)  # Using port 5004 to avoid conflict with other apps 