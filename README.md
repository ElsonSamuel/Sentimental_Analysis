# Sentiment Analysis Project

This project implements a hybrid sentiment analysis system using both VADER and machine learning approaches.

## Setup

1. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Initialize NLTK data:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('vader_lexicon')
   nltk.download('punkt')
   ```

4. Run the application:
   ```
   python app2.py
   ```

## Project Structure

- `app2.py`: Main application file
- `template/`: Frontend templates
- `models/`: ML models and vectorizers
- `data/`: Training and evaluation data
- `static/`: Static files (CSS, JS, etc.)

## Features

- Hybrid sentiment analysis (VADER + ML)
- Multi-language support
- Profanity detection
- Special character handling
- Database storage of results