Here's a complete Python implementation for an AI model to classify SMS messages as spam or legitimate. It uses TF-IDF for text representation and applies classifiers like Naive Bayes, Logistic Regression, and Support Vector Machines (SVM).

Let’s go step by step:

Load the dataset
Preprocess the text
Convert text to numerical features using TF-IDF
Train different classifiers
Evaluate the models
Full Implementation:
python
Copy
Edit
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset (Replace with actual dataset)
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms-spam-collection/SMSSpamCollection"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# Convert labels to binary values (spam=1, ham=0)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()  # Remove leading/trailing spaces
    return text

# Apply preprocessing
df['message'] = df['message'].apply(preprocess_text)

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train classifiers
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(kernel='linear')
}

# Evaluate models
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Example of predicting a new message
new_message = ["Congratulations! You've won a free iPhone. Click here to claim."]
new_message_tfidf = vectorizer.transform(new_message)

for name, model in models.items():
    prediction = model.predict(new_message_tfidf)
    print(f"{name} Prediction:", "Spam" if prediction[0] == 1 else "Legit")
Explanation:
Load Dataset: Reads SMS data containing spam and non-spam (ham) messages.
Preprocessing:
Convert text to lowercase.
Remove numbers and punctuation.
Strip extra spaces.
Feature Extraction:
Use TF-IDF to transform text into numerical vectors.
Model Training:
Train Naive Bayes, Logistic Regression, and SVM classifiers.
Evaluation:
Check accuracy, classification report, and confusion matrix.
Prediction:
Test the model on a new message.
Expected Output:
Accuracy Scores for each model.
Precision, Recall, F1-score from the classification report.
Confusion Matrix showing True Positives, False Positives, etc.
Spam Detection Example: Model predicts if a sample message is Spam or Legit.
Further Improvements:
Use Word Embeddings (Word2Vec, FastText, or BERT) instead of TF-IDF.
Implement Deep Learning with LSTMs or Transformers.
Deploy as an API for real-world use.
# codtech-task4
