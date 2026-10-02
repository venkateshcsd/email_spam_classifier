"""
Email Spam Classifier using Naive Bayes
--------------------------------------------------
A simple text classification model that predicts whether an email message is spam (1) or not spam (0)
using a Bag-of-Words representation (CountVectorizer) and Naive Bayes (MultinomialNB).
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB  # 1. Changed import
from sklearn.metrics import accuracy_score

print("Naive Bayes based Email Spam Checker")

# Example data: List of emails and labels (1 = spam, 0 = not spam)
data = [
    ("Free money now!", 1),
    ("Meeting at 10am tomorrow", 0),
    # ... rest of your dataset ...
]

df = pd.DataFrame(data, columns=["Email", "Label"])

# Convert text into numeric feature vectors (Bag of Words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["Email"])
y = df["Label"]

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Naive Bayes model
model = MultinomialNB()  # 2. Replaced model instantiation
model.fit(X_train, y_train)

# Evaluate on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

def predict_email(text: str) -> str:
    """Predict whether a single email string is spam or not."""
    vectorized = vectorizer.transform([text])
    prediction = model.predict(vectorized)[0]
    return "Spam" if prediction == 1 else "Not Spam"

if __name__ == "__main__":
    user_input = input("Enter an email message to classify: ")
    result = predict_email(user_input)
    print(f"Prediction: {result}")
