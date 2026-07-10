"""
Email Spam Classifier using Logistic Regression
--------------------------------------------------
A simple text classification model that predicts whether an email
message is spam (1) or not spam (0) using a Bag-of-Words representation
(CountVectorizer) and Logistic Regression.
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Logistic Regression based Email Spam Checker")

# Example data: List of emails and labels (1 = spam, 0 = not spam)
data = [
    ("Free money now!", 1),
    ("Meeting at 10am tomorrow", 0),
    ("Earn $500 a day from home", 1),
    ("Lunch at 12pm?", 0),
    ("Get rich quick, no investment!", 1),
    ("Your account has been compromised", 1),
    ("Important update on your account", 0),
    ("Limited time offer: Win a prize!", 1),
    ("Reminder: Doctor's appointment at 3pm", 0),
    ("Claim your free prize now!", 1),
    ("Hello, I hope you're doing well", 0),
    ("Don't miss out on this limited offer", 1),
    ("URGENT: Your bank account needs verification", 1),
    ("Schedule confirmation for our meeting", 0),
    ("You have a new friend request", 0),
    ("Congratulations! You've won a gift card", 1),
    ("Final warning: Your account will be locked", 1),
    ("Your subscription is about to expire", 0),
    ("Special offer: 50% off on all items", 1),
    ("Reminder: Team meeting at 4pm", 0),
    ("Unlock amazing rewards today", 1),
    ("Hey, let's catch up soon", 0),
    ("Don't miss this exclusive offer", 1),
    ("Your password has been changed", 1),
    ("Looking forward to seeing you at the event", 0),
    ("Quick loan approval with no credit check", 1),
    ("New job openings for you", 0),
    ("Limited time offer on electronics", 1),
    ("You're pre-approved for a loan", 1),
    ("Special invitation to our VIP event", 1),
    ("Don't forget to complete your survey", 0),
    ("Check out this amazing opportunity", 1),
    ("You've been selected for an exclusive offer", 1),
    ("Account verification required", 1),
    ("Reminder: Your meeting is scheduled", 0),
    ("You are a winner!", 1),
    ("Free webinar on investment strategies", 1),
    ("Breaking news: Major event happening", 0),
    ("Don't miss your chance to win a laptop", 1),
    ("Your order has been shipped", 0),
    ("Congrats! You've earned a reward", 1),
    ("Limited time promotion on new products", 1),
    ("Your email has been flagged as suspicious", 1),
    ("Special discount just for you", 1),
    ("You've been approved for a credit card", 1),
    ("How to invest wisely in the stock market", 0),
    ("Exclusive access to our new course", 1),
    ("Hey, just wanted to check in", 0),
    ("New free ebook on finance", 1),
    ("You've received a special invitation", 1),
    ("Important security update", 1),
    ("Join our online seminar today", 1),
    ("Reminder: Your package is ready for pickup", 0),
    ("Congratulations! You've been chosen for a free gift", 1),
    ("Alert: Unauthorized login attempt detected", 1),
    ("Hey, let's meet this weekend", 0),
    ("Hurry! This deal won't last long", 1),
    ("Meeting rescheduled to 2pm", 0),
    ("Free trial for premium services", 1),
    ("Account activity alert", 1),
    ("You're eligible for a free gift card", 1),
    ("Urgent: Account confirmation needed", 1),
    ("Your product has been shipped", 0),
    ("You're invited to a special event", 1),
    ("Exclusive offer for premium members", 1),
    ("Win a free vacation now!", 1),
    ("Reminder: Your subscription expires soon", 0),
    ("You've been selected for a free consultation", 1),
    ("Important notice from your bank", 1),
    ("You are almost there! Just one more step", 1),
    ("Important: Your payment is due soon", 1),
    ("Special offer on travel packages", 1),
    ("Don't miss our special discount offer", 1),
    ("Meeting details for tomorrow", 0),
    ("Offer ends tonight!", 1),
    ("Exclusive VIP offers just for you", 1),
    ("Get paid to take surveys", 1),
    ("Your monthly statement is available", 0),
    ("Reminder: Your appointment is in one hour", 0),
    ("Exclusive deal on electronics", 1),
    ("You've received a new message", 0),
    ("Special offer just for you: 30% off", 1),
    ("Watch this amazing video for free", 1),
    ("Hey, I have a question for you", 0),
    ("You have a new credit card offer", 1),
    ("Don't miss your chance to win big", 1),
    ("Order confirmation for your recent purchase", 0),
    ("Limited time only! Get 50% off", 1),
    ("Check out these great investment tips", 0),
    ("Hey, how's everything going?", 0),
    ("Special promotion on our latest product", 1),
    ("Claim your free prize now!", 1),
    ("You have a new connection request", 0),
    ("Your subscription is about to renew", 0),
    ("Get a free iPhone now!", 1),
    ("Offer ends soon! Don't miss out", 1),
    ("Important: Your account is at risk", 1),
    ("Reminder: Your meeting is scheduled for 10am", 0),
    ("Exclusive deals for new members", 1),
    ("You have been pre-approved for a loan", 1),
    ("Quick payday loans, no credit check", 1),
    ("Get your free trial of premium services", 1),
    ("New job opening at your favorite company", 0),
    ("Reminder: Your payment is due tomorrow", 0),
    ("Congratulations! You've won a gift", 1),
    ("Your gift card is ready", 1),
    ("Act now! This offer expires soon", 1),
    ("Final offer: Get a free vacation", 1),
    ("Meeting cancelled, see you next week", 0),
    ("Special offer on new phones", 1),
    ("Complete your registration for free", 1),
    ("Thank you for being a loyal customer", 0),
    ("Apply now for a fast loan", 1),
    ("Exclusive event invitation", 1),
    ("Important update from your provider", 1),
    ("Special invitation for members only", 1),
    ("Reminder: Your free trial ends soon", 0),
    ("Get your free consultation now!", 1),
    ("Your order has been confirmed", 0),
    ("Important: Confirm your subscription", 1),
    ("Get a free consultation from experts", 1),
    ("Earn money from home with no effort", 1),
    ("Congratulations on your new job!", 0),
    ("Special offer for you: 25% off", 1),
    ("Last chance to claim your free gift", 1),
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

# Train the Logistic Regression model
model = LogisticRegression()
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
