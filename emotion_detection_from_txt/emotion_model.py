import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

# Load the dataset (tab-separated)
train_df = pd.read_csv("train.txt", sep=';', header=None, names=["text", "emotion"])
test_df = pd.read_csv("test.txt", sep=';', header=None, names=["text", "emotion"])

# Combine text and label
X_train, y_train = train_df["text"], train_df["emotion"]
X_test, y_test = test_df["text"], test_df["emotion"]

# Build pipeline
model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, "emotion_classifier.pkl")
print("Model saved as emotion_classifier.pkl")