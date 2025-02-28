import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
print("NLTK data downloaded successfully.")


# Load dataset
df = pd.read_csv("Consumer_Complaints.csv")
print("Dataset loaded successfully.")


# Select relevant columns & drop missing values
df = df[['Product', 'Consumer complaint narrative']].dropna()

# Encode target labels
category_mapping = {
    'Credit reporting, repair, or other': 0,
    'Debt collection': 1,
    'Consumer Loan': 2,
    'Mortgage': 3
}
df = df[df['Product'].isin(category_mapping.keys())]
df['Label'] = df['Product'].map(category_mapping)
df.drop(columns=['Product'], inplace=True)

# Text Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['Processed_Text'] = df['Consumer complaint narrative'].apply(clean_text)
df.drop(columns=['Consumer complaint narrative'], inplace=True)

# Stopwords Removal & Stemming
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    words = word_tokenize(text)
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

df['Processed_Text'] = df['Processed_Text'].apply(preprocess_text)

# Convert text to TF-IDF features
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['Processed_Text'])
y = df['Label'].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

# Train Logistic Regression Model
lr_model = LogisticRegression(max_iter=500)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Train SVM Model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

# Model Evaluation
def evaluate_model(model_name, y_test, y_pred):
    print(f"\nModel: {model_name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

evaluate_model("Naive Bayes", y_test, nb_pred)
evaluate_model("Logistic Regression", y_test, lr_pred)
evaluate_model("SVM", y_test, svm_pred)

# Model Comparison
models = ['Naive Bayes', 'Logistic Regression', 'SVM']
accuracies = [
    accuracy_score(y_test, nb_pred),
    accuracy_score(y_test, lr_pred),
    accuracy_score(y_test, svm_pred)
]

plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=accuracies)
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.show()

# Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, preds, model in zip(axes, [nb_pred, lr_pred, svm_pred], models):
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix - {model}')
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
plt.show()

# Function for Predicting New Complaints
def predict_category(text, model):
    cleaned_text = preprocess_text(clean_text(text))
    vectorized_text = tfidf.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    return list(category_mapping.keys())[prediction]

# Example Prediction
new_text = "I have a complaint about my mortgage loan"
print("Predicted Category:", predict_category(new_text, lr_model))
