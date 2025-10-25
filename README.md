# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Support Vector Machine (SVM)

2. Naïve Bayes

3. Logistic Regression

4. Random Forest

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Shivasri 
RegisterNumber:  212224220098
*/
```
```
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
path = r"C:\Users\admin\Downloads\spam.csv"
data = pd.read_csv(path, encoding='latin-1')

# Display first few rows to understand structure
print("Dataset Sample:")
print(data.head(), "\n")

# --- Data Preprocessing ---
# Some spam.csv files contain extra unnamed columns. We’ll keep only what’s needed.
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert labels to binary values: ham -> 0, spam -> 1
data['label_num'] = data.label.map({'ham': 0, 'spam': 1})

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label_num'], test_size=0.2, random_state=42
)

# --- Text Feature Extraction using TF-IDF ---
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# --- Train SVM Model ---
model = SVC(kernel='linear')
model.fit(X_train_tfidf, y_train)

# --- Make Predictions ---
y_pred = model.predict(X_test_tfidf)

# --- Evaluate Model ---
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- Test on custom input ---
sample_text = ["Congratulations! You won a free iPhone. Click here to claim now!",
               "Hi John, please send me the project files by tomorrow."]
sample_features = vectorizer.transform(sample_text)
predictions = model.predict(sample_features)

print("\nSample Predictions:")
for text, label in zip(sample_text, predictions):
    print(f"Message: {text}\nPredicted: {'SPAM' if label else 'HAM'}\n")

```

## Output:
<img width="1905" height="1134" alt="image" src="https://github.com/user-attachments/assets/7b6aee12-7a20-408a-998e-b9c21dc6ace0" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
