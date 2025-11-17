# Always start with these imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load your CSV
df = pd.read_csv('your_file.csv')

# Explore the data
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())  # Check for missing values

# Handle missing values (if any)
df = df.dropna()  # Remove rows with missing values
# OR
df = df.fillna(df.mean())  # Fill with mean (for numeric columns)

# For categorical columns, convert to numbers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['category_column'] = le.fit_transform(df['category_column'])

# SVM Classification
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
df = pd.read_csv('your_file.csv')
X = df.drop('target_column', axis=1)  # Features
y = df['target_column']  # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = SVC(kernel='linear')  # Try 'rbf', 'poly' for other kernels
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier

# Load and split data (same as above)
df = pd.read_csv('your_file.csv')
X = df.drop('target_column', axis=1)
y = df['target_column']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nFeature Importance:\n", pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False))

# Naive Bayes Classification
from sklearn.naive_bayes import GaussianNB

# Load and split data
df = pd.read_csv('your_file.csv')
X = df.drop('target_column', axis=1)
y = df['target_column']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier

# Load and split data
df = pd.read_csv('your_file.csv')
X = df.drop('target_column', axis=1)
y = df['target_column']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nTree Depth:", model.get_depth())

