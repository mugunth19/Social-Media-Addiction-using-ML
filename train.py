import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import joblib

# Load the dataset
file = "Students-Social-Media-Addiction.csv"
df = pd.read_csv(file)

# Convert column names to lowercase
df.columns = df.columns.str.lower()

# Drop the 'student_id' column as it's not useful
df.drop('student_id', axis=1, inplace=True)

# Define the target variable - convert to binary classification
# Addicted if score >= 7, not addicted otherwise
y_binary = (df['addicted_score'] >= 7).astype(int)

# Drop the original 'addicted_score' column from features
df_features = df.drop(columns=['addicted_score'])

# Apply DictVectorizer to handle categorical features
dv = DictVectorizer(sparse=False)
X_dict_features = df_features.to_dict(orient='records')
X_features = dv.fit_transform(X_dict_features)

# Apply scaling to the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# Split the data into train, validation, and test sets
# 80% train+val, 20% test
X_full_train, X_test, y_full_train, y_test = train_test_split(
    X_scaled, y_binary, test_size=0.2, random_state=42
)

# Split train+val into 75% train, 25% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_full_train, y_full_train, test_size=0.25, random_state=42
)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

# Train the Logistic Regression model
model = LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)

# Evaluate on validation set
y_val_pred = model.predict(X_val)
y_val_pred_proba = model.predict_proba(X_val)[:, 1]

print("\n--- Validation Set Results ---")
print("Classification Report for Logistic Regression on Validation Set:")
print(classification_report(y_val, y_val_pred))
print(f"AUC Score on Validation Set: {roc_auc_score(y_val, y_val_pred_proba)}")

# Evaluate on test set
y_test_pred = model.predict(X_test)
y_test_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n--- Test Set Results ---")
print("Classification Report for Logistic Regression on Test Set:")
print(classification_report(y_test, y_test_pred))
print(f"AUC Score on Test Set: {roc_auc_score(y_test, y_test_pred_proba)}")

# Save the model and scaler
joblib.dump(model, "logistic_regression_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(dv, "dict_vectorizer.pkl")

print("\n--- Model Saved ---")
print("Model saved as 'logistic_regression_model.pkl'")
print("Scaler saved as 'scaler.pkl'")
print("DictVectorizer saved as 'dict_vectorizer.pkl'")
