import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
try:
    train_df = pd.read_csv("train.csv")  # Update with your file path
    test_df = pd.read_csv("test.csv")    # Update with your file path
    print("Dataset loaded successfully.")
except FileNotFoundError as e:
    print("File not found:", e)
    exit()

# Basic exploration
print(train_df.info())
print(train_df.head())

# Convert all columns to numeric, coercing errors to NaN (if any)
train_df = train_df.apply(pd.to_numeric, errors='coerce')

# Fill missing values with the column mean (for numeric columns)
train_df.fillna(train_df.mean(), inplace=True)

# Feature and target selection (ensure you drop any non-numeric columns if needed)
X = train_df.drop(columns=["SalePrice", "Id"], errors="ignore")  # Independent variables
y = train_df["SalePrice"]  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Feature importance visualization
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.show()
