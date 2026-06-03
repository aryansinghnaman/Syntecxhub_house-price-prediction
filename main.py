# =========================================
# HOUSE PRICE PREDICTION PROJECT
# =========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error
)

# =========================================
# LOAD DATASET
# =========================================

housing = fetch_california_housing()

df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

df["Price"] = housing.target

print("\nDATASET PREVIEW")
print(df.head())

# =========================================
# BASIC INFO
# =========================================

print("\nDATASET INFO")
print(df.info())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nSTATISTICAL SUMMARY")
print(df.describe())

# =========================================
# DATA VISUALIZATION
# =========================================

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

# =========================================
# FEATURES & TARGET
# =========================================

X = df.drop("Price", axis=1)
y = df["Price"]

# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================
# FEATURE SCALING
# =========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================
# LINEAR REGRESSION MODEL
# =========================================

lr_model = LinearRegression()

lr_model.fit(X_train_scaled, y_train)

lr_predictions = lr_model.predict(X_test_scaled)

# =========================================
# RANDOM FOREST MODEL
# =========================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

# =========================================
# EVALUATION FUNCTION
# =========================================

def evaluate_model(model_name, y_true, predictions):

    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    mae = mean_absolute_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)

    print(f"\n===== {model_name} =====")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R2   : {r2:.4f}")

    return r2

# =========================================
# EVALUATE MODELS
# =========================================

lr_r2 = evaluate_model(
    "Linear Regression",
    y_test,
    lr_predictions
)

rf_r2 = evaluate_model(
    "Random Forest",
    y_test,
    rf_predictions
)

# =========================================
# BEST MODEL SELECTION
# =========================================

if rf_r2 > lr_r2:
    best_model = rf_model
    model_name = "random_forest.pkl"

    print("\nBEST MODEL : RANDOM FOREST")

else:
    best_model = lr_model
    model_name = "linear_regression.pkl"

    print("\nBEST MODEL : LINEAR REGRESSION")

# =========================================
# SAVE MODEL
# =========================================

joblib.dump(best_model, model_name)

joblib.dump(scaler, "scaler.pkl")

print("\nMODEL SAVED SUCCESSFULLY")

# =========================================
# FEATURE IMPORTANCE
# =========================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print(importance)

plt.figure(figsize=(10,6))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance
)

plt.title("Feature Importance")
plt.savefig("feature_importance.png")
plt.close()

# =========================================
# PREDICTION VISUALIZATION
# =========================================

plt.figure(figsize=(8,6))

plt.scatter(y_test, rf_predictions)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted Prices")

plt.savefig("prediction_plot.png")

plt.close()

# =========================================
# SAMPLE PREDICTION
# =========================================

sample = X_test.iloc[0:1]

sample_prediction = best_model.predict(sample)

print("\nSAMPLE PREDICTION")
print(f"Predicted Price : {sample_prediction[0]:.4f}")
print(f"Actual Price    : {y_test.iloc[0]:.4f}")

print("\nPROJECT COMPLETED SUCCESSFULLY")