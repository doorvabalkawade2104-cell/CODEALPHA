
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the dataset
df = pd.read_csv("car data.csv")

print("First five records:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# 2. Remove duplicates
df = df.drop_duplicates()

# 3. Feature engineering
df["Car_Age"] = 2026 - df["Year"]

# 4. Select features and target
X = df[
    [
        "Car_Name",
        "Present_Price",
        "Driven_kms",
        "Fuel_Type",
        "Selling_type",
        "Transmission",
        "Owner",
        "Car_Age"
    ]
]

y = df["Selling_Price"]

# 5. Separate categorical and numerical columns
categorical_cols = [
    "Car_Name",
    "Fuel_Type",
    "Selling_type",
    "Transmission"
]

numerical_cols = [
    "Present_Price",
    "Driven_kms",
    "Owner",
    "Car_Age"
]

# 6. Preprocessing
numeric_pipeline = SimpleImputer(strategy="median")

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

# 7. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Create regression model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# 9. Train the model
model.fit(X_train, y_train)

# 10. Predict prices
y_pred = model.predict(X_test)

# 11. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R2 Score:", round(r2, 3))

# 12. Plot actual vs predicted prices
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.7)

plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")

lower = min(y_test.min(), y_pred.min())
upper = max(y_test.max(), y_pred.max())

plt.plot([lower, upper], [lower, upper], "r--")
plt.tight_layout()
plt.show()

# 13. Predict the price of a sample car
sample_car = X.iloc[[0]]
predicted_price = model.predict(sample_car)[0]

print("\nSample car:", df.iloc[0]["Car_Name"])
print("Predicted Selling Price:", round(predicted_price, 2), "lakhs")
