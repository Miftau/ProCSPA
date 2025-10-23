
"""
Train models for both new and used cars separately.
This script saves two models:
- model_used.pkl
- model_new.pkl
"""

import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib

# --------------------------
# Helper: clean dataframe
# --------------------------
def clean_dataframe(df):
    """Fix datatypes and fill missing values safely."""
    # Strip spaces from column names
    df.columns = df.columns.str.strip()

    # Convert numeric-like columns to numbers (ignore errors)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    # Replace commas in numbers if present
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].astype(str).str.replace(',', '').str.strip()

    # Try converting numeric-looking columns again
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

    return df

# ==============================
# TRAIN MODEL FOR USED CARS
# ==============================
print("🚗 Training model for USED cars...")

used_df = pd.read_csv("used_cars.csv")

# Fill missing values
used_df["fuel_type"].fillna(used_df["fuel_type"].mode()[0], inplace=True)
used_df["accident"].fillna(used_df["accident"].mode()[0], inplace=True)
used_df["clean_title"].fillna(used_df["clean_title"].mode()[0], inplace=True)

categorical_features_used = [
    "brand", "model", "fuel_type", "engine",
    "transmission", "ext_col", "int_col",
    "accident", "clean_title"
]
target_col_used = "price"

if 'milage' in used_df.columns:
    used_df['milage'] = (
        used_df['milage']
        .astype(str)                     # ensure string type
        .str.replace(',', '', regex=False)  # remove commas (e.g. "45,000" → "45000")
        .str.extract('(\d+)')            # extract digits only
        .fillna(0)
        .astype(float)                   # convert to float
    )
    
# --- Clean 'price' column for used cars ---
if 'price' in used_df.columns:
    used_df['price'] = (
        used_df['price']
        .astype(str)
        .str.replace('[₦$,]', '', regex=True)  # remove currency symbols and commas
        .str.extract('(\d+\.?\d*)')             # extract numeric part (handles decimals)
        .fillna(0)
        .astype(float)
    )
X_used = used_df.drop(columns=[target_col_used])
y_used = used_df[target_col_used]

encoder_used = TargetEncoder(cols=categorical_features_used)
X_used_encoded = encoder_used.fit_transform(X_used, y_used)

X_train, X_val, y_train, y_val = train_test_split(
    X_used_encoded, y_used, test_size=0.2, random_state=42
)

used_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

used_model.fit(X_train, y_train)
y_pred_used = used_model.predict(X_val)
rmse_used = np.sqrt(mean_squared_error(y_val, y_pred_used))
print(f"✅ Used Car Model RMSE: {rmse_used:.3f}")

# Save model + encoder
joblib.dump(used_model, "model_used.pkl")
joblib.dump(encoder_used, "encoder_used.pkl")
print("💾 Saved model_used.pkl and encoder_used.pkl")


# ==============================
# TRAIN MODEL FOR NEW CARS
# ==============================
print("\n🚙 Training model for NEW cars...")

new_df = pd.read_csv("car_sales_data.csv")

# Fill missing values if any
new_df.fillna(new_df.mode().iloc[0], inplace=True)
# Normalize new car dataset column names
new_df.columns = new_df.columns.str.lower().str.strip()

categorical_features_new = ["manufacturer", "model", "fuel type"]
numerical_features_new = ["engine size", "year of manufacture", "mileage"]
target_col_new = "price"
print(new_df.columns)


X_new = new_df.drop(columns=[target_col_new])
y_new = new_df[target_col_new]

encoder_new = TargetEncoder(cols=categorical_features_new)
X_new_encoded = encoder_new.fit_transform(X_new, y_new)

X_train, X_val, y_train, y_val = train_test_split(
    X_new_encoded, y_new, test_size=0.2, random_state=42
)

new_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

new_model.fit(X_train, y_train)
y_pred_new = new_model.predict(X_val)
rmse_new = np.sqrt(mean_squared_error(y_val, y_pred_new))
print(f"✅ New Car Model RMSE: {rmse_new:.3f}")

# Save model + encoder
joblib.dump(new_model, "model_new.pkl")
joblib.dump(encoder_new, "encoder_new.pkl")
print("💾 Saved model_new.pkl and encoder_new.pkl")

print("\n🎉 Training completed successfully!")
