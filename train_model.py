import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ==============================
# 1. Load Dataset
# ==============================

data_path = "data/processed/cleaned_properties.csv"

df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==============================
# 2. Define Features and Target
# ==============================

X = df[
    [
        "location",
        "property_type",
        "bhk",
        "area_sqft",
        "bathrooms",
        "property_age"
    ]
]

y = df["price_lakhs"]


# ==============================
# 3. Categorical & Numerical Columns
# ==============================

categorical_features = [
    "location",
    "property_type"
]

numerical_features = [
    "bhk",
    "area_sqft",
    "bathrooms",
    "property_age"
]


# ==============================
# 4. Preprocessing
# ==============================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ==============================
# 5. Create Model
# ==============================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


# ==============================
# 6. Create Pipeline
# ==============================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ==============================
# 7. Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("Training model...")


# ==============================
# 8. Train Model
# ==============================

pipeline.fit(X_train, y_train)

print("Model training completed!")


# ==============================
# 9. Predictions
# ==============================

y_pred = pipeline.predict(X_test)


# ==============================
# 10. Model Performance
# ==============================

r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.4f} Lakhs")
print(f"RMSE     : {rmse:.4f} Lakhs")


# ==============================
# 11. Save Model
# ==============================

model_folder = "models"

os.makedirs(
    model_folder,
    exist_ok=True
)

model_path = os.path.join(
    model_folder,
    "real_estate_model.pkl"
)

with open(model_path, "wb") as file:
    pickle.dump(
        pipeline,
        file
    )


print("\nModel saved successfully!")
print("Location:", model_path)

print("\nTraining completed successfully! 🚀")