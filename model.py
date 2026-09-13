import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_properties.csv")

print("Dataset loaded successfully!")
print(f"Total records: {len(df)}")


# Features and target
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


# Categorical and numerical columns
categorical_columns = [
    "location",
    "property_type"
]

numerical_columns = [
    "bhk",
    "area_sqft",
    "bathrooms",
    "property_age"
]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)


# Machine Learning model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# Create pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train model
pipeline.fit(X_train, y_train)


# Make predictions
y_pred = pipeline.predict(X_test)


# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)


print("\n===== MODEL PERFORMANCE =====")
print(f"Mean Absolute Error: {mae:.2f} Lakhs")
print(f"Root Mean Squared Error: {rmse:.2f} Lakhs")
print(f"R2 Score: {r2:.2f}")


# Save model
with open("models/real_estate_model.pkl", "wb") as file:
    pickle.dump(pipeline, file)


print("\nModel trained successfully!")
print("Model saved to: models/real_estate_model.pkl")