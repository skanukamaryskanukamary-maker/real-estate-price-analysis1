import pandas as pd

# File paths
input_file = "data/raw/properties.csv"
output_file = "data/processed/cleaned_properties.csv"

# Load dataset
df = pd.read_csv(input_file)

print("Original Dataset:")
print(df)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Clean text columns
text_columns = ["location", "property_type"]

for col in text_columns:
    df[col] = df[col].astype(str).str.strip()

# Convert numeric columns
numeric_columns = [
    "bhk",
    "area_sqft",
    "bathrooms",
    "property_age",
    "price_lakhs"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows with missing values
df = df.dropna()

# Keep only valid values
df = df[
    (df["bhk"] > 0) &
    (df["area_sqft"] > 0) &
    (df["bathrooms"] > 0) &
    (df["property_age"] >= 0) &
    (df["price_lakhs"] > 0)
]

# Save cleaned dataset
df.to_csv(output_file, index=False)

print("\nData cleaning completed successfully!")
print(f"Cleaned dataset saved to: {output_file}")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")