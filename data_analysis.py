import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_properties.csv")

print("===== REAL ESTATE DATA ANALYSIS =====")

# Calculate price per square foot
df["price_per_sqft"] = (df["price_lakhs"] * 100000) / df["area_sqft"]

# Overall statistics
print("\n--- Overall Statistics ---")
print(f"Average Price: {df['price_lakhs'].mean():.2f} Lakhs")
print(f"Average Area: {df['area_sqft'].mean():.2f} Sqft")
print(f"Average BHK: {df['bhk'].mean():.2f}")
print(f"Average Price per Sqft: ₹{df['price_per_sqft'].mean():.2f}")

# Location-wise average price
print("\n--- Average Price by Location ---")
location_prices = df.groupby("location")["price_lakhs"].mean().sort_values(ascending=False)
print(location_prices)

# BHK-wise average price
print("\n--- Average Price by BHK ---")
bhk_prices = df.groupby("bhk")["price_lakhs"].mean()
print(bhk_prices)

# Property type-wise average price
print("\n--- Average Price by Property Type ---")
property_type_prices = df.groupby("property_type")["price_lakhs"].mean()
print(property_type_prices)

# Most expensive property
print("\n--- Most Expensive Property ---")
most_expensive = df.loc[df["price_lakhs"].idxmax()]
print(most_expensive)

# Cheapest property
print("\n--- Cheapest Property ---")
cheapest = df.loc[df["price_lakhs"].idxmin()]
print(cheapest)