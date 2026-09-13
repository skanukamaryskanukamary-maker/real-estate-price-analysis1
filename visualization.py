import pandas as pd
import matplotlib

# Use non-GUI backend
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output folder
os.makedirs("data/processed/graphs", exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_properties.csv")

# Calculate price per sqft
df["price_per_sqft"] = (df["price_lakhs"] * 100000) / df["area_sqft"]


# 1. Average Price by Location
location_prices = df.groupby("location")["price_lakhs"].mean()

plt.figure(figsize=(8, 5))
location_prices.sort_values().plot(kind="barh")
plt.title("Average Property Price by Location")
plt.xlabel("Average Price (Lakhs)")
plt.ylabel("Location")
plt.tight_layout()
plt.savefig("data/processed/graphs/price_by_location.png")
plt.close()


# 2. Average Price by BHK
bhk_prices = df.groupby("bhk")["price_lakhs"].mean()

plt.figure(figsize=(7, 5))
bhk_prices.plot(kind="bar")
plt.title("Average Property Price by BHK")
plt.xlabel("BHK")
plt.ylabel("Average Price (Lakhs)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("data/processed/graphs/price_by_bhk.png")
plt.close()


# 3. Area vs Price
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="area_sqft",
    y="price_lakhs",
    hue="location",
    s=100
)
plt.title("Area vs Property Price")
plt.xlabel("Area (Sqft)")
plt.ylabel("Price (Lakhs)")
plt.tight_layout()
plt.savefig("data/processed/graphs/area_vs_price.png")
plt.close()


# 4. Property Type vs Price
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df,
    x="property_type",
    y="price_lakhs"
)
plt.title("Property Type vs Price")
plt.xlabel("Property Type")
plt.ylabel("Price (Lakhs)")
plt.tight_layout()
plt.savefig("data/processed/graphs/property_type_vs_price.png")
plt.close()


print("All visualizations generated successfully!")
print("Graphs saved in: data/processed/graphs/")