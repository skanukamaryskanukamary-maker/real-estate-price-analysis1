
import streamlit as st
import pandas as pd
import pickle


# -----------------------------
# Load Model and Dataset
# ----------------------------

with open("models/real_estate_model.pkl", "rb") as file:
    model = pickle.load(file)

df = pd.read_csv("data/processed/cleaned_properties.csv")


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Real Estate Analysis & Prediction",
    page_icon="🏠",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------

st.title("🏠 Real Estate Analysis & Prediction System")

st.write(
    "Analyze property prices and predict the estimated price "
    "using Machine Learning."
)


# -----------------------------
# Dashboard Statistics
# -----------------------------

st.subheader("📊 Property Market Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Average Price",
        f"₹{df['price_lakhs'].mean():.2f} Lakhs"
    )

with col2:
    st.metric(
        "Average Area",
        f"{df['area_sqft'].mean():.0f} Sqft"
    )

with col3:
    st.metric(
        "Total Properties",
        len(df)
    )

with col4:
    st.metric(
        "Average BHK",
        f"{df['bhk'].mean():.1f}"
    )


st.divider()


# -----------------------------
# Price Prediction
# -----------------------------

st.header("🤖 Property Price Prediction")

col1, col2 = st.columns(2)

with col1:

    location = st.selectbox(
        "Location",
        sorted(df["location"].unique())
    )

    property_type = st.selectbox(
        "Property Type",
        sorted(df["property_type"].unique())
    )

    bhk = st.number_input(
        "Number of BHK",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )


with col2:

    area_sqft = st.number_input(
        "Area (Square Feet)",
        min_value=300,
        max_value=10000,
        value=1200,
        step=50
    )

    bathrooms = st.number_input(
        "Number of Bathrooms",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

    property_age = st.number_input(
        "Property Age (Years)",
        min_value=0,
        max_value=100,
        value=5,
        step=1
    )


# -----------------------------
# Prediction Button
# -----------------------------

if st.button(
    "🔮 Predict Property Price",
    use_container_width=True
):

    input_data = pd.DataFrame({
        "location": [location],
        "property_type": [property_type],
        "bhk": [bhk],
        "area_sqft": [area_sqft],
        "bathrooms": [bathrooms],
        "property_age": [property_age]
    })

    prediction = model.predict(input_data)

    predicted_price = prediction[0]

    st.success(
        f"🏠 Estimated Property Price: ₹{predicted_price:.2f} Lakhs"
    )

    st.info(
        f"Approximately ₹{predicted_price * 100000:,.0f}"
    )


st.divider()


# -----------------------------
# Market Analysis
# -----------------------------

st.header("📈 Market Analysis")

tab1, tab2, tab3, tab4 = st.tabs([
    "📍 Location",
    "🛏️ BHK",
    "📐 Area vs Price",
    "🏢 Property Type"
])


# -----------------------------
# Location Analysis
# -----------------------------

with tab1:

    st.subheader("Average Price by Location")

    location_prices = (
        df.groupby("location")["price_lakhs"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(location_prices)


# -----------------------------
# BHK Analysis
# -----------------------------

with tab2:

    st.subheader("Average Price by BHK")

    bhk_prices = (
        df.groupby("bhk")["price_lakhs"]
        .mean()
    )

    st.bar_chart(bhk_prices)


# -----------------------------
# Area vs Price Analysis
# -----------------------------

with tab3:

    st.subheader("Area vs Property Price")

    chart_data = df[
        ["area_sqft", "price_lakhs"]
    ].set_index("area_sqft")

    st.scatter_chart(chart_data)


# -----------------------------
# Property Type Analysis
# -----------------------------

with tab4:

    st.subheader("Average Price by Property Type")

    type_prices = (
        df.groupby("property_type")["price_lakhs"]
        .mean()
    )

    st.bar_chart(type_prices)


st.divider()


# -----------------------------
# Property Dataset
# -----------------------------

st.header("🏘️ Property Dataset")

st.dataframe(
    df,
    use_container_width=True
)
