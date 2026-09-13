import streamlit as st
import pandas as pd
import pickle


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Real Estate AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ============================================================
# LOAD MODEL AND DATASET
# ============================================================

try:

    with open("models/real_estate_model.pkl", "rb") as file:
        model = pickle.load(file)

    df = pd.read_csv(
        "data/processed/cleaned_properties.csv"
    )

except FileNotFoundError as e:

    st.error("❌ Required file not found.")
    st.write("Please check your project folder structure.")
    st.code(str(e))
    st.stop()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Main title */

    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 8px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 25px;
    }

    /* Section title */

    .section-title {
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    /* Hero section */

    .hero-box {
        padding: 28px;
        border-radius: 16px;
        border: 1px solid #dddddd;
        text-align: center;
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 800;
    }

    .hero-text {
        font-size: 17px;
        margin-top: 8px;
    }

    /* Info cards */

    .info-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #dddddd;
        text-align: center;
        min-height: 120px;
    }

    .info-title {
        font-size: 16px;
        font-weight: 600;
    }

    .info-value {
        font-size: 25px;
        font-weight: 700;
        margin-top: 8px;
    }

    /* Footer */

    .footer {
        text-align: center;
        padding: 25px 10px;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "# 🏠 Real Estate AI"
)

st.sidebar.write(
    "Real Estate Price Analysis & Prediction System"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "📌 Navigation",
    [
        "📊 Dashboard",
        "🤖 Price Prediction",
        "🤖 ML Model Performance",
        "📜 Prediction History",
        "📈 Market Analysis",
        "🏘️ Property Dataset",
        "ℹ️ About Project"
    ]
)

st.sidebar.divider()

st.sidebar.markdown(
    "### 🛠️ Technologies"
)

st.sidebar.caption("Python")
st.sidebar.caption("Pandas")
st.sidebar.caption("Scikit-learn")
st.sidebar.caption("Streamlit")
st.sidebar.caption("Machine Learning")

st.sidebar.divider()

st.sidebar.caption(
    "🏠 Real Estate AI Project"
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "📊 Dashboard":

    st.markdown(
        """
        <div class="hero-title">
            🏠 Real Estate Price Analysis & Prediction
        </div>

        <div class="hero-text">
            Analyze property prices, explore market trends,
            and predict property values using Machine Learning.
        </div>
        """,
        unsafe_allow_html=True
    )

    
    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🔎 Dashboard Filter</div>',
        unsafe_allow_html=True
    )

    location_options = (
        ["All Locations"]
        + sorted(
            df["location"]
            .dropna()
            .unique()
        )
    )

    selected_location = st.selectbox(
        "📍 Select Location",
        location_options
    )

    if selected_location == "All Locations":

        dashboard_df = df.copy()

    else:

        dashboard_df = df[
            df["location"] == selected_location
        ].copy()

    st.divider()

    # --------------------------------------------------------
    # MARKET OVERVIEW
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">📊 Market Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    # Average Price

    with col1:

        if len(dashboard_df) > 0:

            average_price = dashboard_df[
                "price_lakhs"
            ].mean()

            st.metric(
                "💰 Average Price",
                f"₹{average_price:.2f} Lakhs"
            )

        else:

            st.metric(
                "💰 Average Price",
                "N/A"
            )

    # Average Area

    with col2:

        if len(dashboard_df) > 0:

            average_area = dashboard_df[
                "area_sqft"
            ].mean()

            st.metric(
                "📐 Average Area",
                f"{average_area:.0f} Sqft"
            )

        else:

            st.metric(
                "📐 Average Area",
                "N/A"
            )

    # Total Properties

    with col3:

        st.metric(
            "🏘️ Properties",
            len(dashboard_df)
        )

    # Average BHK

    with col4:

        if len(dashboard_df) > 0:

            average_bhk = dashboard_df[
                "bhk"
            ].mean()

            st.metric(
                "🛏️ Average BHK",
                f"{average_bhk:.1f}"
            )

        else:

            st.metric(
                "🛏️ Average BHK",
                "N/A"
            )

    st.divider()

    # --------------------------------------------------------
    # DISTRIBUTION CHARTS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">📊 Property Distribution</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "📍 Properties by Location"
        )

        location_count = (
            dashboard_df["location"]
            .value_counts()
            .head(10)
        )

        st.bar_chart(
            location_count
        )

    with col2:

        st.subheader(
            "🏢 Properties by Type"
        )

        type_count = (
            dashboard_df["property_type"]
            .value_counts()
        )

        st.bar_chart(
            type_count
        )

    st.divider()

    # --------------------------------------------------------
    # BHK CHART
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🛏️ Average Price by BHK</div>',
        unsafe_allow_html=True
    )

    bhk_dashboard = (
        dashboard_df
        .groupby("bhk")["price_lakhs"]
        .mean()
        .sort_index()
    )

    st.bar_chart(
        bhk_dashboard
    )

    st.divider()

    # --------------------------------------------------------
    # AREA VS PRICE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">📐 Area vs Property Price</div>',
        unsafe_allow_html=True
    )

    area_price_data = dashboard_df[
        [
            "area_sqft",
            "price_lakhs"
        ]
    ].dropna()

    st.scatter_chart(
        area_price_data,
        x="area_sqft",
        y="price_lakhs"
    )


# ============================================================
# PRICE PREDICTION
# ============================================================

elif page == "🤖 Price Prediction":

    st.markdown(
        '<div class="section-title">🤖 Property Price Prediction</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter the property details to estimate its market price."
    )

    st.divider()

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # PROPERTY DETAILS
    # --------------------------------------------------------

    with col1:

        location = st.selectbox(
            "📍 Location",
            sorted(
                df["location"]
                .dropna()
                .unique()
            )
        )

        property_type = st.selectbox(
            "🏢 Property Type",
            sorted(
                df["property_type"]
                .dropna()
                .unique()
            )
        )

        bhk = st.number_input(
            "🛏️ Number of BHK",
            min_value=1,
            max_value=10,
            value=2,
            step=1
        )

    with col2:

        area_sqft = st.number_input(
            "📐 Area (Square Feet)",
            min_value=300,
            max_value=10000,
            value=1200,
            step=50
        )

        bathrooms = st.number_input(
            "🚿 Number of Bathrooms",
            min_value=1,
            max_value=10,
            value=2,
            step=1
        )

        property_age = st.number_input(
            "🏗️ Property Age (Years)",
            min_value=0,
            max_value=100,
            value=5,
            step=1
        )

    st.write("")

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Property Price",
        use_container_width=True
    ):

        try:

            # Create input dataframe

            input_data = pd.DataFrame({
                "location": [location],
                "property_type": [property_type],
                "bhk": [bhk],
                "area_sqft": [area_sqft],
                "bathrooms": [bathrooms],
                "property_age": [property_age]
            })

            # Model prediction

            prediction = model.predict(
                input_data
            )

            predicted_price = float(
                prediction[0]
            )

            # ------------------------------------------------
            # SAVE HISTORY
            # ------------------------------------------------

            st.session_state.prediction_history.append(
                {
                    "Location": location,
                    "Property Type": property_type,
                    "BHK": bhk,
                    "Area (Sqft)": area_sqft,
                    "Bathrooms": bathrooms,
                    "Property Age": property_age,
                    "Predicted Price (Lakhs)": round(
                        predicted_price,
                        2
                    )
                }
            )

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.success(
                f"🏠 Estimated Property Price: ₹{predicted_price:.2f} Lakhs"
            )

            st.info(
                f"💰 Approximately ₹{predicted_price * 100000:,.0f}"
            )

            st.success(
                "✅ Property price prediction completed successfully."
            )

        except Exception as e:

            st.error(
                f"❌ Prediction failed: {e}"
            )


# ============================================================
# ML MODEL PERFORMANCE
# ============================================================

elif page == "🤖 ML Model Performance":

    st.markdown(
        '<div class="section-title">🤖 ML Model Performance</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Performance metrics of the trained Random Forest Regression model."
    )

    st.divider()

    # --------------------------------------------------------
    # PERFORMANCE METRICS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "🎯 R² Score",
            "0.6767"
        )

    with col2:

        st.metric(
            "📉 MAE",
            "17.0967 Lakhs"
        )

    with col3:

        st.metric(
            "📈 RMSE",
            "17.0984 Lakhs"
        )

    st.divider()

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "🤖 Model Information"
    )

    info_col1, info_col2 = st.columns(2)

    with info_col1:

        st.info(
            "🤖 Model: Random Forest Regressor"
        )

    with info_col2:

        st.info(
            "📊 Evaluation Dataset: 20% Test Split"
        )

    st.divider()

    # --------------------------------------------------------
    # PERFORMANCE SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "📌 Performance Summary"
    )

    st.write(
        """
        The Random Forest Regression model was trained using
        property information such as location, property type,
        BHK, area, bathrooms, and property age.

        The model achieved an R² Score of 0.6767 on the test data.
        The MAE was 17.0967 Lakhs and the RMSE was 17.0984 Lakhs.
        """
    )

    st.success(
        "✅ Model training and evaluation completed successfully."
    )

    st.warning(
        "⚠️ The current dataset contains only 11 properties, "
        "so these performance metrics should be considered "
        "as project/demo evaluation results rather than "
        "production-level model accuracy."
    )


# ============================================================
# PREDICTION HISTORY
# ============================================================

elif page == "📜 Prediction History":

    st.markdown(
        '<div class="section-title">📜 Prediction History</div>',
        unsafe_allow_html=True
    )

    st.write(
        "View and download your previous property predictions."
    )

    st.divider()

    if len(st.session_state.prediction_history) == 0:

        st.info(
            "No predictions yet. Go to Price Prediction and make a prediction."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.prediction_history
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "🔮 Total Predictions",
                len(history_df)
            )

        with col2:

            average_prediction = (
                history_df[
                    "Predicted Price (Lakhs)"
                ].mean()
            )

            st.metric(
                "💰 Average Predicted Price",
                f"₹{average_prediction:.2f} Lakhs"
            )

        st.divider()

        # ----------------------------------------------------
        # HISTORY TABLE
        # ----------------------------------------------------

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        csv = history_df.to_csv(
            index=False
        )

        st.download_button(
            "📥 Download Prediction History",
            csv,
            "prediction_history.csv",
            "text/csv",
            use_container_width=True
        )

        st.write("")

        # ----------------------------------------------------
        # CLEAR HISTORY
        # ----------------------------------------------------

        if st.button(
            "🗑️ Clear Prediction History",
            use_container_width=True
        ):

            st.session_state.prediction_history = []

            st.rerun()


# ============================================================
# MARKET ANALYSIS
# ============================================================

elif page == "📈 Market Analysis":

    st.markdown(
        '<div class="section-title">📈 Market Analysis</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Explore property prices using different market indicators."
    )

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📍 Location",
            "🛏️ BHK",
            "📐 Area vs Price",
            "🏢 Property Type"
        ]
    )

    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    with tab1:

        st.subheader(
            "📍 Average Price by Location"
        )

        location_prices = (
            df.groupby("location")[
                "price_lakhs"
            ]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        st.bar_chart(
            location_prices
        )

    # --------------------------------------------------------
    # BHK
    # --------------------------------------------------------

    with tab2:

        st.subheader(
            "🛏️ Average Price by BHK"
        )

        bhk_prices = (
            df.groupby("bhk")[
                "price_lakhs"
            ]
            .mean()
            .sort_index()
        )

        st.bar_chart(
            bhk_prices
        )

    # --------------------------------------------------------
    # AREA VS PRICE
    # --------------------------------------------------------

    with tab3:

        st.subheader(
            "📐 Area vs Property Price"
        )

        chart_data = df[
            [
                "area_sqft",
                "price_lakhs"
            ]
        ].dropna()

        st.scatter_chart(
            chart_data,
            x="area_sqft",
            y="price_lakhs"
        )

    # --------------------------------------------------------
    # PROPERTY TYPE
    # --------------------------------------------------------

    with tab4:

        st.subheader(
            "🏢 Average Price by Property Type"
        )

        type_prices = (
            df.groupby("property_type")[
                "price_lakhs"
            ]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        st.bar_chart(
            type_prices
        )


# ============================================================
# PROPERTY DATASET
# ============================================================

elif page == "🏘️ Property Dataset":

    st.markdown(
        '<div class="section-title">🏘️ Property Dataset</div>',
        unsafe_allow_html=True
    )

    st.write(
        f"Total dataset records: {len(df)}"
    )

    st.divider()

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        search_location = st.selectbox(
            "📍 Filter by Location",
            ["All Locations"]
            + sorted(
                df["location"]
                .dropna()
                .unique()
            )
        )

    with col2:

        search_type = st.selectbox(
            "🏢 Filter by Property Type",
            ["All Types"]
            + sorted(
                df["property_type"]
                .dropna()
                .unique()
            )
        )

    # --------------------------------------------------------
    # APPLY FILTERS
    # --------------------------------------------------------

    filtered_df = df.copy()

    if search_location != "All Locations":

        filtered_df = filtered_df[
            filtered_df["location"] == search_location
        ]

    if search_type != "All Types":

        filtered_df = filtered_df[
            filtered_df["property_type"] == search_type
        ]

    st.write(
        f"### Displaying {len(filtered_df)} records"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.markdown(
        '<div class="section-title">ℹ️ About This Project</div>',
        unsafe_allow_html=True
    )

    st.write("")

    st.markdown(
        """
        ### 🏠 Real Estate Price Analysis & Prediction System
        """
    )

    st.write(
        """
        This is a Machine Learning based application designed to
        analyze real estate property data and estimate property prices.

        Users can explore market trends, filter property data,
        and predict the estimated price of a property using
        important property features.
        """
    )

    st.divider()

    st.subheader(
        "🛠️ Technologies Used"
    )

    tech_col1, tech_col2, tech_col3 = st.columns(3)

    with tech_col1:

        st.info("🐍 Python")
        st.info("🐼 Pandas")

    with tech_col2:

        st.info("🤖 Scikit-learn")
        st.info("🌐 Streamlit")

    with tech_col3:

        st.info("🧠 Machine Learning")
        st.info("📦 Pickle")

    st.divider()

    st.subheader(
        "🤖 Machine Learning Model"
    )

    st.write(
        """
        The trained Machine Learning model uses the following
        property features:

        • Location
        • Property Type
        • Number of BHK
        • Area in Square Feet
        • Number of Bathrooms
        • Property Age

        Based on these inputs, the model estimates the property price.
        """
    )

    st.divider()

    st.subheader(
        "📊 Dataset Information"
    )

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:

        st.metric(
            "Total Properties",
            len(df)
        )

    with info_col2:

        st.metric(
            "Average Price",
            f"₹{df['price_lakhs'].mean():.2f} L"
        )

    with info_col3:

        st.metric(
            "Average Area",
            f"{df['area_sqft'].mean():.0f} Sqft"
        )

    st.divider()

    st.success(
        "✅ Real Estate Price Analysis & Prediction System"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    "🏠 **Real Estate Price Analysis & Prediction System**"
)

st.caption(
    "Machine Learning Project"
)