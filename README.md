# 🏠 Real Estate Price Analysis & Prediction System

A Machine Learning based web application that analyzes real estate property data, explores market trends, and predicts property prices based on property characteristics.

## 📌 Project Overview

The **Real Estate Price Analysis & Prediction System** is developed using Python, Streamlit, Pandas, Scikit-learn, and Random Forest Machine Learning.

The application allows users to:

- Analyze property prices
- Explore real estate market trends
- Filter properties by location and property type
- Visualize property data using charts
- Predict property prices using Machine Learning
- View prediction history
- Download prediction history as CSV

---

## 🎯 Objectives

The main objectives of this project are:

1. To analyze real estate property data.
2. To identify price trends based on property features.
3. To visualize real estate market information.
4. To build a Machine Learning model for property price prediction.
5. To provide an easy-to-use web interface for users.

---

## 🛠️ Technologies Used

### Programming Language
- Python

### Libraries & Frameworks
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib

### Machine Learning
- Random Forest Regressor
- One-Hot Encoding
- Train-Test Split

### Data Visualization
- Streamlit Charts
- Matplotlib

---

## 📊 Dataset

The current dataset contains:

- **61 property records**
- **7 columns**

### Dataset Features

| Feature | Description |
|---|---|
| location | Property location |
| property_type | Type of property |
| bhk | Number of bedrooms |
| area_sqft | Property area in square feet |
| bathrooms | Number of bathrooms |
| property_age | Age of property |
| price_lakhs | Property price in lakhs |

### Locations Included

- Hyderabad
- Bangalore
- Chennai
- Pune
- Mumbai

### Property Types

- Apartment
- Villa

---

## 🤖 Machine Learning Model

The project uses a **Random Forest Regressor** for predicting property prices.

### Input Features

The model uses:

- Location
- Property Type
- BHK
- Area in Square Feet
- Number of Bathrooms
- Property Age

### Target

The target variable is:

`price_lakhs`

---

## 📈 Model Performance

The current trained model achieved the following results on the test split:

| Metric | Result |
|---|---:|
| R² Score | **0.9002** |
| MAE | **14.0412 Lakhs** |
| RMSE | **22.7044 Lakhs** |

> Note: The dataset is currently relatively small, so these results should be considered project-level evaluation results rather than real-world production performance.

---

## 🖥️ Application Features

### 📊 Dashboard

The dashboard provides an overview of the real estate dataset.

It includes:

- Average Property Price
- Average Property Area
- Total Properties
- Average BHK
- Properties by Location
- Properties by Property Type
- Average Price by BHK
- Area vs Price Analysis

---

### 🤖 Price Prediction

Users can enter property details such as:

- Location
- Property Type
- BHK
- Area
- Bathrooms
- Property Age

The Machine Learning model then predicts the estimated property price.

---

### 🤖 ML Model Performance

This section displays the model evaluation metrics:

- R² Score
- MAE
- RMSE

---

### 📜 Prediction History

The application stores previous predictions during the session.

Users can:

- View prediction history
- Check total predictions
- View average predicted price
- Download prediction history as CSV
- Clear prediction history

---

### 📈 Market Analysis

Users can explore the dataset using different filters and visualizations.

Available analysis includes:

- Location Analysis
- BHK Analysis
- Area vs Price
- Property Type Analysis

---

### 🏘️ Property Dataset

Users can view and filter the property dataset based on:

- Location
- Property Type

---

### ℹ️ About Project

The application also contains an About section explaining the project, technologies, Machine Learning approach, and objectives.

---

## 📂 Project Structure

```text
Real Estate Price Analysis & Prediction System/
│
├── app.py
├── app_backup.py
├── train_model.py
├── expand_dataset.py
├── README.md
│
├── data/
│   └── processed/
│       ├── cleaned_properties.csv
│       └── cleaned_properties_backup.csv
│
├── models/
│   └── real_estate_model.pkl
│
└── ...