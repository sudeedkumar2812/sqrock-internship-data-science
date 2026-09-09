# ============================================================
# STOCK MARKET PREDICTION SYSTEM
# Data Science Internship - Phase 2
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# ============================================================
# FILE PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Stock Market Prediction System",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# LOAD DATA AND MODEL
# ============================================================

@st.cache_data
def load_stock_data():

    data = pd.read_csv(
        DATA_DIR / "AAPL_historical_stock_data.csv"
    )

    data["Date"] = pd.to_datetime(data["Date"])

    # Moving averages
    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA50"] = data["Close"].rolling(50).mean()
    data["MA200"] = data["Close"].rolling(200).mean()

    return data


@st.cache_resource
def load_model():

    model = joblib.load(
        MODEL_DIR / "linear_regression_stock_model.pkl"
    )

    scaler = joblib.load(
        MODEL_DIR / "stock_feature_scaler.pkl"
    )

    features = joblib.load(
        MODEL_DIR / "stock_feature_columns.pkl"
    )

    return model, scaler, features


stock_data = load_stock_data()
model, scaler, features = load_model()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📈 Stock Market Predictor")

st.sidebar.markdown(
    """
    ### Navigation

    Use the sections below to explore the stock market data,
    analyze historical trends, and generate predictions.
    """
)

selected_section = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Market Analysis",
        "Prediction",
        "Model Performance"
    ]
)


# ============================================================
# HEADER
# ============================================================

st.title("📈 Stock Market Prediction System")

st.markdown(
    """
    ### AAPL Stock Analysis & Next-Day Price Prediction

    This application analyzes historical stock market data
    and uses Machine Learning to predict the next trading
    day's closing price.

    **Model Used:** Linear Regression
    """
)

st.divider()


# ============================================================
# OVERVIEW
# ============================================================

if selected_section == "Overview":

    st.subheader("📊 Market Overview")

    latest = stock_data.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Latest Close",
            f"${latest['Close']:.2f}"
        )

    with col2:
        st.metric(
            "Day High",
            f"${latest['High']:.2f}"
        )

    with col3:
        st.metric(
            "Day Low",
            f"${latest['Low']:.2f}"
        )

    with col4:
        st.metric(
            "Trading Volume",
            f"{latest['Volume']:,.0f}"
        )

    st.subheader("Historical Closing Price")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        stock_data["Date"],
        stock_data["Close"]
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price ($)")
    ax.set_title("AAPL Historical Closing Price")

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write(
            f"**Records:** {len(stock_data):,}"
        )

    with col2:
        st.write(
            f"**Date Range:** "
            f"{stock_data['Date'].min().date()} "
            f"to "
            f"{stock_data['Date'].max().date()}"
        )


# ============================================================
# MARKET ANALYSIS
# ============================================================

elif selected_section == "Market Analysis":

    st.subheader("📊 Market Analysis")

    # Price chart
    st.markdown("### Price Trend")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        stock_data["Date"],
        stock_data["Close"],
        label="Close"
    )

    ax.plot(
        stock_data["Date"],
        stock_data["MA20"],
        label="20-Day MA"
    )

    ax.plot(
        stock_data["Date"],
        stock_data["MA50"],
        label="50-Day MA"
    )

    ax.plot(
        stock_data["Date"],
        stock_data["MA200"],
        label="200-Day MA"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.set_title("AAPL Price and Moving Averages")

    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

    # Volume
    st.markdown("### Trading Volume")

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        stock_data["Date"],
        stock_data["Volume"]
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Volume")
    ax.set_title("AAPL Trading Volume")

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

    # Statistics
    st.markdown("### Market Statistics")

    statistics = stock_data[
        ["Open", "High", "Low", "Close", "Volume"]
    ].describe()

    st.dataframe(
        statistics,
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

elif selected_section == "Prediction":

    st.subheader("🤖 Next-Day Stock Price Prediction")

    st.info(
        "The model predicts the next trading day's "
        "closing price using the latest available "
        "market information."
    )

    latest = stock_data.iloc[-1]

    prediction_features = pd.DataFrame(
        [[
            latest["Open"],
            latest["High"],
            latest["Low"],
            latest["Close"],
            latest["Volume"],
            latest["MA20"],
            latest["MA50"],
            latest["MA200"]
        ]],
        columns=features
    )

    scaled_features = scaler.transform(
        prediction_features
    )

    prediction = model.predict(
        scaled_features
    )[0]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Latest Closing Price",
            f"${latest['Close']:.2f}"
        )

    with col2:
        st.metric(
            "Predicted Next-Day Close",
            f"${prediction:.2f}"
        )

    price_difference = prediction - latest["Close"]

    percentage_change = (
        price_difference /
        latest["Close"]
    ) * 100

    st.subheader("Prediction Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Price Change",
            f"${price_difference:.2f}"
        )

    with col2:
        st.metric(
            "Predicted Change %",
            f"{percentage_change:.2f}%"
        )

    st.subheader("Recent Price Trend")

    recent_data = stock_data.tail(100)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        recent_data["Date"],
        recent_data["Close"],
        label="Historical Close"
    )

    ax.scatter(
        stock_data["Date"].iloc[-1],
        prediction,
        label="Predicted Next-Day Price",
        s=100
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.set_title("Recent AAPL Price Trend and Prediction")

    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

    st.warning(
        "This prediction is for educational purposes only "
        "and should not be considered financial advice."
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif selected_section == "Model Performance":

    st.subheader("📊 Machine Learning Model Performance")

    performance = pd.read_csv(
        MODEL_DIR / "model_performance.csv"
    )

    st.dataframe(
        performance,
        use_container_width=True
    )

    st.markdown(
        """
        ### Evaluation Metrics

        **MAE:** Mean Absolute Error

        **MSE:** Mean Squared Error

        **RMSE:** Root Mean Squared Error

        **R² Score:** Coefficient of Determination

        The model with the lowest RMSE was selected
        as the best-performing model.
        """
    )

    best_model = performance.loc[
        performance["RMSE"].idxmin()
    ]

    st.success(
        f"🏆 Best Model: {best_model['Model']}"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "MAE",
            f"{best_model['MAE']:.4f}"
        )

    with col2:
        st.metric(
            "MSE",
            f"{best_model['MSE']:.4f}"
        )

    with col3:
        st.metric(
            "RMSE",
            f"{best_model['RMSE']:.4f}"
        )

    with col4:
        st.metric(
            "R² Score",
            f"{best_model['R2 Score']:.4f}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Stock Market Prediction System | "
    "Data Science Internship – Phase 2 | "
    "Educational Project"
)