# ============================================================
# LOAN APPROVAL PREDICTION SYSTEM
# Data Science Internship - Phase 2
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt


# ============================================================
# FILE PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        DATA_DIR / "cleaned_loan_approval_dataset.csv"
    )

    return data


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        MODEL_DIR / "loan_approval_model.pkl"
    )

    features = joblib.load(
        MODEL_DIR / "feature_columns.pkl"
    )

    return model, features


# ============================================================
# LOAD MODEL PERFORMANCE
# ============================================================

@st.cache_data
def load_performance():

    performance = pd.read_csv(
        MODEL_DIR / "model_performance.csv"
    )

    return performance


data = load_data()
model, features = load_model()
performance = load_performance()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏦 Loan Approval Predictor")

st.sidebar.markdown(
    """
    ### Navigation

    Explore loan application analytics,
    generate loan approval predictions,
    and compare machine learning models.
    """
)

selected_section = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Loan Analysis",
        "Prediction",
        "Model Performance"
    ]
)


# ============================================================
# HEADER
# ============================================================

st.title("🏦 Loan Approval Prediction System")

st.markdown(
    """
    ### Financial Analysis & Loan Approval Prediction

    This application analyzes applicant information
    and uses Machine Learning classification models
    to predict whether a loan application is likely
    to be **Approved** or **Rejected**.

    **Best Model:** Machine Learning Classification
    """
)

st.divider()


# ============================================================
# OVERVIEW
# ============================================================

if selected_section == "Overview":

    st.subheader("📊 Loan Application Overview")

    total_applications = len(data)

    approved_count = (
        data["Loan_Status"]
        .eq("Approved")
        .sum()
    )

    rejected_count = (
        data["Loan_Status"]
        .eq("Rejected")
        .sum()
    )

    approval_rate = (
        approved_count /
        total_applications
    ) * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Applications",
            f"{total_applications:,}"
        )

    with col2:

        st.metric(
            "Approved",
            f"{approved_count:,}"
        )

    with col3:

        st.metric(
            "Rejected",
            f"{rejected_count:,}"
        )

    with col4:

        st.metric(
            "Approval Rate",
            f"{approval_rate:.2f}%"
        )

    # --------------------------------------------------------
    # Approval distribution
    # --------------------------------------------------------

    st.subheader("Loan Approval Distribution")

    status_counts = (
        data["Loan_Status"]
        .value_counts()
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        status_counts.index,
        status_counts.values
    )

    ax.set_title(
        "Loan Approval Distribution"
    )

    ax.set_xlabel(
        "Loan Status"
    )

    ax.set_ylabel(
        "Number of Applications"
    )

    plt.tight_layout()

    st.pyplot(fig)

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.write(
            f"**Records:** {len(data):,}"
        )

    with col2:

        st.write(
            f"**Model Features:** {len(features)}"
        )

    with col3:

        st.write(
            "**Target:** Loan_Status"
        )


# ============================================================
# LOAN ANALYSIS
# ============================================================

elif selected_section == "Loan Analysis":

    st.subheader("📊 Loan Application Analysis")

    # --------------------------------------------------------
    # Applicant Income
    # --------------------------------------------------------

    st.markdown(
        "### Applicant Income Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.hist(
        data["Applicant_Income"].dropna(),
        bins=30
    )

    ax.set_title(
        "Applicant Income Distribution"
    )

    ax.set_xlabel(
        "Applicant Income"
    )

    ax.set_ylabel(
        "Number of Applicants"
    )

    plt.tight_layout()

    st.pyplot(fig)

    # --------------------------------------------------------
    # Loan Amount
    # --------------------------------------------------------

    st.markdown(
        "### Loan Amount Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.hist(
        data["Loan_Amount"].dropna(),
        bins=30
    )

    ax.set_title(
        "Loan Amount Distribution"
    )

    ax.set_xlabel(
        "Loan Amount"
    )

    ax.set_ylabel(
        "Number of Applications"
    )

    plt.tight_layout()

    st.pyplot(fig)

    # --------------------------------------------------------
    # Credit History
    # --------------------------------------------------------

    st.markdown(
        "### Credit History vs Loan Approval"
    )

    credit_table = pd.crosstab(
        data["Credit_History"],
        data["Loan_Status"]
    )

    st.dataframe(
        credit_table,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Education
    # --------------------------------------------------------

    st.markdown(
        "### Education vs Loan Approval"
    )

    education_table = pd.crosstab(
        data["Education"],
        data["Loan_Status"]
    )

    st.dataframe(
        education_table,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Property Area
    # --------------------------------------------------------

    st.markdown(
        "### Property Area vs Loan Approval"
    )

    property_table = pd.crosstab(
        data["Property_Area"],
        data["Loan_Status"]
    )

    st.dataframe(
        property_table,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Employment Status
    # --------------------------------------------------------

    st.markdown(
        "### Employment Status vs Loan Approval"
    )

    employment_table = pd.crosstab(
        data["Employment_Status"],
        data["Loan_Status"]
    )

    st.dataframe(
        employment_table,
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

elif selected_section == "Prediction":

    st.subheader(
        "🤖 Loan Approval Prediction"
    )

    st.info(
        "Enter applicant information below to "
        "generate a loan approval prediction."
    )

    col1, col2 = st.columns(2)

    # ========================================================
    # LEFT COLUMN
    # ========================================================

    with col1:

        gender = st.selectbox(
            "Gender",
            data["Gender"]
            .dropna()
            .unique()
            .tolist()
        )

        married = st.selectbox(
            "Marital Status",
            data["Married"]
            .dropna()
            .unique()
            .tolist()
        )

        dependents = st.number_input(
            "Dependents",
            min_value=0.0,
            max_value=float(
                data["Dependents"].max()
            ),
            value=0.0,
            step=1.0
        )

        education = st.selectbox(
            "Education",
            data["Education"]
            .dropna()
            .unique()
            .tolist()
        )

        employment_status = st.selectbox(
            "Employment Status",
            data["Employment_Status"]
            .dropna()
            .unique()
            .tolist()
        )

        property_area = st.selectbox(
            "Property Area",
            data["Property_Area"]
            .dropna()
            .unique()
            .tolist()
        )

        credit_history = st.selectbox(
            "Credit History",
            sorted(
                data["Credit_History"]
                .dropna()
                .unique()
                .tolist()
            )
        )

    # ========================================================
    # RIGHT COLUMN
    # ========================================================

    with col2:

        age = st.number_input(
            "Age",
            min_value=int(
                data["Age"].min()
            ),
            max_value=int(
                data["Age"].max()
            ),
            value=int(
                data["Age"].median()
            ),
            step=1
        )

        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0.0,
            value=float(
                data["Applicant_Income"].median()
            ),
            step=1000.0
        )

        coapplicant_income = st.number_input(
            "Coapplicant Income",
            min_value=0.0,
            value=float(
                data["Coapplicant_Income"].median()
            ),
            step=1000.0
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0.0,
            value=float(
                data["Loan_Amount"].median()
            ),
            step=1000.0
        )

        loan_term = st.number_input(
            "Loan Term (Months)",
            min_value=1.0,
            value=float(
                data["Loan_Term"].median()
            ),
            step=12.0
        )


    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    total_income = (
        applicant_income +
        coapplicant_income
    )

    if total_income > 0:

        loan_to_income_ratio = (
            loan_amount /
            total_income
        )

    else:

        loan_to_income_ratio = 0


    # ========================================================
    # CREATE INPUT DATA
    # ========================================================

    input_data = pd.DataFrame({

        "Gender": [gender],

        "Married": [married],

        "Dependents": [dependents],

        "Education": [education],

        "Employment_Status": [
            employment_status
        ],

        "Applicant_Income": [
            applicant_income
        ],

        "Coapplicant_Income": [
            coapplicant_income
        ],

        "Loan_Amount": [
            loan_amount
        ],

        "Loan_Term": [
            loan_term
        ],

        "Credit_History": [
            credit_history
        ],

        "Property_Area": [
            property_area
        ],

        "Age": [
            age
        ],

        "Total_Income": [
            total_income
        ],

        "Loan_to_Income_Ratio": [
            loan_to_income_ratio
        ]
    })


    # ========================================================
    # ENSURE MODEL FEATURE ORDER
    # ========================================================

    input_data = input_data[
        features
    ]


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    if st.button(
        "🔍 Predict Loan Approval",
        use_container_width=True
    ):

        prediction = model.predict(
            input_data
        )[0]

        st.divider()

        if prediction == "Approved":

            st.success(
                "✅ Loan Application Likely Approved"
            )

        else:

            st.error(
                "❌ Loan Application Likely Rejected"
            )


        # ====================================================
        # PROBABILITY
        # ====================================================

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = (
                model.predict_proba(
                    input_data
                )[0]
            )

            classes = model.classes_

            probability_df = pd.DataFrame({

                "Loan Status": classes,

                "Probability": probabilities
            })

            probability_df[
                "Probability"
            ] = (
                probability_df[
                    "Probability"
                ] * 100
            )

            probability_df[
                "Probability"
            ] = probability_df[
                "Probability"
            ].round(2)

            st.subheader(
                "Prediction Probability"
            )

            st.dataframe(
                probability_df,
                use_container_width=True
            )

            approved_rows = (
                probability_df[
                    probability_df[
                        "Loan Status"
                    ] == "Approved"
                ]
            )

            if not approved_rows.empty:

                approval_probability = (
                    approved_rows[
                        "Probability"
                    ].iloc[0]
                )

                st.metric(
                    "Approval Probability",
                    f"{approval_probability:.2f}%"
                )


    st.warning(
        "This prediction is for educational purposes "
        "only and should not be considered financial advice."
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif selected_section == "Model Performance":

    st.subheader(
        "📊 Machine Learning Model Performance"
    )

    st.dataframe(
        performance,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    best_model = performance.iloc[0]

    st.success(
        f"🏆 Best Model: {best_model['Model']}"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Accuracy",
            f"{best_model['Accuracy']:.4f}"
        )

    with col2:

        st.metric(
            "Precision",
            f"{best_model['Precision']:.4f}"
        )

    with col3:

        st.metric(
            "Recall",
            f"{best_model['Recall']:.4f}"
        )

    with col4:

        st.metric(
            "F1 Score",
            f"{best_model['F1 Score']:.4f}"
        )

    st.markdown(
        """
        ### Evaluation Metrics

        **Accuracy:** Overall percentage of correct predictions.

        **Precision:** Measures the correctness of positive predictions.

        **Recall:** Measures how many actual positive cases are identified.

        **F1 Score:** Harmonic balance between precision and recall.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Loan Approval Prediction System | "
    "Data Science Internship – Phase 2 | "
    "Educational Project"
)