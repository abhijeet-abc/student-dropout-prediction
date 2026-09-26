import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------------------------------
# LOAD MODEL FILES
# ---------------------------------------------------------

MODEL_PATH = "models/random_forest.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURE_PATH = "models/feature_columns.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    model_loaded = True

except Exception as e:
    model_loaded = False
    st.error(f"Error loading model files: {e}")


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("🎓 Student Dropout System")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dataset Overview",
        "🔮 Student Prediction",
        "📈 Risk Analysis",
        "💡 Intervention"
    ]
)


# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.title("🎓 Student Dropout Prediction System")

    st.subheader(
        "Machine Learning Based Student Risk Prediction"
    )

    st.write(
        """
        This application predicts student dropout risk using
        Machine Learning techniques.
        """
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Machine Learning Model",
            "Random Forest"
        )

    with col2:
        st.metric(
            "Features",
            len(feature_columns) if model_loaded else "N/A"
        )

    with col3:
        st.metric(
            "System",
            "Prediction"
        )

    st.markdown("---")

    st.info(
        """
        **Project Workflow**

        Student Data → Data Processing → Temporal Features →
        Machine Learning → Dropout Probability →
        Risk Category → Intervention Recommendation
        """
    )

    st.success(
        "Model files are connected successfully."
        if model_loaded
        else "Model files could not be loaded."
    )


# =========================================================
# DATASET OVERVIEW
# =========================================================

elif page == "📊 Dataset Overview":

    st.title("📊 Dataset Overview")

    DATA_PATH = "data/student_dropout_temporal_dataset.csv"

    if os.path.exists(DATA_PATH):

        df = pd.read_csv(DATA_PATH)

        st.write(
            f"Dataset contains **{df.shape[0]} students** "
            f"and **{df.shape[1]} columns**."
        )

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Students",
                df.shape[0]
            )

        with col2:
            st.metric(
                "Features",
                df.shape[1]
            )

        with col3:
            if "Target" in df.columns:
                st.metric(
                    "Target Classes",
                    df["Target"].nunique()
                )
            else:
                st.metric(
                    "Target Classes",
                    "N/A"
                )

        st.markdown("---")

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

        st.subheader("Dataset Information")

        st.write(
            df.describe(include="all")
        )

    else:

        st.error(
            "Dataset file not found. "
            "Please check the data folder."
        )


# =========================================================
# STUDENT PREDICTION
# =========================================================

elif page == "🔮 Student Prediction":

    st.title("🔮 Student Dropout Prediction")

    st.write(
        "Enter the student's information below."
    )

    if not model_loaded:

        st.error(
            "Model files are not loaded."
        )

    else:

        student_values = {}

        st.subheader("Student Features")

        # Create input fields
        for feature in feature_columns:

            student_values[feature] = st.number_input(
                feature,
                value=0.0
            )

        st.markdown("---")

        if st.button(
            "🔍 Predict Dropout Risk",
            type="primary"
        ):

            try:

                # Create dataframe
                input_df = pd.DataFrame(
                    [student_values]
                )

                # Ensure correct column order
                input_df = input_df[
                    feature_columns
                ]

                # Prediction
                prediction = model.predict(
                    input_df
                )

                probabilities = model.predict_proba(
                    input_df
                )

                # Find Dropout class
                dropout_label = label_encoder.transform(
                    ["Dropout"]
                )[0]

                dropout_index = list(
                    model.classes_
                ).index(dropout_label)

                dropout_probability = probabilities[
                    0,
                    dropout_index
                ]

                risk_score = (
                    dropout_probability * 100
                )

                # Risk category
                if risk_score < 30:

                    risk_category = "Low Risk"

                elif risk_score < 60:

                    risk_category = "Moderate Risk"

                else:

                    risk_category = "High Risk"

                # Predicted class
                predicted_class = label_encoder.inverse_transform(
                    prediction
                )[0]

                st.markdown("---")

                st.subheader(
                    "Prediction Result"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Predicted Outcome",
                        predicted_class
                    )

                with col2:

                    st.metric(
                        "Dropout Probability",
                        f"{risk_score:.2f}%"
                    )

                with col3:

                    st.metric(
                        "Risk Category",
                        risk_category
                    )

                st.markdown("---")

                if risk_category == "Low Risk":

                    st.success(
                        "🟢 Student has Low Dropout Risk."
                    )

                elif risk_category == "Moderate Risk":

                    st.warning(
                        "🟡 Student has Moderate Dropout Risk."
                    )

                else:

                    st.error(
                        "🔴 Student has High Dropout Risk."
                    )

            except Exception as e:

                st.error(
                    f"Prediction error: {e}"
                )


# =========================================================
# RISK ANALYSIS
# =========================================================

elif page == "📈 Risk Analysis":

    st.title("📈 Student Risk Analysis")

    DATA_PATH = "data/student_dropout_risk_results.csv"

    if os.path.exists(DATA_PATH):

        risk_df = pd.read_csv(
            DATA_PATH
        )

        st.subheader(
            "Risk Prediction Results"
        )

        st.dataframe(
            risk_df.head(50),
            use_container_width=True
        )

        st.markdown("---")

        # Try to find risk category column
        risk_column = None

        possible_columns = [
            "Risk Category",
            "Risk_Category",
            "Risk"
        ]

        for col in possible_columns:

            if col in risk_df.columns:

                risk_column = col
                break

        if risk_column:

            st.subheader(
                "Risk Distribution"
            )

            risk_counts = (
                risk_df[risk_column]
                .value_counts()
            )

            st.bar_chart(
                risk_counts
            )

        else:

            st.info(
                "Risk category column was not found."
            )

    else:

        st.error(
            "Risk results file not found."
        )


# =========================================================
# INTERVENTION
# =========================================================

elif page == "💡 Intervention":

    st.title("💡 Intervention Recommendation")

    st.write(
        """
        Recommended interventions based on the student's
        predicted dropout risk.
        """
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.success("🟢 LOW RISK")

        st.write(
            """
            • Regular academic monitoring

            • Encourage student engagement

            • Continue normal mentoring
            """
        )

    with col2:

        st.warning("🟡 MODERATE RISK")

        st.write(
            """
            • Academic mentoring

            • Attendance monitoring

            • Personalized guidance

            • Regular follow-up
            """
        )

    with col3:

        st.error("🔴 HIGH RISK")

        st.write(
            """
            • Immediate academic counseling

            • Faculty intervention

            • Attendance monitoring

            • Financial/support assessment

            • Frequent follow-up
            """
        )

    st.markdown("---")

    st.info(
        """
        The purpose of this module is to help institutions
        identify students who may require additional support.
        """
    )