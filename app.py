import os
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

warnings.filterwarnings("ignore")

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="ParkinVoice AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "parkinsons_model.pkl"
DATA_PATH = "parkinsons.csv"

FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP",
    "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer",
    "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
]

# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #081526 0%, #10243b 100%);
    color: #edf4ff;
}
[data-testid="stSidebar"] {
    background-color: #0b192b;
    border-right: 1px solid #263b52;
}
h1, h2, h3 {
    color: #e9f4ff !important;
}
p, label, .stMarkdown {
    color: #d3e1ef;
}
.hero {
    background: linear-gradient(110deg, #163b59, #155e70);
    padding: 30px;
    border-radius: 20px;
    border: 1px solid #39758b;
    margin-bottom: 22px;
}
.hero h1 {
    font-size: 38px;
    margin-bottom: 8px;
}
.hero p {
    font-size: 16px;
    color: #d8edf5;
}
.metric-card {
    background: #142b42;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #2a4862;
    min-height: 115px;
}
.small-note {
    color: #9fb8ce;
    font-size: 13px;
}
div.stButton > button {
    background: #27b6a3;
    color: #061b29;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    min-height: 45px;
}
div.stButton > button:hover {
    background: #53d6c2;
    color: #061b29;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL AND OPTIONAL DATASET
# --------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found: {MODEL_PATH}"

    try:
        return joblib.load(MODEL_PATH), None
    except Exception as exc:
        return None, (
            "Could not load the saved model. Check that it was "
            "saved with joblib and that scikit-learn versions match. "
            f"Details: {exc}"
        )


@st.cache_data
def load_reference_data():
    if os.path.exists(DATA_PATH):
        try:
            return pd.read_csv(DATA_PATH)
        except Exception:
            return None
    return None


model, model_error = load_model()
reference_df = load_reference_data()

# Use the saved model's feature order whenever available.
if model is not None and hasattr(model, "feature_names_in_"):
    FEATURE_COLUMNS = list(model.feature_names_in_)
elif reference_df is not None:
    FEATURE_COLUMNS = [
        c for c in reference_df.columns
        if c not in ["name", "status", "subject"]
    ]
else:
    FEATURE_COLUMNS = FEATURES

# Keep only numerical measurement columns.
if reference_df is not None:
    valid_features = [
        c for c in FEATURE_COLUMNS
        if c in reference_df.columns
        and pd.api.types.is_numeric_dtype(reference_df[c])
    ]
    if len(valid_features) == len(FEATURE_COLUMNS):
        FEATURE_COLUMNS = valid_features

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 ParkinVoice AI")
    st.caption("VOICE ANALYTICS • MACHINE LEARNING")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Voice Screening",
            "Batch CSV Analysis",
            "Model Insights",
            "Prediction History",
            "About Project"
        ]
    )

    st.divider()
    st.markdown("### System status")

    if model is not None:
        st.success("ML model loaded")
    else:
        st.error("ML model unavailable")

    st.caption("Built with Python, Streamlit and scikit-learn")

# --------------------------------------------------
# COMMON HELPERS
# --------------------------------------------------
def predict_rows(data):
    """Predict using the saved model and the model's feature order."""
    if model is None:
        raise RuntimeError(model_error or "Model unavailable.")

    data = data[FEATURE_COLUMNS].copy()
    data = data.apply(pd.to_numeric, errors="coerce")

    if data.isna().any().any():
        raise ValueError("Please provide valid numeric values for every feature.")

    predictions = model.predict(data)

    probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(data)
        except Exception:
            probabilities = None

    return predictions, probabilities


def prediction_label(value):
    try:
        return "Parkinson's group" if int(value) == 1 else "Control group"
    except (TypeError, ValueError):
        return str(value)


def save_history(source, result):
    st.session_state.history.insert(0, {
        "Time": pd.Timestamp.now().strftime("%d %b %Y, %H:%M"),
        "Source": source,
        "Result": str(result)
    })


def metric_card(title, value, subtitle):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="small-note">{title}</div>
            <h2>{value}</h2>
            <div class="small-note">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_prediction(value, probability=None):
    label = prediction_label(value)

    if str(value) == "1":
        st.warning(
            "The model output belongs to the Parkinson's group "
            "class in the training dataset."
        )
    elif str(value) == "0":
        st.success(
            "The model output belongs to the control group "
            "class in the training dataset."
        )
    else:
        st.info(f"Model output: {label}")

    if probability is not None:
        st.metric("Model class probability", f"{probability:.1f}%")
        st.caption(
            "This is the model's estimated class probability, "
            "not a clinical probability of having the disease."
        )

    st.info(
        "This is an educational research result, not a medical "
        "diagnosis. Consult a qualified healthcare professional "
        "for medical assessment."
    )


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------
if page == "Overview":
    st.markdown("""
    <div class="hero">
        <h1>ParkinVoice AI</h1>
        <p>Explore voice measurements through machine learning.</p>
        <p>
        A research dashboard that applies a trained classifier
        to acoustic features associated with Parkinson's disease.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "INPUT FEATURES",
            len(FEATURE_COLUMNS),
            "Voice measurement variables"
        )

    with c2:
        metric_card(
            "MODEL",
            type(model).__name__ if model is not None else "Unavailable",
            "Loaded from your saved model file"
        )

    with c3:
        metric_card(
            "SESSION ANALYSES",
            len(st.session_state.history),
            "Predictions recorded in this session"
        )

    st.markdown("### How the platform works")

    a, b, c = st.columns(3)

    with a:
        st.markdown("#### 01 · Enter")
        st.write("Provide voice measurements or upload a CSV file.")

    with b:
        st.markdown("#### 02 · Analyze")
        st.write("The saved machine learning model processes the measurements.")

    with c:
        st.markdown("#### 03 · Review")
        st.write("Review the model output and available visualizations.")

    st.markdown("### Dataset preview")

    if reference_df is not None:
        st.dataframe(reference_df.head(8), use_container_width=True)
        st.caption(
            f"Loaded {len(reference_df)} rows from {DATA_PATH}."
        )
    else:
        st.info(
            "Optional: place parkinsons.csv in the project folder "
            "to enable dataset previews and reference statistics."
        )

    if model_error:
        st.error(model_error)

# --------------------------------------------------
# MANUAL VOICE SCREENING
# --------------------------------------------------
elif page == "Voice Screening":
    st.markdown("## Voice Measurement Screening")
    st.write(
        "Enter the numerical acoustic measurements expected by "
        "your trained model. This page does not record or analyze "
        "raw audio."
    )

    if model is None:
        st.error(model_error)
        st.stop()

    with st.form("voice_measurement_form"):
        st.markdown("### Acoustic measurements")

        if reference_df is not None:
            st.caption(
                "Suggested slider ranges are based on the reference "
                "dataset where possible. Values can be edited."
            )
        else:
            st.caption(
                "Enter values using the units and scale used during "
                "model training."
            )

        entered = {}

        # Organize inputs into three columns.
        cols = st.columns(3)

        for i, feature in enumerate(FEATURE_COLUMNS):
            col = cols[i % 3]

            with col:
                default = 0.0
                minimum = 0.0
                maximum = 1.0
                step = 0.001

                if reference_df is not None and feature in reference_df.columns:
                    series = pd.to_numeric(
                        reference_df[feature], errors="coerce"
                    ).dropna()

                    if not series.empty:
                        minimum = float(series.min())
                        maximum = float(series.max())
                        default = float(series.median())

                        if minimum == maximum:
                            minimum -= 0.001
                            maximum += 0.001

                        step = max((maximum - minimum) / 1000, 0.000001)

                # Use number_input so users can enter precise measurements.
                entered[feature] = st.number_input(
                    feature,
                    min_value=float(minimum),
                    max_value=float(maximum),
                    value=float(np.clip(default, minimum, maximum)),
                    step=float(step),
                    format="%.6f",
                    help="Use the same feature units as your training dataset."
                )

        submitted = st.form_submit_button(
            "Analyze Measurements",
            use_container_width=True
        )

    if submitted:
        try:
            input_df = pd.DataFrame([entered], columns=FEATURE_COLUMNS)
            preds, probs = predict_rows(input_df)

            pred = preds[0]
            probability = None

            if probs is not None and hasattr(model, "classes_"):
                class_list = list(model.classes_)
                if pred in class_list:
                    probability = float(
                        probs[0, class_list.index(pred)] * 100
                    )

            st.markdown("### Screening result")
            show_prediction(pred, probability)

            st.markdown("### Submitted measurements")
            st.dataframe(input_df, use_container_width=True)

            save_history("Manual measurements", prediction_label(pred))

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

# --------------------------------------------------
# BATCH CSV ANALYSIS
# --------------------------------------------------
elif page == "Batch CSV Analysis":
    st.markdown("## Batch CSV Analysis")
    st.write(
        "Upload a CSV containing the same numerical feature columns "
        "used to train your model."
    )

    if model is None:
        st.error(model_error)
        st.stop()

    uploaded = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            st.markdown("### Uploaded data preview")
            st.dataframe(batch_df.head(10), use_container_width=True)

            missing = [
                c for c in FEATURE_COLUMNS if c not in batch_df.columns
            ]

            if missing:
                st.error(
                    "The CSV is missing required feature columns: "
                    + ", ".join(missing)
                )
            else:
                if batch_df.empty:
                    st.warning("The uploaded CSV contains no rows.")
                else:
                    if batch_df[FEATURE_COLUMNS].isna().any().any():
                        st.warning(
                            "Missing values were found. Clean the CSV "
                            "before making predictions."
                        )
                    else:
                        if st.button(
                            "Analyze CSV",
                            use_container_width=True
                        ):
                            preds, probs = predict_rows(batch_df)

                            results_df = batch_df.copy()
                            results_df["Model_Prediction"] = [
                                prediction_label(p) for p in preds
                            ]

                            if probs is not None and hasattr(model, "classes_"):
                                class_list = list(model.classes_)
                                results_df["Model_Class_Probability_%"] = [
                                    round(float(
                                        probs[i, class_list.index(preds[i])] * 100
                                    ), 2)
                                    for i in range(len(preds))
                                ]

                            st.markdown("### Analysis results")
                            st.dataframe(
                                results_df,
                                use_container_width=True
                            )

                            counts = results_df[
                                "Model_Prediction"
                            ].value_counts().reset_index()
                            counts.columns = ["Class", "Count"]

                            fig = px.bar(
                                counts,
                                x="Class",
                                y="Count",
                                color="Class",
                                title="Predicted class distribution",
                                template="plotly_dark"
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            st.download_button(
                                "Download results CSV",
                                data=results_df.to_csv(index=False).encode("utf-8"),
                                file_name="parkinvoice_analysis.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                            save_history(
                                "Batch CSV",
                                f"{len(results_df)} rows analyzed"
                            )

        except Exception as exc:
            st.error(f"Could not process the uploaded CSV: {exc}")

    st.caption(
        "If your CSV contains a 'status' column, it is not used as "
        "an input feature. This page produces predictions, not an "
        "independent evaluation of model accuracy."
    )

# --------------------------------------------------
# MODEL INSIGHTS
# --------------------------------------------------
elif page == "Model Insights":
    st.markdown("## Model Insights")
    st.write(
        "Explore the model configuration and the distribution of "
        "features in the optional reference dataset."
    )

    if model is not None:
        st.markdown("### Model information")

        info = {
            "Model type": type(model).__name__,
            "Number of input features": getattr(
                model, "n_features_in_", len(FEATURE_COLUMNS)
            ),
            "Expected feature names": len(FEATURE_COLUMNS),
            "Classes": str(getattr(model, "classes_", "Not exposed"))
        }

        st.json(info)

        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "Feature": FEATURE_COLUMNS,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)

            st.markdown("### Feature importance")

            fig = px.bar(
                importance_df.head(12).sort_values("Importance"),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Most influential features in the trained model",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Feature importance describes how the fitted model "
                "uses its inputs. It does not establish causation "
                "or clinical importance."
            )

    if reference_df is not None:
        numeric_cols = [
            c for c in FEATURE_COLUMNS
            if c in reference_df.columns
        ]

        if numeric_cols:
            selected = st.selectbox(
                "Select a feature to explore",
                numeric_cols
            )

            fig = px.histogram(
                reference_df,
                x=selected,
                color="status" if "status" in reference_df.columns else None,
                marginal="box",
                title=f"Distribution of {selected}",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Correlation heatmap")

            corr = reference_df[numeric_cols].corr()

            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="Teal",
                title="Correlation between voice measurements"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Add parkinsons.csv to your project folder to enable "
            "reference-data visualizations."
        )

# --------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------
elif page == "Prediction History":
    st.markdown("## Prediction History")
    st.write(
        "History is stored only in the current Streamlit session. "
        "It is not a permanent database."
    )

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)

        st.download_button(
            "Download history",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="parkinvoice_history.csv",
            mime="text/csv"
        )

        if st.button("Clear session history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No analyses have been recorded in this session yet.")

# --------------------------------------------------
# ABOUT
# --------------------------------------------------
elif page == "About Project":
    st.markdown("## About ParkinVoice AI")

    st.markdown("""
    **Project title:** ParkinVoice AI — Voice-Based Parkinson's
    Disease Screening Using Machine Learning.

    **Objective:** Explore how acoustic voice measurements can be
    used as input to a machine learning classifier.

    **Input:** Numerical voice measurements from a Parkinson's
    dataset, such as frequency, jitter, shimmer, noise, and other
    acoustic descriptors.

    **Machine learning:** The saved classifier is loaded from
    `parkinsons_model.pkl`. The interface uses the input feature
    order expected by that model.

    **Deployment:** Streamlit application suitable for deployment
    on Azure App Service.
    """)

    st.warning(
        "This application is a student research prototype. It is "
        "not validated for clinical use and must not be used to "
        "diagnose, rule out, or make treatment decisions about "
        "Parkinson's disease."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "ParkinVoice AI • Academic ML project • "
    "Research use only — not a medical diagnostic tool"
)