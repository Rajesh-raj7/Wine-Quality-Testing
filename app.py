
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Wine Intelligence | AI Analytics",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PREMIUM DARK THEME
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');

:root {
    --wine: #761D35;
    --wine-light: #A33B53;
    --gold: #D7B46A;
    --cream: #F3EBDD;
    --muted: #A9A0A0;
    --panel: #191719;
    --background: #100E10;
}

.stApp {
    background: var(--background);
    color: var(--cream);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: #171316;
    border-right: 1px solid #35242B;
}

[data-testid="stSidebar"] * {
    color: var(--cream);
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--cream) !important;
}

h1 {
    font-size: 2.7rem !important;
}

p, label, span {
    font-family: 'DM Sans', sans-serif;
}

.hero {
    background: linear-gradient(115deg, #32121F, #671D35 58%, #25131B);
    border: 1px solid #854052;
    border-radius: 18px;
    padding: 34px;
    margin: 10px 0 26px 0;
}

.hero h1 {
    color: #F7EEDC !important;
    margin-bottom: 8px;
}

.hero p {
    color: #E1C9C7;
    font-size: 1rem;
}

.eyebrow {
    color: #D7B46A;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
}

.metric-card {
    background: linear-gradient(145deg, #211B20, #171316);
    border: 1px solid #45303A;
    border-radius: 14px;
    padding: 20px;
    min-height: 112px;
}

.metric-label {
    color: #B6A9AB;
    font-size: 0.8rem;
    margin-bottom: 10px;
}

.metric-value {
    color: #E5C47F;
    font-size: 1.8rem;
    font-weight: 700;
}

.metric-note {
    color: #9C8E92;
    font-size: 0.75rem;
    margin-top: 5px;
}

.section-label {
    color: #D7B46A;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.panel {
    background: #191619;
    border: 1px solid #392B32;
    border-radius: 15px;
    padding: 22px;
}

.result-card {
    background: linear-gradient(135deg, #421525, #24151D);
    border: 1px solid #A56C54;
    border-radius: 18px;
    padding: 28px;
    text-align: center;
}

.result-number {
    font-size: 4rem;
    font-weight: 700;
    color: #E4C17B;
    font-family: 'Playfair Display', serif;
}

div.stButton > button {
    background: linear-gradient(90deg, #8C2945, #B04A5F);
    color: white;
    border: 1px solid #C17A79;
    border-radius: 9px;
    padding: 10px 22px;
    font-weight: 700;
    width: 100%;
}

div.stButton > button:hover {
    background: #A9445B;
    border-color: #E2BD7A;
    color: white;
}

div[data-baseweb="input"] input,
div[data-baseweb="select"] > div {
    background: #211C20;
    color: #F3EBDD;
    border-color: #57414A;
}

.stTabs [data-baseweb="tab"] {
    color: #C9BFC0;
}

hr {
    border-color: #3B2C33;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# DATASET
# =========================================================

@st.cache_data
def load_data():
    df = pd.read_csv("winequality-red.csv")

    if "quality" not in df.columns:
        raise ValueError(
            "The dataset must contain a 'quality' column."
        )

    if df.isnull().values.any():
        df = df.dropna()

    return df


# =========================================================
# MACHINE LEARNING MODELS
# =========================================================

def create_models():

    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=0
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=0
        ),

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000))
        ]),

        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="linear"))
        ]),

        "Gaussian Naive Bayes": GaussianNB(),

        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ])
    }


@st.cache_resource
def train_models(data):

    X = data.drop(columns=["quality"])
    y = data["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=0,
        stratify=y
    )

    models = create_models()
    results = []
    predictions = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        predictions[name] = pred

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(
                y_test, pred, average="weighted",
                zero_division=0
            ),
            "Recall": recall_score(
                y_test, pred, average="weighted",
                zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, pred, average="weighted",
                zero_division=0
            )
        })

        # Retrain the model on the complete dataset
        # for the final prediction application.
        model.fit(X, y)

    results_df = pd.DataFrame(results)

    return (
        models,
        results_df,
        X_test,
        y_test,
        predictions,
        X.columns.tolist()
    )


# =========================================================
# LOAD APPLICATION DATA
# =========================================================

try:
    data = load_data()

    (
        models,
        model_results,
        X_test,
        y_test,
        predictions,
        features
    ) = train_models(data)

except Exception as error:
    st.error(f"Application initialization failed: {error}")
    st.stop()


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown(
        "<h2 style='color:#D7B46A;'>🍷 Wine Intelligence</h2>",
        unsafe_allow_html=True
    )

    st.caption("ARTIFICIAL INTELLIGENCE & ANALYTICS")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Wine Prediction",
            "Dataset Insights",
            "About the Model"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown(
        """
        <p style="color:#D7B46A;font-weight:700;">
        PROJECT INFORMATION
        </p>
        <p style="font-size:13px;color:#B8AEB0;">
        Red Wine Quality Classification<br>
        Machine Learning Application<br>
        Built with Python & Streamlit
        </p>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.caption("Portfolio Project | 2026")


# =========================================================
# REUSABLE METRIC CARD
# =========================================================

def metric_card(label, value, note=""):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# PAGE 1: OVERVIEW
# =========================================================

if page == "Overview":

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">AI POWERED WINE ANALYTICS</div>
            <h1>Wine Intelligence</h1>
            <p>
            Discover the science behind wine quality.
            Explore chemical characteristics, compare
            machine learning algorithms and predict
            wine quality using artificial intelligence.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-label">PROJECT OVERVIEW</div>',
        unsafe_allow_html=True
    )

    st.write("")

    total_samples = len(data)
    total_features = len(features)
    quality_classes = data["quality"].nunique()
    best_model = model_results.loc[
        model_results["Accuracy"].idxmax()
    ]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("WINE SAMPLES", f"{total_samples:,}")

    with c2:
        metric_card("CHEMICAL FEATURES", total_features)

    with c3:
        metric_card("QUALITY CLASSES", quality_classes)

    with c4:
        metric_card(
            "TOP TEST ACCURACY",
            f"{best_model['Accuracy'] * 100:.2f}%",
            best_model["Model"]
        )

    st.write("")
    st.markdown("### Understanding Wine Quality")

    left, right = st.columns([1.2, 1])

    with left:

        st.markdown(
            """
            <div class="panel">
                <h3>About the Project</h3>
                <p style="color:#C3B8BA;">
                This project uses physicochemical properties
                of red wine to predict its quality score.
                </p>
                <p style="color:#C3B8BA;">
                Six classification algorithms are trained
                and evaluated using a test dataset.
                </p>
                <p style="color:#D7B46A;">
                Objective: Data-driven wine quality prediction.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:

        st.markdown(
            """
            <div class="panel">
                <h3>Machine Learning Pipeline</h3>
                <p style="color:#D7B46A;">01 — Dataset loading</p>
                <p style="color:#D7B46A;">02 — Feature extraction</p>
                <p style="color:#D7B46A;">03 — Model training</p>
                <p style="color:#D7B46A;">04 — Model evaluation</p>
                <p style="color:#D7B46A;">05 — Quality prediction</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")
    st.markdown("### Wine Quality Distribution")

    quality_counts = (
        data["quality"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        quality_counts,
        color="#C18B54"
    )

    st.caption(
        "Distribution of quality scores in the original dataset."
    )


# =========================================================
# PAGE 2: WINE PREDICTION
# =========================================================

elif page == "Wine Prediction":

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">INTELLIGENT PREDICTION ENGINE</div>
            <h1>Wine Quality Prediction</h1>
            <p>
            Enter the chemical properties of your wine
            and let machine learning estimate its quality.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Configure Prediction")

    selected_model = st.selectbox(
        "Choose Machine Learning Algorithm",
        list(models.keys()),
        index=0
    )

    st.caption(
        "Random Forest is selected by default. "
        "You can compare all six algorithms."
    )

    st.divider()

    st.markdown("### Chemical Properties")

    st.write(
        "Enter values for the 11 physicochemical features."
    )

    user_input = {}

    with st.form("wine_prediction_form"):

        col1, col2 = st.columns(2)

        for index, feature in enumerate(features):

            minimum = float(data[feature].min())
            maximum = float(data[feature].max())
            default = float(data[feature].median())

            step = max((maximum - minimum) / 100, 0.001)

            target_col = col1 if index % 2 == 0 else col2

            with target_col:

                user_input[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    step=step,
                    format="%.4f",
                    help=(
                        f"Observed dataset range: "
                        f"{minimum:.4f} to {maximum:.4f}"
                    )
                )

        st.write("")

        submitted = st.form_submit_button(
            "🍷  PREDICT WINE QUALITY",
            use_container_width=True
        )

    if submitted:

        input_df = pd.DataFrame(
            [user_input],
            columns=features
        )

        selected_estimator = models[selected_model]

        prediction = selected_estimator.predict(input_df)[0]

        st.write("")
        st.markdown("### Prediction Result")

        st.markdown(
            f"""
            <div class="result-card">
                <div class="eyebrow">PREDICTED QUALITY SCORE</div>
                <div class="result-number">{int(prediction)}</div>
                <p style="color:#E5D4D0;">
                    Model: {selected_model}
                </p>
                <p style="color:#D7B46A;">
                    Prediction generated successfully
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info(
            "This is a machine learning estimate, not a "
            "laboratory measurement or professional wine assessment."
        )

        st.session_state["last_prediction"] = int(prediction)
        st.session_state["last_model"] = selected_model


# =========================================================
# PAGE 3: DATASET INSIGHTS
# =========================================================

elif page == "Dataset Insights":

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">EXPLORATORY DATA ANALYSIS</div>
            <h1>Dataset Insights</h1>
            <p>
            Explore the patterns, distributions and
            relationships hidden inside the red wine dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Dataset Preview")

    st.dataframe(
        data.head(10),
        use_container_width=True,
        hide_index=True
    )

    st.write("")

    st.markdown("### Statistical Summary")

    st.dataframe(
        data.describe().round(3),
        use_container_width=True
    )

    st.divider()

    st.markdown("### Feature Distribution")

    selected_feature = st.selectbox(
        "Select a chemical property",
        features
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    fig.patch.set_facecolor("#191619")
    ax.set_facecolor("#191619")

    sns.histplot(
        data[selected_feature],
        kde=True,
        color="#C18B54",
        ax=ax
    )

    ax.set_title(
        f"Distribution of {selected_feature}",
        color="#F3EBDD"
    )

    ax.set_xlabel(selected_feature, color="#D7B46A")
    ax.set_ylabel("Frequency", color="#D7B46A")

    ax.tick_params(colors="#D7B46A")

    for spine in ax.spines.values():
        spine.set_color("#57414A")

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.markdown("### Chemical Feature Correlation")

    fig, ax = plt.subplots(figsize=(12, 8))

    fig.patch.set_facecolor("#191619")

    sns.heatmap(
        data.corr(numeric_only=True),
        cmap="RdYlBu_r",
        annot=True,
        fmt=".2f",
        linewidths=0.3,
        ax=ax
    )

    ax.set_title(
        "Feature Correlation Matrix",
        color="#F3EBDD"
    )

    ax.tick_params(colors="#D7B46A")

    st.pyplot(fig)
    plt.close(fig)


# =========================================================
# PAGE 4: ABOUT THE MODEL
# =========================================================

elif page == "About the Model":

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">MODEL PERFORMANCE & TECHNOLOGY</div>
            <h1>About the Models</h1>
            <p>
            Six supervised machine learning algorithms
            are evaluated to understand their performance
            in wine quality classification.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Model Performance Comparison")

    display_results = model_results.copy()

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]:
        display_results[column] = (
            display_results[column] * 100
        ).round(2)

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Metrics are calculated on a held-out test set. "
        "They are recalculated by this application and "
        "may differ from the original README results."
    )

    st.divider()

    st.markdown("### Accuracy Comparison")

    accuracy_chart = (
        model_results
        .set_index("Model")[["Accuracy"]]
        .sort_values("Accuracy", ascending=False)
        * 100
    )

    st.bar_chart(
        accuracy_chart,
        color="#C18B54"
    )

    st.divider()

    st.markdown("### Random Forest Feature Importance")

    rf_model = models["Random Forest"]

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": rf_model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    st.bar_chart(
        importance.set_index("Feature"),
        color="#C18B54"
    )

    st.caption(
        "Feature importance represents the Random Forest "
        "model's relative contribution of each feature. "
        "It does not establish causation."
    )

    st.divider()

    st.markdown("### Algorithms Used")

    algorithm_details = {
        "Random Forest":
            "An ensemble of decision trees that combines "
            "multiple predictions.",

        "Decision Tree":
            "A tree-based classifier that makes decisions "
            "using feature-based rules.",

        "Logistic Regression":
            "A classification algorithm that models "
            "class probabilities.",

        "Support Vector Machine":
            "A classifier that finds decision boundaries "
            "between different classes.",

        "Gaussian Naive Bayes":
            "A probabilistic classifier based on "
            "Bayes' theorem.",

        "K-Nearest Neighbors":
            "A distance-based algorithm that classifies "
            "a sample using its nearest neighbors."
    }

    for name, description in algorithm_details.items():

        with st.expander(name):

            st.write(description)

    st.divider()

    st.markdown("### Technology Stack")

    t1, t2, t3 = st.columns(3)

    with t1:
        metric_card("LANGUAGE", "Python")

    with t2:
        metric_card("ML FRAMEWORK", "Scikit-learn")

    with t3:
        metric_card("WEB FRAMEWORK", "Streamlit")


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;padding:20px 0;">
        <p style="color:#D7B46A;font-size:17px;">
            🍷 WINE INTELLIGENCE
        </p>
        <p style="color:#96898D;font-size:12px;">
            AI-Powered Wine Quality Classification
        </p>
        <p style="color:#796D70;font-size:11px;">
            Built with Python • Pandas • Scikit-learn • Streamlit
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
