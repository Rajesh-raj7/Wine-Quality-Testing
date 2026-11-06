import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide"
)

# Application heading
st.title("🍷 Wine Quality Prediction System")
st.markdown(
    "Predict red wine quality using "
    "Machine Learning (Random Forest Classifier)."
)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("winequality-red.csv")

# Train model
@st.cache_resource
def train_model():
    data = load_data()

    X = data.drop("quality", axis=1)
    y = data["quality"]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    return model

# Main application
try:
    data = load_data()
    model = train_model()

    st.divider()
    st.subheader("Enter Wine Characteristics")

    features = list(data.drop("quality", axis=1).columns)
    user_input = {}

    col1, col2 = st.columns(2)

    for i, feature in enumerate(features):

        minimum = float(data[feature].min())
        maximum = float(data[feature].max())
        default = float(data[feature].median())

        with (col1 if i % 2 == 0 else col2):

            user_input[feature] = st.number_input(
                feature.replace("_", " ").title(),
                min_value=minimum,
                max_value=maximum,
                value=default,
                step=0.01,
                format="%.3f"
            )

    st.divider()

    if st.button("Predict Wine Quality", type="primary"):

        input_data = pd.DataFrame(
            [user_input],
            columns=features
        )

        prediction = model.predict(input_data)[0]

        st.success("Prediction completed!")

        st.metric(
            label="Predicted Wine Quality",
            value=int(prediction)
        )

        st.info(
            f"Predicted wine quality score: {prediction}"
        )

    st.divider()

    st.caption(
        "Model: Random Forest Classifier | "
        "Dataset: Red Wine Quality"
    )

except FileNotFoundError:
    st.error(
        "Dataset not found. Please ensure "
        "winequality-red.csv is in the project folder."
    )

except Exception as e:
    st.error(f"Something went wrong: {e}")
