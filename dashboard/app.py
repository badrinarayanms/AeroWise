import sklearn
print(sklearn.__version__)
import streamlit as st
import joblib
import pandas as pd

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="AeroWise",
    page_icon="✈️",
    layout="wide"
)

# -------------------------
# Load Model
# -------------------------
model = joblib.load(
    "../models/demand_model.pkl"
)

preprocessor = joblib.load(
    "../models/preprocessor.pkl"
)
routes = joblib.load(
    "../models/routes.pkl"
)

# -------------------------
# Flight Recommendation Logic
# -------------------------
def recommend_flights(
    predicted_passengers,
    aircraft_capacity=180,
    occupancy_target=0.85
):

    seats_needed = (
        predicted_passengers /
        occupancy_target
    )

    flights_needed = (
        seats_needed /
        aircraft_capacity
    )

    return round(flights_needed)


# -------------------------
# UI
# -------------------------
st.title("✈️ AeroWise")
st.subheader(
    "Airline Demand Forecasting & Schedule Optimization Platform"
)

st.markdown("---")

st.header("Demand Forecast")

# Inputs
year = st.selectbox(
    "Year",
    [2024, 2025, 2026]
)

quarter = st.selectbox(
    "Quarter",
    [1, 2, 3, 4]
)

distance = st.number_input(
    "Distance (Miles)",
    min_value=0
)

fare = st.number_input(
    "Fare ($)",
    min_value=0.0
)

route = st.selectbox(
    "Select Route",
    sorted(routes)
)

# Predict Button
if st.button("Predict Demand"):

    input_df = pd.DataFrame({
        "Year": [year],
        "Quarter": [quarter],
        "Distance": [distance],
        "Fare": [fare],
        "route": [route]
    })

    processed_input = (
        preprocessor.transform(input_df)
    )

    predicted_passengers = (
        model.predict(processed_input)[0]
    )

    recommended_flights = (
        recommend_flights(
            predicted_passengers
        )
    )

    st.success(
        f"Predicted Passengers: "
        f"{round(predicted_passengers)}"
    )

    st.info(
        f"Recommended Flights/Day: "
        f"{recommended_flights}"
    )