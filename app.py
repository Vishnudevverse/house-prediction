import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="House Price Predictor", page_icon="🏡", layout="wide")

if 'comparison_list' not in st.session_state:
    st.session_state.comparison_list = []

@st.cache_resource
def load_assets():
    # Loading from the new clean directory structure
    model = joblib.load(os.path.join('artifacts', 'house_model.pkl'))
    imputer = joblib.load(os.path.join('artifacts', 'imputer.pkl'))
    features = joblib.load(os.path.join('artifacts', 'features.pkl'))
    return model, imputer, features

try:
    model, imputer, features = load_assets()
except FileNotFoundError:
    st.error("🚨 Artifacts not found! Please run train_model.py first to generate the models.")
    st.stop()

# --- UI HEADER ---
st.title("🏡 Advanced House Price Prediction System")
st.markdown("Estimate the market price of a property using Machine Learning.")
st.divider()

# --- SIDEBAR: CURRENCY SETTINGS ---
st.sidebar.header("💱 Currency Settings")
currency_choice = st.sidebar.radio("Select Display Currency", ["USD ($)", "INR (₹)"])

# Dynamic Exchange Rate Field
if currency_choice == "INR (₹)":
    exchange_rate = st.sidebar.number_input("Current USD to INR Rate", min_value=50.0, max_value=120.0, value=83.5, step=0.1)
    currency_symbol = "₹"
else:
    exchange_rate = 1.0
    currency_symbol = "$"

st.sidebar.divider()

# --- SIDEBAR: PROPERTY INPUTS ---
st.sidebar.header("📝 Property Metrics")
income = st.sidebar.number_input("Avg. Area Income ($)", min_value=10000, max_value=250000, value=68000, step=1000)
age = st.sidebar.slider("Avg. Area House Age", min_value=1.0, max_value=15.0, value=6.0, step=0.1)
rooms = st.sidebar.slider("Avg. Area Number of Rooms", min_value=1.0, max_value=15.0, value=7.0, step=0.1)
bedrooms = st.sidebar.slider("Avg. Area Number of Bedrooms", min_value=1.0, max_value=10.0, value=4.0, step=0.1)
population = st.sidebar.number_input("Area Population", min_value=1000, max_value=100000, value=36000, step=1000)

# --- MAIN DASHBOARD ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Action Panel")
    predict_btn = st.button("🔮 Predict Price", type="primary", use_container_width=True)

with col2:
    st.subheader("Prediction Results")
    if predict_btn:
        # Prepare input
        input_df = pd.DataFrame([[income, age, rooms, bedrooms, population]], columns=features)
        input_cleaned = pd.DataFrame(imputer.transform(input_df), columns=features)
        
        # Base prediction is in USD
        base_prediction_usd = model.predict(input_cleaned)[0]
        
        # Apply currency math
        final_prediction = base_prediction_usd * exchange_rate
        
        # Display Metric
        st.metric(
            label=f"Estimated Property Value ({currency_choice})", 
            value=f"{currency_symbol} {final_prediction:,.2f}", 
            delta="Market Estimate"
        )
        
        # Save to comparison board
        st.session_state.comparison_list.append({
            "Income ($)": f"${income:,}",
            "Age": age,
            "Rooms": rooms,
            "Bedrooms": bedrooms,
            "Population": f"{population:,}",
            "Price": f"{currency_symbol} {final_prediction:,.2f}"
        })
        st.success("✅ Prediction successful! Added to your comparison board.")
    else:
        st.info("Adjust the metrics in the sidebar and click 'Predict Price'.")

# --- COMPARISON BOARD ---
st.divider()
st.subheader("📊 Property Comparison Board")
if st.session_state.comparison_list:
    comp_df = pd.DataFrame(st.session_state.comparison_list)
    st.dataframe(comp_df, use_container_width=True)
    
    if st.button("Clear History"):
        st.session_state.comparison_list = []
        st.rerun()
else:
    st.markdown("No properties compared yet.")