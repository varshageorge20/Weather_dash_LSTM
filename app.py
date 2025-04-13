import streamlit as st
import pandas as pd
from weather_fetcher import get_weather_data
from model import forecast_temperature_lstm

st.set_page_config(page_title="Weather Forecast Dashboard", layout="centered")
st.title("🌦️ Weather Forecast Dashboard")
st.write("Get 5-day forecast and short-term LSTM-based predictions")

# Input section
API_KEY = st.secrets["api_key"]
city = st.text_input("Enter a city:", value="Austin")

# Main logic only runs if both are provided
if API_KEY and city:
    try:
        df = get_weather_data(city, API_KEY)

        if not df.empty:
            st.subheader("Current Weather Data")
            st.line_chart(df.set_index("timestamp")[["temp", "humidity"]])

            future_hours = st.slider("Hours to forecast", 1, 12, 5)
            st.subheader("Forecasted Temperatures")

            with st.spinner("Training LSTM and forecasting..."):
                forecast_df = forecast_temperature_lstm(df, future_hours)
                st.line_chart(forecast_df.set_index("timestamp")["predicted_temp"])
                st.dataframe(forecast_df)
        else:
            st.warning("No data found. Try a different city.")
    except Exception as e:
        st.error(f"Error fetching or processing data: {e}")
else:
    st.info("Please enter your API key and city name to begin.")
