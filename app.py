import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

st.title("Netflix Subscribers Forecasting App")

with open('arima_model.pkl', 'rb') as pkl:
    model = pickle.load(pkl)

uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    ts = df.set_index('Date')['Subscribers']

    forecast = model.forecast(steps=4)

    st.subheader("Forecasted Subscribers for next 4 quarters")
    st.write(forecast)

    plt.figure(figsize=(10,5))
    plt.plot(ts, label='Actual Subscribers', marker='o')
    plt.plot(forecast.index, forecast.values, marker='x', color='red', label='Forecasted')
    plt.legend()
    st.pyplot(plt)
