import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Data Story Builder", layout="wide")
st.title("Data Story Builder")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    # Detect column types
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    date_candidates = []

    for col in df.columns:
        try:
            pd.to_datetime(df[col])
            date_candidates.append(col)
        except:
            continue

    # Sidebar input
    st.sidebar.header("Configuration")
    date_field = st.sidebar.selectbox("Select a date field", date_candidates)
    value_field = st.sidebar.selectbox("Select a numeric field", numeric_cols)

    # Safe conversion for datetime
    try:
        df[date_field] = pd.to_datetime(df[date_field], errors="coerce")
        if df[date_field].isnull().all():
            st.error(f"The selected date column '{date_field}' could not be parsed as datetime.")
            st.stop()
    except Exception as e:
        st.error(f"Error converting '{date_field}' to datetime: {e}")
        st.stop()

    df[value_field] = pd.to_numeric(df[value_field], errors='coerce')
    df = df.dropna(subset=[date_field, value_field])

    # KPI Metrics
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total", f"{df[value_field].sum():,.2f}")
    col2.metric("Average", f"{df[value_field].mean():,.2f}")
    col3.metric("Maximum", f"{df[value_field].max():,.2f}")

    # Trend Over Time
    st.subheader("Trend Over Time")
    df_grouped = df.groupby(df[date_field].dt.to_period("M"))[value_field].sum().reset_index()
    df_grouped[date_field] = df_grouped[date_field].dt.to_timestamp()

    fig = px.line(
        df_grouped,
        x=date_field,
        y=value_field,
        title="Monthly Trend",
        labels={date_field: "Date", value_field: "Value"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Forecast
    st.subheader("Forecast")
    X = (df_grouped[date_field] - df_grouped[date_field].min()).dt.days.values.reshape(-1, 1)
    y = df_grouped[value_field].values

    model = LinearRegression().fit(X, y)
    df_grouped["Forecast"] = model.predict(X)

    fig2 = px.line(
        df_grouped,
        x=date_field,
        y="Forecast",
        title="Forecasted Trend",
        labels={date_field: "Date", "Forecast": "Predicted Value"}
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Optional raw data preview
    with st.expander("View Raw Data"):
        st.dataframe(df.head(50))

