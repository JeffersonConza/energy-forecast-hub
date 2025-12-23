import os
import streamlit as st
import requests
import datetime
import pandas as pd
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(
    page_title="Energy Forecast Hub", 
    page_icon="⚡",
    layout="centered"
)

# 2. Header and Description
st.title("⚡ Energy Forecast Hub")
st.markdown("### Power Consumption Predictor")
st.markdown("Select a future date to predict the household energy usage using our **Random Forest AI Model**.")

# 3. User Input (Date Picker)
col1, col2 = st.columns(2)
with col1:
    selected_date = st.date_input(
        "Select Date",
        datetime.date.today() + datetime.timedelta(days=1),
        min_value=datetime.date(2020, 1, 1),
        max_value=datetime.date(2030, 12, 31)
    )

with col2:
    st.markdown("**Day of Week**")
    day_name = selected_date.strftime("%A")
    st.info(f"{day_name}")

# Option to predict multiple days
predict_range = st.checkbox("Predict multiple days", value=False)
num_days = 1
if predict_range:
    num_days = st.slider("Number of days to forecast", 1, 30, 7)

# 4. The Logic
default_url = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
API_URL = st.text_input(
    "API Endpoint", 
    default_url,
    help="Update this if your API is hosted elsewhere"
)

if st.button("Generate Forecast", type="primary", use_container_width=True):
    
    predictions = []
    
    with st.spinner(f"Consulting the AI Model for {num_days} day(s)..."):
        try:
            # Predict for single or multiple days
            for i in range(num_days):
                current_date = selected_date + datetime.timedelta(days=i)
                payload = {"date": current_date.strftime("%Y-%m-%d")}
                
                response = requests.post(API_URL, json=payload, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    predictions.append({
                        "date": current_date.strftime("%Y-%m-%d"),  # Store as string
                        "day": current_date.strftime("%A"),
                        "consumption": data["predicted_consumption_kw"],
                        "model": data.get("model_used", "Unknown")
                    })
                else:
                    st.error(f"Error for {current_date}: {response.status_code}")
                    break
            
            if predictions:
                # 5. Display Results
                st.divider()
                st.success(f"✅ Prediction Complete! ({len(predictions)} day(s))")
                
                if num_days == 1:
                    # Single day metrics
                    m1, m2 = st.columns(2)
                    m1.metric("Predicted Consumption", f"{predictions[0]['consumption']:.2f} kW")
                    m2.metric("Model Version", predictions[0]['model'])
                else:
                    # Multiple days visualization
                    df = pd.DataFrame(predictions)
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Average", f"{df['consumption'].mean():.2f} kW")
                    col2.metric("Peak", f"{df['consumption'].max():.2f} kW")
                    col3.metric("Lowest", f"{df['consumption'].min():.2f} kW")
                    
                    # Line chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df['date'],
                        y=df['consumption'],
                        mode='lines+markers',
                        name='Predicted Consumption',
                        line=dict(color='#1f77b4', width=3),
                        marker=dict(size=8)
                    ))
                    
                    fig.update_layout(
                        title="Energy Consumption Forecast",
                        xaxis_title="Date",
                        yaxis_title="Consumption (kW)",
                        hovermode='x unified',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Data table
                    with st.expander("📊 Detailed Forecast Table"):
                        display_df = df.copy()
                        display_df['consumption'] = display_df['consumption'].round(2)
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Additional context
                with st.expander("📋 Prediction Details"):
                    if num_days == 1:
                        st.write(f"**Date:** {predictions[0]['date']}")
                        st.write(f"**Day:** {predictions[0]['day']}")
                        st.write(f"**Model:** {predictions[0]['model']}")
                    else:
                        st.write(f"**Date Range:** {predictions[0]['date']} to {predictions[-1]['date']}")
                        st.write(f"**Total Days:** {len(predictions)}")
                        st.write(f"**Model:** {predictions[0]['model']}")
                
        except requests.exceptions.ConnectionError:
            st.error("🚨 Connection Refused!")
            st.warning("⚠️ Is your FastAPI backend running?")
            st.code("uvicorn app:app --reload", language="bash")
            
        except requests.exceptions.Timeout:
            st.error("⏱️ Request Timed Out")
            st.warning("The API is taking too long to respond. Please try again.")
            
        except requests.exceptions.RequestException as e:
            st.error("❌ Request Failed")
            st.exception(e)

# 6. Sidebar
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    This app predicts household energy consumption using machine learning.
    
    **Features:**
    - Random Forest regression model
    - Single & multi-day forecasts
    - Visual trend analysis
    - Historical data training
    
    **How to use:**
    1. Select a date (or date range)
    2. Click 'Generate Forecast'
    3. View predictions & insights
    """)
    
    st.divider()
    st.markdown("**API Status**")
    
    if st.button("Check API Connection", use_container_width=True):
        try:
            health_response = requests.get(API_URL.replace("/predict", "/"), timeout=3)
            if health_response.status_code == 200:
                st.success("✅ API Online")
            else:
                st.warning(f"⚠️ API returned {health_response.status_code}")
        except:
            st.error("❌ API Offline")