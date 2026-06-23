import os
import streamlit as st
import requests
import datetime
import time
import pandas as pd
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(
    page_title="Energy Forecast Hub", 
    page_icon="⚡",
    layout="wide"
)

# Custom CSS for Premium Styling (Glassmorphism & Contrast)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .header-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 25px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    
    .status-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .status-label {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    
    .status-value {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    
    .status-ping {
        font-size: 0.8rem;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)

# 2. Base Endpoint URLs with Environment Overrides
PYTHON_API = os.getenv("PYTHON_API_URL", "http://127.0.0.1:8000")
R_API = os.getenv("R_API_URL", "http://127.0.0.1:8001")
JULIA_API = os.getenv("JULIA_API_URL", "http://127.0.0.1:8002")

# Mapping for backend labels
BACKENDS = {
    "Python (FastAPI)": {"url": PYTHON_API, "color": "#0ea5e9", "name": "Python"},
    "R (Plumber)": {"url": R_API, "color": "#a855f7", "name": "R"},
    "Julia (Oxygen)": {"url": JULIA_API, "color": "#f97316", "name": "Julia"}
}

# 3. Request Caching decorator with short TTL (60s)
@st.cache_data(ttl=60, show_spinner=False)
def fetch_prediction_cached(url: str, date_str: str):
    headers = {"Content-Type": "application/json"}
    payload = {"date": date_str}
    start = time.time()
    try:
        response = requests.post(f"{url}/predict", json=payload, timeout=5)
        latency = (time.time() - start) * 1000
        if response.status_code == 200:
            data = response.json()
            # Handle potential vector wrapper unpacking
            date_val = data.get("date")
            if isinstance(date_val, list):
                date_val = date_val[0]
            pred_val = data.get("predicted_consumption_kw")
            if isinstance(pred_val, list):
                pred_val = pred_val[0]
            model_val = data.get("model_used")
            if isinstance(model_val, list):
                model_val = model_val[0]
                
            return {
                "success": True,
                "date": date_val,
                "predicted_consumption_kw": float(pred_val),
                "model_used": model_val,
                "latency_ms": latency
            }
        else:
            return {"success": False, "error": f"HTTP {response.status_code}", "latency_ms": latency}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Timeout", "latency_ms": (time.time() - start) * 1000}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Connection Offline", "latency_ms": (time.time() - start) * 1000}
    except Exception as e:
        return {"success": False, "error": str(e), "latency_ms": (time.time() - start) * 1000}

# Helper to query ping health status
def query_health(url: str):
    start = time.time()
    try:
        response = requests.get(url, timeout=2)
        latency = (time.time() - start) * 1000
        if response.status_code == 200:
            data = response.json()
            status_val = data.get("status")
            if isinstance(status_val, list):
                status_val = status_val[0]
            if status_val == "online":
                return True, round(latency, 1)
        return False, 0.0
    except Exception:
        return False, 0.0

# 4. Header Section
st.markdown("""
<div class="header-box">
    <h1 style="color: #f8fafc; margin: 0; font-weight: 800; font-size: 2.5rem; letter-spacing: -0.02em;">⚡ Energy Forecast Hub</h1>
    <p style="color: #94a3b8; font-size: 1.15rem; margin-top: 6px; font-weight: 400;">
        Interactive Multi-Language Forecaster comparing Python, R, and Julia model execution latencies and prediction consistency.
    </p>
</div>
""", unsafe_allow_html=True)

# 5. Sidebar Configuration Inputs
with st.sidebar:
    st.markdown("### ⚙️ Forecast Settings")
    st.divider()
    
    # Single date vs Date Range selector
    mode = st.radio("Forecast Mode", ["Single Date", "Date Range"], index=1)
    
    today = datetime.date.today()
    if mode == "Single Date":
        selected_date = st.date_input(
            "Select Date", 
            today + datetime.timedelta(days=1),
            min_value=datetime.date(2006, 1, 1),
            max_value=datetime.date(2030, 12, 31)
        )
        start_date = selected_date
        end_date = selected_date
    else:
        date_range = st.date_input(
            "Select Date Range",
            [today + datetime.timedelta(days=1), today + datetime.timedelta(days=7)],
            min_value=datetime.date(2006, 1, 1),
            max_value=datetime.date(2030, 12, 31)
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_date, end_date = date_range
        elif isinstance(date_range, (list, tuple)) and len(date_range) == 1:
            start_date = date_range[0]
            end_date = start_date
        else:
            start_date = today + datetime.timedelta(days=1)
            end_date = start_date
            
    st.divider()
    
    # Backend selection
    backend_choice = st.selectbox(
        "Service Backend",
        ["Compare All", "Python (FastAPI)", "R (Plumber)", "Julia (Oxygen)"]
    )
    
    st.divider()
    
    # URL Overrides
    with st.expander("🔌 Advanced API Endpoints"):
        override_py = st.text_input("Python URL", PYTHON_API)
        override_r = st.text_input("R URL", R_API)
        override_ju = st.text_input("Julia URL", JULIA_API)
        
        # Apply overrides
        BACKENDS["Python (FastAPI)"]["url"] = override_py
        BACKENDS["R (Plumber)"]["url"] = override_r
        BACKENDS["Julia (Oxygen)"]["url"] = override_ju

# 6. Service Status Profiler Grid
st.markdown("#### 🔌 Microservice Profiler")
p_col1, p_col2, p_col3 = st.columns(3)

status_results = {}
for label, col in zip(["Python (FastAPI)", "R (Plumber)", "Julia (Oxygen)"], [p_col1, p_col2, p_col3]):
    url = BACKENDS[label]["url"]
    online, ping = query_health(url)
    status_results[label] = {"online": online, "ping": ping}
    
    status_indicator = "🟢 Active" if online else "🔴 Offline"
    ping_str = f"{ping:.1f} ms" if online else "N/A"
    color_val = "#10b981" if online else "#ef4444"
    
    col.markdown(f"""
    <div class="status-card">
        <div class="status-label">{label}</div>
        <div class="status-value" style="color: {color_val};">{status_indicator}</div>
        <div class="status-ping">Ping Latency: {ping_str}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Identify active backends to run predictions on
active_backends = []
if backend_choice == "Compare All":
    active_backends = list(BACKENDS.keys())
else:
    active_backends = [backend_choice]

# 7. Generate Forecast Run
generate = st.sidebar.button("Generate Forecast", type="primary", use_container_width=True)

if generate or 'predictions_df' in st.session_state:
    # Build list of date strings
    delta = (end_date - start_date).days
    date_list = [start_date + datetime.timedelta(days=i) for i in range(delta + 1)]
    date_strings = [d.strftime("%Y-%m-%d") for d in date_list]
    
    # Store predictions in a list of dicts
    pred_records = []
    errors = []
    
    # If this is a manual click, run query
    if generate:
        with st.spinner("Contacting services and generating predictions..."):
            for d_str in date_strings:
                for b_lbl in active_backends:
                    b_info = BACKENDS[b_lbl]
                    res = fetch_prediction_cached(b_info["url"], d_str)
                    
                    if res["success"]:
                        pred_records.append({
                            "Date": d_str,
                            "Backend": b_info["name"],
                            "Consumption (kW)": res["predicted_consumption_kw"],
                            "Latency (ms)": res["latency_ms"],
                            "Model": res["model_used"]
                        })
                    else:
                        errors.append((b_lbl, d_str, res["error"]))
        
        # Save to session state
        st.session_state['predictions_df'] = pd.DataFrame(pred_records)
        st.session_state['errors'] = errors
        
    df_preds = st.session_state.get('predictions_df', pd.DataFrame())
    errors = st.session_state.get('errors', [])
    
    # Graceful degradation alerts
    if errors:
        for b_lbl, d_str, err in errors:
            st.toast(f"⚠️ {b_lbl} offline or failed for {d_str}: {err}")
        
        # Group errors by service and show user help commands
        failed_services = list(set([e[0] for e in errors]))
        for service in failed_services:
            st.warning(f"🚨 **{service}** failed to respond. The dashboard is degrading gracefully, using remaining online backends.")
            with st.expander(f"🔧 How to start {service} locally?"):
                if "Python" in service:
                    st.code("poetry run python app.py  # or run via uvicorn directly", language="bash")
                elif "R" in service:
                    st.code("Rscript -e \"pr <- plumber::pr('R_project/api.R'); plumber::pr_run(pr, port=8001)\"", language="bash")
                elif "Julia" in service:
                    st.code("julia --project=\".\" src/api.jl", language="bash")

    if not df_preds.empty:
        # Display Summary & Graphs
        st.markdown("### 📊 Forecast Results & Comparisons")
        
        # Visualization: Plotly Line Chart
        fig = go.Figure()
        
        # Plot one trace per backend
        for b_lbl in active_backends:
            b_name = BACKENDS[b_lbl]["name"]
            b_color = BACKENDS[b_lbl]["color"]
            
            sub_df = df_preds[df_preds["Backend"] == b_name].sort_values("Date")
            if not sub_df.empty:
                fig.add_trace(go.Scatter(
                    x=sub_df["Date"],
                    y=sub_df["Consumption (kW)"],
                    mode="lines+markers",
                    name=f"{b_name} Forecast",
                    line=dict(color=b_color, width=3),
                    marker=dict(size=7),
                    hovertemplate="<b>%{x}</b><br>Consumption: %{y:.2f} kW<br>Backend: " + b_name + "<extra></extra>"
                ))
        
        fig.update_layout(
            title="Household Daily Energy Forecast Comparison",
            xaxis_title="Date",
            yaxis_title="Predicted Consumption (kW)",
            hovermode="x unified",
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=80, b=40),
            height=450
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics Cards Grid
        st.markdown("#### 📈 Analytics Summary")
        m_cols = st.columns(len(active_backends))
        
        for idx, b_lbl in enumerate(active_backends):
            b_name = BACKENDS[b_lbl]["name"]
            b_color = BACKENDS[b_lbl]["color"]
            sub_df = df_preds[df_preds["Backend"] == b_name]
            
            if not sub_df.empty:
                avg_val = sub_df["Consumption (kW)"].mean()
                peak_val = sub_df["Consumption (kW)"].max()
                avg_latency = sub_df["Latency (ms)"].mean()
                
                with m_cols[idx]:
                    st.markdown(f"""
                    <div style="border-left: 5px solid {b_color}; padding-left: 15px; background: rgba(30, 41, 59, 0.25); padding-top: 10px; padding-bottom: 10px; border-radius: 4px;">
                        <h4 style="margin: 0; color: {b_color};">{b_name} Performance</h4>
                        <p style="margin: 5px 0 2px 0; font-size: 0.9rem; color: #94a3b8;">Average: <b>{avg_val:.2f} kW</b></p>
                        <p style="margin: 2px 0 2px 0; font-size: 0.9rem; color: #94a3b8;">Peak: <b>{peak_val:.2f} kW</b></p>
                        <p style="margin: 2px 0 0 0; font-size: 0.9rem; color: #94a3b8;">Avg Latency: <b>{avg_latency:.1f} ms</b></p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Detailed Forecast Data Table
        st.divider()
        with st.expander("📊 View Detailed Forecast Comparison Data"):
            pivot_df = df_preds.pivot(index="Date", columns="Backend", values="Consumption (kW)").reset_index()
            st.dataframe(pivot_df.style.format(precision=2), use_container_width=True, hide_index=True)
            
            st.markdown("**API Response Benchmarks (milliseconds)**")
            pivot_latency = df_preds.pivot(index="Date", columns="Backend", values="Latency (ms)").reset_index()
            st.dataframe(pivot_latency.style.format(precision=1), use_container_width=True, hide_index=True)
    else:
        st.info("No prediction data fetched. Check if any microservices are online and click 'Generate Forecast' in the sidebar.")
else:
    st.info("👈 Configure the dates and service backends in the sidebar, and click 'Generate Forecast' to start.")