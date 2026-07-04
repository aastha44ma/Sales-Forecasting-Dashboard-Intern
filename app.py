import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="End-to-End Sales Forecasting", layout="wide")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    raw_df = pd.read_csv('streamlit_raw_data.csv')
    raw_df['Order Date'] = pd.to_datetime(raw_df['Order Date'])
    raw_df['Year'] = raw_df['Order Date'].dt.year
    raw_df['Month'] = raw_df['Order Date'].dt.to_period('M').astype(str)
    
    anomalies_df = pd.read_csv('streamlit_anomalies.csv')
    anomalies_df['Order Date'] = pd.to_datetime(anomalies_df['Order Date'])
    
    clusters_df = pd.read_csv('streamlit_clusters.csv')
    return raw_df, anomalies_df, clusters_df

raw_df, anomalies_df, clusters_df = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page:", 
                        ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Segments"])

# ==========================================
# PAGE 1: SALES OVERVIEW DASHBOARD
# ==========================================
if page == "Sales Overview":
    st.title("📊 Sales Overview Dashboard")
    
    # Interactive Filters
    col1, col2 = st.columns(2)
    selected_region = col1.selectbox("Filter by Region", ["All"] + list(raw_df['Region'].unique()))
    selected_category = col2.selectbox("Filter by Category", ["All"] + list(raw_df['Category'].unique()))
    
    filtered_df = raw_df.copy()
    if selected_region != "All":
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df['Category'] == selected_category]
        
    # Charts
    st.subheader("Total Sales by Year")
    yearly_sales = filtered_df.groupby('Year')['Sales'].sum()
    st.bar_chart(yearly_sales)
    
    st.subheader("Monthly Sales Trend")
    monthly_trend = filtered_df.groupby('Month')['Sales'].sum()
    st.line_chart(monthly_trend)


# ==========================================
# PAGE 2: FORECAST EXPLORER
# ==========================================
elif page == "Forecast Explorer":
    st.title("📈 Forecast Explorer (Prophet Model)")
    
    col1, col2 = st.columns(2)
    segment_type = col1.selectbox("Segment Type", ["Category", "Region"])
    
    if segment_type == "Category":
        segment_value = col2.selectbox("Select Category", raw_df['Category'].unique())
        segment_df = raw_df[raw_df['Category'] == segment_value]
    else:
        segment_value = col2.selectbox("Select Region", raw_df['Region'].unique())
        segment_df = raw_df[raw_df['Region'] == segment_value]
        
    horizon = st.slider("Forecast Horizon (Months Ahead)", min_value=1, max_value=3, value=3)
    
    if st.button("Generate Forecast"):
        with st.spinner('Training Prophet Model...'):
            # FIX 1: Changed freq='M' to freq='ME'
            monthly_sales = segment_df.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales'].sum().reset_index()
            prophet_df = monthly_sales.rename(columns={'Order Date': 'ds', 'Sales': 'y'})
            
            model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            model.fit(prophet_df)
            
            # FIX 2: Changed freq='M' to freq='ME'
            future = model.make_future_dataframe(periods=horizon, freq='ME')
            forecast = model.predict(future)
            
            fig = model.plot(forecast)
            st.pyplot(fig)
            
            st.subheader("Forecast Output")
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(horizon))
# ==========================================
# PAGE 3: ANOMALY REPORT
# ==========================================
elif page == "Anomaly Report":
    st.title("🚨 Anomaly Detection Report")
    st.markdown("Displays weeks where sales were unusually high or low based on the Multi-Source model.")
    
    # Load and display the chart saved from Colab
    if os.path.exists('charts/Task5_Anomalies_MultiSource.png'):
        st.image('charts/Task5_Anomalies_MultiSource.png', caption="Weekly Sales Anomalies")
    else:
        st.warning("Chart image not found. Ensure 'charts/' folder is uploaded to GitHub.")
        
    st.subheader("Detected Anomalies Table")
    # Filter for Isolation Forest anomalies (represented by -1)
    anomalies_only = anomalies_df[anomalies_df['Anomaly_IF'] == -1]
    st.dataframe(anomalies_only[['Order Date', 'Sales', 'Macro_Retail_Index']].sort_values(by='Order Date'))

# ==========================================
# PAGE 4: PRODUCT DEMAND SEGMENTS
# ==========================================
elif page == "Product Segments":
    st.title("📦 Product Demand Segments")
    
    # Load and display the chart saved from Colab
    if os.path.exists('charts/Task6_PCA_Clusters.png'):
        st.image('charts/Task6_PCA_Clusters.png', caption="K-Means Product Segmentation")
    else:
        st.warning("Chart image not found. Ensure 'charts/' folder is uploaded to GitHub.")
        
    st.subheader("Sub-Category Segment Mappings")
    st.dataframe(clusters_df[['Sub-Category', 'Total_Sales', 'YoY_Growth', 'Demand_Segment']])
