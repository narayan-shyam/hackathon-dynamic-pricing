"""
Phase 7: Streamlit UI for Dynamic Pricing Predictions
GlobalMart Tide Detergent Pricing Strategy Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Dynamic Pricing Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-box {
        background-color: #e8f4fd;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def make_api_request(endpoint, method="GET", data=None):
    """Make API request with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code} - {response.text}"
    except requests.exceptions.ConnectionError:
        return None, "Could not connect to API. Please ensure the FastAPI server is running."
    except Exception as e:
        return None, f"Error: {str(e)}"

def check_api_health():
    """Check if API is healthy"""
    result, error = make_api_request("/health")
    if result:
        return result.get("status") == "healthy"
    return False

# Main Application
def main():
    # Header
    st.markdown('<h1 class="main-header">🏷️ Dynamic Pricing Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**GlobalMart Tide Detergent AI-Powered Pricing Strategy**")
    
    # API Health Check
    if check_api_health():
        st.success("✅ API Connection Healthy")
    else:
        st.error("❌ API Connection Failed. Please start the FastAPI server.")
        st.stop()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", [
        "Single Prediction", 
        "Batch Predictions", 
        "Price Optimization", 
        "Model Analytics",
        "Historical Data"
    ])
    
    if page == "Single Prediction":
        single_prediction_page()
    elif page == "Batch Predictions":
        batch_prediction_page()
    elif page == "Price Optimization":
        price_optimization_page()
    elif page == "Model Analytics":
        model_analytics_page()
    elif page == "Historical Data":
        historical_data_page()

def single_prediction_page():
    """Single prediction interface"""
    st.header("📊 Single Prediction")
    st.write("Enter product and market parameters to predict units sold.")
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Pricing Information")
        mrp = st.number_input("Maximum Retail Price (₹)", min_value=50.0, max_value=200.0, value=100.0, step=1.0)
        no_promo_price = st.number_input("No Promotion Price (₹)", min_value=40.0, max_value=180.0, value=90.0, step=1.0)
        selling_price = st.number_input("Current Selling Price (₹)", min_value=30.0, max_value=160.0, value=80.0, step=1.0)
        competitor_price = st.number_input("Competitor Price (₹)", min_value=30.0, max_value=160.0, value=85.0, step=1.0)
    
    with col2:
        st.subheader("Market & Customer Data")
        ctr = st.slider("Click-Through Rate", min_value=0.001, max_value=0.1, value=0.02, step=0.001, format="%.3f")
        abandoned_cart_rate = st.slider("Cart Abandonment Rate", min_value=0.05, max_value=0.5, value=0.2, step=0.01)
        bounce_rate = st.slider("Bounce Rate", min_value=0.1, max_value=0.7, value=0.3, step=0.01)
        is_metro = st.checkbox("Metro City Location", value=True)
    
    # Date features
    st.subheader("Date Features")
    col3, col4, col5, col6 = st.columns(4)
    
    with col3:
        month = st.selectbox("Month", list(range(1, 13)), index=5)
    with col4:
        day = st.selectbox("Day", list(range(1, 32)), index=14)
    with col5:
        dayofweek = st.selectbox("Day of Week", list(range(1, 8)), index=2)
    with col6:
        quarter = st.selectbox("Quarter", list(range(1, 5)), index=1)
    
    # Prediction button
    if st.button("🔮 Predict Units Sold", type="primary"):
        # Prepare data
        prediction_data = {
            "MRP": mrp,
            "NoPromoPrice": no_promo_price,
            "SellingPrice": selling_price,
            "CTR": ctr,
            "AbandonedCartRate": abandoned_cart_rate,
            "BounceRate": bounce_rate,
            "IsMetro": is_metro,
            "month": month,
            "day": day,
            "dayofweek": dayofweek,
            "quarter": quarter,
            "competitor_price": competitor_price
        }
        
        # Make prediction
        with st.spinner("Making prediction..."):
            result, error = make_api_request("/predict", method="POST", data=prediction_data)
        
        if result:
            # Display results
            st.success("Prediction completed!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Predicted Units Sold",
                    value=f"{result['predicted_units_sold']:.0f}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    label="Confidence Score",
                    value=f"{result['confidence_score']:.1%}",
                    delta=None
                )
            
            with col3:
                discount_rate = (mrp - selling_price) / mrp * 100
                st.metric(
                    label="Discount Rate",
                    value=f"{discount_rate:.1f}%",
                    delta=None
                )
            
            # Recommendation
            st.markdown(f"""
            <div class="recommendation-box">
                <h4>💡 Pricing Recommendation</h4>
                <p>{result['pricing_recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional insights
            revenue = selling_price * result['predicted_units_sold']
            st.info(f"**Projected Revenue:** ₹{revenue:,.2f}")
            
        else:
            st.error(f"Prediction failed: {error}")

def batch_prediction_page():
    """Batch prediction interface"""
    st.header("📋 Batch Predictions")
    st.write("Upload multiple scenarios for batch prediction.")
    
    # Option to use sample data or upload
    option = st.radio("Choose input method:", ["Use Sample Data", "Upload CSV"])
    
    if option == "Use Sample Data":
        # Generate sample data
        if st.button("Generate Sample Data"):
            np.random.seed(42)
            sample_data = []
            
            for i in range(10):
                sample_data.append({
                    "MRP": np.random.uniform(90, 120),
                    "NoPromoPrice": np.random.uniform(80, 110),
                    "SellingPrice": np.random.uniform(70, 100),
                    "CTR": np.random.uniform(0.01, 0.05),
                    "AbandonedCartRate": np.random.uniform(0.1, 0.3),
                    "BounceRate": np.random.uniform(0.2, 0.5),
                    "IsMetro": np.random.choice([True, False]),
                    "month": np.random.randint(1, 13),
                    "day": np.random.randint(1, 29),
                    "dayofweek": np.random.randint(1, 8),
                    "quarter": np.random.randint(1, 5),
                    "competitor_price": np.random.uniform(75, 105)
                })
            
            st.session_state.batch_data = sample_data
            st.success("Sample data generated!")
    
    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.batch_data = df.to_dict('records')
                st.success(f"Uploaded {len(df)} records")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    # Batch prediction
    if 'batch_data' in st.session_state and st.button("🚀 Run Batch Prediction"):
        batch_request = {"predictions": st.session_state.batch_data}
        
        with st.spinner("Processing batch predictions..."):
            result, error = make_api_request("/predict/batch", method="POST", data=batch_request)
        
        if result:
            st.success("Batch prediction completed!")
            
            # Summary metrics
            summary = result['summary']
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Predictions", summary['total_predictions'])
            with col2:
                st.metric("Avg Units Predicted", f"{summary['average_units_predicted']:.1f}")
            with col3:
                st.metric("Total Units", f"{summary['total_units_predicted']:.0f}")
            with col4:
                st.metric("Avg Confidence", f"{summary['avg_confidence']:.1%}")
            
            # Results table
            results_df = pd.DataFrame([
                {
                    "Predicted Units": r['predicted_units_sold'],
                    "Confidence": f"{r['confidence_score']:.1%}",
                    "Recommendation": r['pricing_recommendation']
                }
                for r in result['results']
            ])
            
            st.subheader("Detailed Results")
            st.dataframe(results_df)
            
            # Visualization
            fig = px.histogram(
                x=[r['predicted_units_sold'] for r in result['results']],
                title="Distribution of Predicted Units Sold",
                labels={'x': 'Predicted Units', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error(f"Batch prediction failed: {error}")

def price_optimization_page():
    """Price optimization interface"""
    st.header("🎯 Price Optimization")
    st.write("Find the optimal price to maximize units sold.")
    
    # Input parameters (simplified for optimization)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Base Parameters")
        mrp = st.number_input("MRP (₹)", min_value=50.0, max_value=200.0, value=100.0)
        no_promo_price = st.number_input("No Promo Price (₹)", min_value=40.0, max_value=180.0, value=90.0)
        competitor_price = st.number_input("Competitor Price (₹)", min_value=30.0, max_value=160.0, value=85.0)
        is_metro = st.checkbox("Metro Location", value=True)
    
    with col2:
        st.subheader("Optimization Range")
        min_price = st.number_input("Minimum Price (₹)", min_value=30.0, max_value=100.0, value=60.0)
        max_price = st.number_input("Maximum Price (₹)", min_value=70.0, max_value=150.0, value=120.0)
        
        # Fixed values for simplification
        ctr = 0.025
        abandoned_cart_rate = 0.2
        bounce_rate = 0.3
        month = 6
        day = 15
        dayofweek = 3
        quarter = 2
    
    if st.button("🔍 Find Optimal Price", type="primary"):
        # Prepare optimization request
        base_features = {
            "MRP": mrp,
            "NoPromoPrice": no_promo_price,
            "SellingPrice": (min_price + max_price) / 2,  # Will be optimized
            "CTR": ctr,
            "AbandonedCartRate": abandoned_cart_rate,
            "BounceRate": bounce_rate,
            "IsMetro": is_metro,
            "month": month,
            "day": day,
            "dayofweek": dayofweek,
            "quarter": quarter,
            "competitor_price": competitor_price
        }
        
        price_range = [min_price, max_price]
        
        with st.spinner("Optimizing price..."):
            # Manual optimization using multiple predictions
            prices = np.arange(min_price, max_price + 1, 1)
            predictions = []
            
            for price in prices:
                test_features = base_features.copy()
                test_features["SellingPrice"] = price
                
                result, error = make_api_request("/predict", method="POST", data=test_features)
                if result:
                    predictions.append({
                        "price": price,
                        "units": result["predicted_units_sold"],
                        "revenue": price * result["predicted_units_sold"]
                    })
        
        if predictions:
            # Find optimal price
            optimal_units = max(predictions, key=lambda x: x["units"])
            optimal_revenue = max(predictions, key=lambda x: x["revenue"])
            
            st.success("Optimization completed!")
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Optimal Price (Units)",
                    f"₹{optimal_units['price']:.0f}",
                    f"{optimal_units['units']:.0f} units"
                )
            
            with col2:
                st.metric(
                    "Optimal Price (Revenue)",
                    f"₹{optimal_revenue['price']:.0f}",
                    f"₹{optimal_revenue['revenue']:.0f}"
                )
            
            with col3:
                current_pred = next((p for p in predictions if abs(p['price'] - base_features['SellingPrice']) < 0.5), predictions[0])
                improvement = optimal_units['units'] - current_pred['units']
                st.metric(
                    "Improvement",
                    f"{improvement:.0f} units",
                    f"{improvement/current_pred['units']*100:.1f}%"
                )
            
            # Visualization
            df_viz = pd.DataFrame(predictions)
            
            # Price vs Units chart
            fig1 = px.line(df_viz, x='price', y='units', 
                          title='Price vs Predicted Units Sold',
                          labels={'price': 'Price (₹)', 'units': 'Predicted Units'})
            fig1.add_vline(x=optimal_units['price'], line_dash="dash", 
                          annotation_text=f"Optimal: ₹{optimal_units['price']}")
            st.plotly_chart(fig1, use_container_width=True)
            
            # Price vs Revenue chart
            fig2 = px.line(df_viz, x='price', y='revenue', 
                          title='Price vs Projected Revenue',
                          labels={'price': 'Price (₹)', 'revenue': 'Revenue (₹)'})
            fig2.add_vline(x=optimal_revenue['price'], line_dash="dash", 
                          annotation_text=f"Optimal: ₹{optimal_revenue['price']}")
            st.plotly_chart(fig2, use_container_width=True)
            
        else:
            st.error("Optimization failed")

def model_analytics_page():
    """Model analytics and insights"""
    st.header("📈 Model Analytics")
    st.write("Explore model performance and feature importance.")
    
    # Model info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Information")
        info_result, info_error = make_api_request("/model/info")
        
        if info_result:
            st.json(info_result)
        else:
            st.error(f"Could not fetch model info: {info_error}")
    
    with col2:
        st.subheader("Model Metrics")
        metrics_result, metrics_error = make_api_request("/metrics/model")
        
        if metrics_result:
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric("R² Score", f"{metrics_result['r2_score']:.3f}")
                st.metric("RMSE", f"{metrics_result['rmse']:.2f}")
            with col2_2:
                st.metric("MAE", f"{metrics_result['mae']:.2f}")
                st.metric("Training Samples", f"{metrics_result['training_samples']:,}")
        else:
            st.error(f"Could not fetch metrics: {metrics_error}")
    
    # Feature importance
    st.subheader("Feature Importance")
    importance_result, importance_error = make_api_request("/features/importance")
    
    if importance_result:
        importance_df = pd.DataFrame(importance_result['feature_importance'])
        
        # Bar chart
        fig = px.bar(importance_df.head(10), 
                    x='importance', y='feature',
                    orientation='h',
                    title='Top 10 Most Important Features')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Full table
        with st.expander("View All Features"):
            st.dataframe(importance_df)
    else:
        st.error(f"Could not fetch feature importance: {importance_error}")

def historical_data_page():
    """Historical data analysis"""
    st.header("📊 Historical Data Analysis")
    st.write("Analyze historical pricing and sales patterns.")
    
    # Generate sample historical data for demonstration
    if st.button("Generate Sample Historical Data"):
        np.random.seed(42)
        
        # Generate 90 days of data
        dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
        
        historical_data = []
        base_price = 80
        
        for i, date in enumerate(dates):
            # Add seasonal and trend effects
            seasonal_effect = 10 * np.sin(2 * np.pi * i / 30)  # Monthly cycle
            trend_effect = i * 0.1  # Slight upward trend
            noise = np.random.normal(0, 5)
            
            price = base_price + seasonal_effect + trend_effect + noise
            price = max(60, min(120, price))  # Constrain price range
            
            units = 100 - (price - 70) * 0.8 + np.random.normal(0, 10)
            units = max(20, units)
            
            historical_data.append({
                'date': date,
                'price': round(price, 2),
                'units_sold': round(units, 0),
                'revenue': round(price * units, 2),
                'competitor_price': round(price + np.random.normal(0, 5), 2)
            })
        
        df_hist = pd.DataFrame(historical_data)
        st.session_state.historical_df = df_hist
        
        # Display summary
        st.success(f"Generated {len(df_hist)} days of historical data")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Price", f"₹{df_hist['price'].mean():.2f}")
        with col2:
            st.metric("Avg Units", f"{df_hist['units_sold'].mean():.0f}")
        with col3:
            st.metric("Total Revenue", f"₹{df_hist['revenue'].sum():,.0f}")
        with col4:
            st.metric("Days Analyzed", len(df_hist))
    
    # Analysis if data exists
    if 'historical_df' in st.session_state:
        df_hist = st.session_state.historical_df
        
        # Time series charts
        st.subheader("Time Series Analysis")
        
        # Price and units over time
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['price'], 
                                name='Price', yaxis='y', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['units_sold'], 
                                name='Units Sold', yaxis='y2', line=dict(color='red')))
        
        fig.update_layout(
            title='Price vs Units Sold Over Time',
            xaxis_title='Date',
            yaxis=dict(title='Price (₹)', side='left', color='blue'),
            yaxis2=dict(title='Units Sold', side='right', overlaying='y', color='red')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Revenue trend
        fig_revenue = px.line(df_hist, x='date', y='revenue', 
                             title='Revenue Trend Over Time')
        st.plotly_chart(fig_revenue, use_container_width=True)
        
        # Price elasticity analysis
        st.subheader("Price Elasticity Analysis")
        correlation = df_hist['price'].corr(df_hist['units_sold'])
        st.metric("Price-Units Correlation", f"{correlation:.3f}")
        
        # Scatter plot
        fig_scatter = px.scatter(df_hist, x='price', y='units_sold', 
                               title='Price vs Units Sold Relationship',
                               trendline='ols')
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Data table
        with st.expander("View Raw Data"):
            st.dataframe(df_hist)

# Run the app
if __name__ == "__main__":
    main()
