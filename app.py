import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Custom interface imports
from interface.Predict_freight import predict_freight_cost
from interface.predict_invoice_flag import predict_flag_invoice

# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="Invoice Intelligence AI", 
    page_icon="💼", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI decoration
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # --- Sidebar Navigation ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2936/2936630.png", width=80)
        st.title("AI Modules")
        st.markdown("Select a predictive module to run analysis.")
        st.divider()
        app_mode = st.radio("Navigation", ["🚢 Freight Cost Predictor", "⚠️ Invoice Flagger"], label_visibility="collapsed")
        
        st.divider()
        st.caption("System Status: Online 🟢")
        st.caption("Model Version: v1.2")

    # --- Routing ---
    if "Freight" in app_mode:
        freight_interface()
    else:
        invoice_interface()

def freight_interface():
    st.markdown('<p class="main-header">Freight Cost Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Estimate shipping and freight costs based on transaction volume.</p>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ How to use this module", expanded=True):
        st.write("Enter one or multiple transaction amounts (in Dollars) separated by commas to generate bulk freight cost predictions.")

    st.markdown("### Input Parameters")
    user_input = st.text_input(
        "Transaction Amounts (USD $)", 
        value="18500, 9000, 3000, 200",
        help="Example: 1500, 2000, 3500"
    )
    
    if st.button("Calculate Freight Costs", type="primary"):
        try:
            # Parse input
            dollars_list = [float(x.strip()) for x in user_input.split(",")]
            input_data = {'Dollars': dollars_list}
            
            with st.spinner("Processing through ML Model..."):
                result_df = predict_freight_cost(input_data)
            
            st.divider()
            st.markdown("### Prediction Results")
            
            col1, col2 = st.columns([1, 1.5])
            
            with col1:
                st.dataframe(result_df, use_container_width=True)
                
            with col2:
                # Handle transpose (.T) logic from the backend
                if 'Dollars' in result_df.index:
                    y_vals = result_df.loc['Predict_freight'].values if 'Predict_freight' in result_df.index else result_df.iloc[:, -1].values
                else:
                    y_vals = result_df.iloc[:, -1].values
                    
                fig = px.bar(
                    x=[f"${x:,.2f}" for x in dollars_list], 
                    y=y_vals,
                    labels={'x': 'Input Transaction ($)', 'y': 'Predicted Freight Cost ($)'},
                    color=y_vals,
                    color_continuous_scale="Teal",
                    title="Freight Cost Breakdown"
                )
                fig.update_layout(xaxis_type='category', plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")

def invoice_interface():
    st.markdown('<p class="main-header">Invoice Anomaly Flagger</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Detect irregular invoices requiring manual review using 5 key metrics.</p>', unsafe_allow_html=True)
    
    st.markdown("### Invoice Details")
    
    # Using columns for a clean, dashboard-like form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        inv_qty = st.number_input("Invoice Quantity", min_value=0.0, value=150.0, step=1.0, help="Total quantity listed on the specific invoice.")
        total_qty = st.number_input("Total Item Quantity", min_value=0.0, value=1500.0, step=1.0, help="Total historical or aggregated item quantity.")
        
    with col2:
        inv_dollars = st.number_input("Invoice Dollars ($)", min_value=0.0, value=2500.0, step=100.0, help="Dollar amount of the specific invoice.")
        total_dollars = st.number_input("Total Item Dollars ($)", min_value=0.0, value=25000.0, step=100.0, help="Total historical or aggregated dollar amount.")
        
    with col3:
        freight = st.number_input("Freight Amount ($)", min_value=0.0, value=350.0, step=10.0, help="Associated freight charges.")
        st.markdown("<br>", unsafe_allow_html=True) # Spacing alignment
        analyze_btn = st.button("Run Risk Analysis", type="primary")

    if analyze_btn:
        try:
            # Package the 5 required features
            input_data = {
                'invoice_Quantity': [inv_qty],
                'invoice_Dollars': [inv_dollars],
                'Freight': [freight],
                'total_item_Quantity': [total_qty],
                'total_item_Dollars': [total_dollars]
            }
            
            with st.spinner("Analyzing transaction patterns..."):
                result_df = predict_flag_invoice(input_data)
            
            st.divider()
            st.markdown("### Analysis Report")
            
            # Extract flag value securely despite .T backend logic
            if 'Predict_flag' in result_df.columns:
                flag_val = result_df['Predict_flag'].values[0]
            else:
                flag_val = result_df.iloc[:, -1].values[0]
            
            # Visual Result Representation
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                if flag_val == 1:
                    st.error("🚨 **ACTION REQUIRED**")
                    st.metric(label="Risk Status", value="FLAGGED", delta="High Anomaly Score", delta_color="inverse")
                    st.warning("This invoice deviates from standard patterns and requires manual review.")
                else:
                    st.success("✅ **CLEARED**")
                    st.metric(label="Risk Status", value="NORMAL", delta="Within Thresholds", delta_color="normal")
                    st.info("This invoice aligns with expected metrics. No further action needed.")
            
            with res_col2:
                # Simple visual representation of input distribution
                features = list(input_data.keys())
                values = [input_data[f][0] for f in features]
                
                fig = px.pie(
                    values=values, 
                    names=features, 
                    hole=0.4,
                    title="Feature Weight Distribution (Current Invoice)",
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(showlegend=False, margin=dict(t=40, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Analysis Error: {e}")
            st.info("Hint: Check your `predict_invoice_flag.py` backend. If your invoice model expects 5 features, ensure the `.T` (transpose) operation isn't distorting the dataframe shape before prediction.")

if __name__ == "__main__":
    main()