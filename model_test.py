import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import os

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Instamart Customer Analytics",
    page_icon="🛒",
    layout="wide"
)

# ---------------------------
# Load model and scaler
# ---------------------------
@st.cache_resource
def load_my_model_and_scaler():
    try:
        if os.path.exists("my_model.h5") and os.path.exists("scaler.pkl"):
            model = load_model("my_model.h5")
            with open("scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            return model, scaler
        else:
            st.error("Model file 'my_model.h5' or 'scaler.pkl' not found!")
            return None, None
    except Exception as e:
        st.error(f"Error loading model/scaler: {str(e)}")
        return None, None

model, scaler = load_my_model_and_scaler()

st.title('🛒 INSTAMART CUSTOMER ANALYTICS')
st.markdown("### Predict customer reorder behavior using Deep Learning")

if model is None or scaler is None:
    st.stop()

# ---------------------------
# Inputs
# ---------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Input Parameters")

    st.subheader("Order Information")
    order_dow = st.number_input('Day of Week (0=Sunday, 6=Saturday)', 0, 6, 0)
    order_hour_of_day = st.number_input('Hour of Day (0-23)', 0, 23, 10)
    days_since_prior_order = st.number_input('Days Since Prior Order', 0, 30, 7)

    st.subheader("Product Information")
    product_id = st.number_input('Product ID', 1, 49680, 1000)
    add_to_cart_order = st.number_input('Add to Cart Order', 1, 95, 1)
    aisle_id = st.number_input('Aisle ID', 1, 134, 1)
    department_id = st.number_input('Department ID', 1, 21, 1)

    st.subheader("Encoded Features")
    encode_dept = st.number_input('Encoded Department', value=0.5, format="%.4f")
    encode_aisle = st.number_input('Encoded Aisle', value=0.5, format="%.4f")
    encode_prod = st.number_input('Encoded Product', value=0.5, format="%.4f")

    st.subheader("Customer Behavior")
    avg_days_btw_orders = st.number_input('Average Days Between Orders', 1, 29, 7)
    total_orders = st.number_input('Total Orders', 1, 99, 10)
    reorder_ratio = st.number_input('Reorder Ratio', 0.0, 1.0, 0.5, step=0.01)
    number_of_times_reordered = st.number_input('Number of Times Reordered', 0, None, 0)
    No_time_reord = st.number_input('No Time Reordered', 0, None, 0)

with col2:
    st.header("Prediction Results")

    if st.button("🔮 Predict Reorder Probability"):
        try:
            # Prepare and scale input
            input_features = np.array([[
                order_dow, order_hour_of_day, days_since_prior_order, product_id,
                add_to_cart_order, aisle_id, department_id, encode_dept,
                encode_aisle, encode_prod, avg_days_btw_orders, total_orders,
                reorder_ratio, number_of_times_reordered, No_time_reord
            ]])

            # Scale using saved scaler
            scaled_input = scaler.transform(input_features)

            # Predict
            with st.spinner("Making prediction..."):
                prediction = model.predict(scaled_input, verbose=0)

            # Softmax output: [prob_class_0, prob_class_1]
            prob_class_1 = prediction[0][1]  # probability of 'Reorder'
            prob_class_0 = prediction[0][0]

            st.success("✅ Prediction Complete!")

            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric("Reorder Probability", f"{prob_class_1:.4f}")
            with col_metric2:
                # Use your 0.21 threshold from training
                reorder_decision = "Yes" if prob_class_1 > 0.21 else "No"
                st.metric("Will Reorder?", reorder_decision)

            st.subheader("Prediction Confidence")
            st.progress(min(prob_class_1, 1.0))

            st.subheader("Interpretation")
            if prob_class_1 > 0.7:
                st.success("🟢 High likelihood of reorder - Strong customer loyalty indicated")
            elif prob_class_1 > 0.5:
                st.warning("🟡 Moderate likelihood of reorder - Customer shows some interest")
            else:
                st.info("🔴 Low likelihood of reorder - May need marketing intervention")

            st.subheader("Input Summary")
            feature_df = pd.DataFrame({
                'Feature': [
                    'Day of Week','Hour of Day','Days Since Prior','Product ID',
                    'Cart Order','Aisle ID','Department ID','Avg Days Between Orders',
                    'Total Orders','Reorder Ratio','Times Reordered','Times Not Reordered'
                ],
                'Value': [
                    order_dow, order_hour_of_day, days_since_prior_order, product_id,
                    add_to_cart_order, aisle_id, department_id, avg_days_btw_orders,
                    total_orders, reorder_ratio, number_of_times_reordered, No_time_reord
                ]
            })
            st.dataframe(feature_df, use_container_width=True)

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
