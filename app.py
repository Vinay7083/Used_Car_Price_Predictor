import streamlit as st
import pandas as pd
import joblib
import sklearn

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('random_forest_model1.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# Check model compatibility
if model:
    st.sidebar.write("✅ Model loaded successfully!")
    expected_features = getattr(model, 'feature_names_in_', [])
else:
    expected_features = []

# Streamlit App UI
st.title("🚗 Used Car Price Estimator")
st.write("Fill in the details below to predict the car price.")

# Create input fields
user_input = {}
columns = ["year", "mileage", "engine_size", "horsepower", "fuel_type", "brand"]  # Adjust as per dataset

for col in columns:
    if col in ["year", "mileage", "horsepower", "engine_size"]:
        user_input[col] = st.number_input(f"Enter {col.replace('_', ' ').title()}", min_value=0, step=1)
    else:
        user_input[col] = st.selectbox(f"Select {col.replace('_', ' ').title()}", ["Toyota", "Ford", "BMW", "Honda"])  # Modify options

# Convert input to DataFrame
single_entry_df = pd.DataFrame([user_input])

# Ensure input feature order matches model
if expected_features:
    single_entry_df = single_entry_df.reindex(columns=expected_features, fill_value=0)

# Display features for debugging
st.write("### Debugging Info")
st.write("**Expected Features:**", expected_features)
st.write("**Input Features:**", single_entry_df.columns.tolist())

# Prediction button
if st.button("Predict Car Price"):
    if model:
        try:
            prediction = model.predict(single_entry_df)
            st.success(f"Predicted Car Price: ${prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    else:
        st.error("Model not loaded. Check for errors above.")

# Show environment details
st.sidebar.write("**Scikit-learn Version:**", sklearn.__version__)
