import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder
import PIL
from PIL import Image  # Required to process the image
import joblib

# Initialize expected_features to None
expected_features = None

# Load the trained model
#try:
    # model = pickle.load(open('random_forest_model1.pkl', 'rb'))
    #model = joblib.load('random_forest_model1.pkl')
    # if hasattr(model, 'feature_names_in_'):
    #     expected_features = model.feature_names_in_
    # else:
    #     st.warning("Model does not have 'feature_names_in_' attribute. Expected features are unknown.")
#except Exception as e:
    #st.error(f"Error loading model: {str(e)}")
    #model = None  # Ensure model is set to None if loading fails

# Title
st.markdown(
    """
    <style>
        .title {
             background-color: #060270;  /* Set your desired background color here */
            color: White;                   /* Set text color */
            padding: 15px;              /* Add some padding around the title */
            font-size: 24px;            /* Set font size */
            font-weight: bold; 
            text-align: center;         /* Center the text */
             [data-testid="stSidebar"] {
             background-color: #56A1EB !important;  
        }
        div.stButton > button:first-child {
            background-color: #060270; /* Change this to your preferred color */
            color: white; /* Text color */
            border-radius: 10px;
            padding: 10px 24px;
            font-size: 16px;
            font-weight: bold;
            border: none;
        }
        div.stButton > button:first-child:hover {
            background-color: #042052; /* Slightly darker shade for hover effect */
        }
    </style>
    <div class="title">
        Used Car Price Prediction
    </div>
    """, unsafe_allow_html=True)
    # Adding a custom header in the sidebar with a background color
st.sidebar.markdown(
    """
    <style>
        .sidebar-header {
            background-color: #060270;  /* Set your desired background color here */
            color: White;               /* Set text color */
           padding: 10px 10px 15px;              /* Add some padding around the title */
            font-size: 18px;            /* Set font size */
            text-align: center;         /* Center the text */
            font-weight: bold;          /* Make text bold */
             [data-testid="stSidebar"] {
             background-color: #56A1EB !important;   /* Change to your preferred color */
        }                                               
        }
    </style>
    <div class="sidebar-header">
        Enter Feature Details
    </div>
    """, unsafe_allow_html=True)


st.write("Enter the car details below to predict its final price.")

# # **Image Upload**
# st.subheader("Upload Car Image (Optional)")
# uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_image is not None:
#     # Open the image using PIL
#     image = Image.open(uploaded_image)
#     st.image(image, caption='Uploaded Image', use_container_width=True)
#     st.write("")
#     st.write("Image uploaded successfully!")

# # User Inputs (Modify based on expected features)
col1, col2 = st.sidebar.columns(2)  # Split the sidebar into two columns

with col1:
    wheel_system = st.selectbox("Wheel System", ['FWD', 'AWD', '4WD', 'RWD' ,'4X2'])
    body_type = st.selectbox("Body Type", ['Sedan' ,'Coupe' ,'SUV / Crossover', 'Pickup Truck', 'Wagon', 'Minivan','Convertible', 'Hatchback', 'Van'])
    fuel_type = st.selectbox("Fuel Type", ['Gasoline', 'Flex Fuel Vehicle', 'Diesel', 'Hybrid', 'Biodiesel','Compressed Natural Gas', 'Propane'])
    maximum_seating  = st.selectbox("Maximum Seating", ['5 seats', '4 seats', '8 seats', '7 seats' ,'6 seats', '15 seats', '3 seats','9 seats', '10 seats', '2 seats', '12 seats'])
    engine_cylinders =  st.selectbox("Engine Cylinders",['I4', 'V6', 'V8', 'H4', 'I6', 'V8 Flex Fuel Vehicle', 'V6 Flex Fuel Vehicle','I4 Diesel', 'I6 Diesel', 'I5' ,'I3', 'I4 Hybrid' ,'I4 Flex Fuel Vehicle',
    'V6 Diesel' ,'V6 Hybrid', 'V6 Biodiesel', 'V12', 'W12', 'H6', 'W12 Flex Fuel Vehicle' ,'H4 Hybrid', 'I4 Compressed Natural Gas', 'V10','I6 Hybrid', 'R2', 'V8 Propane', 'V6 Compressed Natural Gas'])
    make_name =  st.selectbox("Make Name", ['Chevrolet', 'Lexus', 'Jeep', 'Hyundai', 'Cadillac', 'Chrysler', 'Dodge','Nissan', 'Honda', 'Kia' ,'Ford', 'Lincoln', 'Subaru' ,'BMW' ,'Audi',
 'Mercedes-Benz', 'Volkswagen' ,'Jaguar', 'Mazda', 'GMC', 'Toyota' ,'Acura','INFINITI', 'RAM', 'Buick', 'Land Rover', 'Mitsubishi', 'Volvo', 'Genesis','Saab', 'Bentley', 'MINI', 'Alfa Romeo', 'FIAT', 'Rolls-Royce', 
 'Scion', 'Porsche', 'Saturn', 'Maserati', 'Pontiac', 'Mercury' ,'Oldsmobile', 'Aston Martin' ,'Suzuki', 'Isuzu', 'Hummer', 'Plymouth'])

    # listing_color  =  st.selectbox("Listing Color",  ['SILVER', 'BLACK', 'RED', 'WHITE', 'UNKNOWN', 'BLUE', 'GRAY', 'BROWN', 'YELLOW', 'ORANGE', 'GREEN' ,'PURPLE', 'TEAL' ,'GOLD', 'PINK'])

    engine_displacement  = st.number_input("Engine Displacement ", min_value=1000.0, max_value=9000.0)
    length  = st.number_input("Length ", min_value=120.0, max_value=280.0)
with col2:

        
    horsepower  = st.number_input("Horsepower", min_value=1.0, max_value=1000.0)
    mileage  = st.number_input("Mileage ", min_value=1.0, max_value=1225238.0)
    fuel_tank_volume = st.number_input("Fuel Tank Volume (Liters    )", min_value=1.0, max_value=50.0)
    city_fuel_economy  = st.number_input("City Fuel Economy ", min_value=1.0, max_value=100.0)
    highway_fuel_economy  = st.number_input("Highway Fuel Economy ", min_value=1.0, max_value=100.0)
    listing_color  =  st.selectbox("Listing Color",  ['SILVER', 'BLACK', 'RED', 'WHITE', 'UNKNOWN', 'BLUE', 'GRAY', 'BROWN', 'YELLOW', 'ORANGE', 'GREEN' ,'PURPLE', 'TEAL' ,'GOLD', 'PINK'])
    wheelbase  = st.number_input("Wheel Base ", min_value= 70.0, max_value=180.0)
    # engine_displacement  = st.number_input("Engine Displacement ", min_value=1000.0, max_value=9000.0)
    # length  = st.number_input("Length ", min_value=120.0, max_value=280.0)
    height   = st.number_input("Height  ", min_value=40.0, max_value=120.0)

# Create user input DataFrame for all features
user_data = pd.DataFrame({
    "horsepower": [horsepower],
    "city_fuel_economy": [city_fuel_economy],
    "highway_fuel_economy": [highway_fuel_economy],
    "fuel_tank_volume": [fuel_tank_volume],
    "length": [length],
    "height": [height],
    "engine_displacement": [engine_displacement],
    "mileage": [mileage],
    "wheelbase":[ wheelbase],
    "body_type": [body_type],
    "make_name": [make_name],
    "fuel_type": [fuel_type],
    "maximum_seating": [maximum_seating],
    "engine_cylinders": [engine_cylinders],
    "listing_color": [listing_color],
    "Wheel System": [wheel_system]  
})
st.subheader("Entered Car Details")
st.dataframe(user_data)

df = pd.get_dummies(user_data, columns=['Wheel System','body_type','fuel_type','maximum_seating','engine_cylinders','make_name','listing_color'],dtype=int)

expected_features=['horsepower', 'mileage', 'fuel_tank_volume', 'city_fuel_economy', 'highway_fuel_economy', 'wheelbase', 'length', 'engine_displacement', 'height', 'wheel_system_4X2', 'wheel_system_AWD', 'wheel_system_FWD', 'wheel_system_RWD', 'body_type_Coupe', 'body_type_Hatchback', 'body_type_Minivan', 'body_type_Pickup Truck', 'body_type_SUV / Crossover', 'body_type_Sedan', 'body_type_Van', 'body_type_Wagon', 'fuel_type_Compressed Natural Gas', 'fuel_type_Diesel', 'fuel_type_Flex Fuel Vehicle', 'fuel_type_Gasoline', 'fuel_type_Hybrid', 'fuel_type_Propane', 'maximum_seating_12 seats', 'maximum_seating_15 seats', 'maximum_seating_2 seats', 'maximum_seating_3 seats', 'maximum_seating_4 seats', 'maximum_seating_5 seats', 'maximum_seating_6 seats', 'maximum_seating_7 seats', 'maximum_seating_8 seats', 'maximum_seating_9 seats', 'engine_cylinders_H4 Hybrid', 'engine_cylinders_H6', 'engine_cylinders_I3', 'engine_cylinders_I4', 'engine_cylinders_I4 Compressed Natural Gas', 'engine_cylinders_I4 Diesel', 'engine_cylinders_I4 Flex Fuel Vehicle', 'engine_cylinders_I4 Hybrid', 'engine_cylinders_I5', 'engine_cylinders_I6', 'engine_cylinders_I6 Diesel', 'engine_cylinders_I6 Hybrid', 'engine_cylinders_R2', 'engine_cylinders_V10', 'engine_cylinders_V12', 'engine_cylinders_V6', 'engine_cylinders_V6 Biodiesel', 'engine_cylinders_V6 Compressed Natural Gas', 'engine_cylinders_V6 Diesel', 'engine_cylinders_V6 Flex Fuel Vehicle', 'engine_cylinders_V6 Hybrid', 'engine_cylinders_V8', 'engine_cylinders_V8 Flex Fuel Vehicle', 'engine_cylinders_V8 Propane', 'engine_cylinders_W12', 'engine_cylinders_W12 Flex Fuel Vehicle', 'make_name_Alfa Romeo', 'make_name_Aston Martin', 'make_name_Audi', 'make_name_BMW', 'make_name_Bentley', 'make_name_Buick', 'make_name_Cadillac', 'make_name_Chevrolet', 'make_name_Chrysler', 'make_name_Dodge', 'make_name_FIAT', 'make_name_Ford', 'make_name_GMC', 'make_name_Genesis', 'make_name_Honda', 'make_name_Hummer', 'make_name_Hyundai', 'make_name_INFINITI', 'make_name_Isuzu', 'make_name_Jaguar', 'make_name_Jeep', 'make_name_Kia', 'make_name_Land Rover', 'make_name_Lexus', 'make_name_Lincoln', 'make_name_MINI', 'make_name_Maserati', 'make_name_Mazda', 'make_name_Mercedes-Benz', 'make_name_Mercury', 'make_name_Mitsubishi', 'make_name_Nissan', 'make_name_Oldsmobile', 'make_name_Plymouth', 'make_name_Pontiac', 'make_name_Porsche', 'make_name_RAM', 'make_name_Rolls-Royce', 'make_name_Saab', 'make_name_Saturn', 'make_name_Scion', 'make_name_Subaru', 'make_name_Suzuki', 'make_name_Toyota', 'make_name_Volkswagen', 'make_name_Volvo', 'listing_color_BLUE', 'listing_color_BROWN', 'listing_color_GOLD', 'listing_color_GRAY', 'listing_color_GREEN', 'listing_color_ORANGE', 'listing_color_PINK', 'listing_color_PURPLE', 'listing_color_RED', 'listing_color_SILVER', 'listing_color_TEAL', 'listing_color_UNKNOWN', 'listing_color_WHITE', 'listing_color_YELLOW']

# Ensure all expected features are present in df
# for col in expected_features:
#     if col not in df.columns:
#         df[col] = 0  # Add missing columns with 0 values

# Create a DataFrame with all expected features
single_entry_df = pd.DataFrame([{col: df.get(col, 0) for col in expected_features}])

# Keep only expected features (avoid extra columns)
#df = df[expected_features]

# # Apply OneHotEncoding to categorical features
# encoder = OneHotEncoder(drop='first', sparse_output=False)
# encoded_categorical_data = encoder.fit_transform(categorical_data)

# # Create DataFrame for encoded categorical data
# encoded_df = pd.DataFrame(encoded_categorical_data, columns=encoder.get_feature_names_out(categorical_data.columns))

# # Concatenate numeric and categorical features
# user_data = pd.concat([user_data, encoded_df], axis=1)

# **Ensure feature order matches the model**
# if expected_features is not None:
#     # Align columns to match the model's expected features
#     missing_columns = set(expected_features) - set(user_data.columns)
#     if missing_columns:
#         st.warning(f"Missing expected features: {missing_columns}")
#         # Add missing columns with default values (e.g., 0)
#         for col in missing_columns:
#             user_data[col] = 0
#     extra_columns = set(user_data.columns) - set(expected_features)
#     if extra_columns:
#         st.warning(f"Dropping extra columns: {extra_columns}")
#         user_data = user_data.drop(columns=extra_columns)
    
#     # Reorder the columns to match the model's expected feature order
#     user_data = user_data[expected_features]

# Display entered data


# Prediction
if st.button("Predict Car Price"):
    try:
        model = joblib.load('random_forest_model1.pkl')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        model = None  # Ensure model is set to None if loading fails
    
    if model is None:
        st.error("Model is not loaded. Cannot make predictions.")
    else:
        try:
            # Make prediction
            predicted_price = model.predict(single_entry_df)
            st.success(f'Predicted Car Price: ${np.round(predicted_price[0], 2)}')
        except Exception as e:  
            st.error(f"Prediction Error: {str(e)}")
