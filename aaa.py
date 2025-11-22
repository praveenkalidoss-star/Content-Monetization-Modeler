import streamlit as st
import numpy as np
import pandas as pd
import pickle

# -------------------------------
# Load Saved Model & Scaler
# -------------------------------
model = pickle.load(open("final_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("📊 YouTube Ad Revenue Prediction App")
st.write("Predict YouTube video ad revenue")

# -------------------------------
# Category / Device / Country Mapping
# -------------------------------
category_map = {
    "Entertainment": 0,
    "Gaming": 1,
    "Education": 2,
    "Tech": 3,
    "Comedy": 4,
    "Music": 5
}

device_map = {
    "Mobile": 0,
    "Desktop": 1,
    "Tablet": 2,
    "TV": 3
}

country_map = {
    "India": 0,
    "USA": 1,
    "UK": 2,
    "Canada": 3,
    "Australia": 4,
    "Germany": 5
}

day_of_week_map = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}


# -------------------------------
# User Inputs
# -------------------------------
st.header("Enter Video Details")

views = st.number_input("Views", min_value=0, value=10000)
likes = st.number_input("Likes", min_value=0, value=500)
comments = st.number_input("Comments", min_value=0, value=100)
engagement_rate = (likes + comments) / views if views > 0 else 0

watch_time_minutes = st.number_input("Watch Time (minutes)", min_value=0.0, value=10000.0)
video_length_minutes = st.number_input("Video Length (minutes)", min_value=0.0, value=10.0)
subscribers = st.number_input("Subscribers Count", min_value=0, value=100000)

# -------------------------------
# Replace numeric dropdown with text labels
# -------------------------------
category_label = st.selectbox("Category", list(category_map.keys()))
device_label = st.selectbox("Device", list(device_map.keys()))
country_label = st.selectbox("Country", list(country_map.keys()))

# Convert text → numeric for model
category = category_map[category_label]
device = device_map[device_label]
country = country_map[country_label]

year = st.selectbox("Year", [2023, 2024, 2025])
month = st.selectbox("Month", list(range(1, 13)))
day = st.selectbox("Day", list(range(1, 32)))
day_of_week_label = st.selectbox("Day of Week", list(day_of_week_map.keys()))
day_of_week = day_of_week_map[day_of_week_label]
hour = st.selectbox("Hour (0-23)", list(range(0, 24)))
is_weekend = st.selectbox("Is Weekend? (0=No, 1=Yes)", [0, 1])

# -------------------------------
# Create DataFrame for prediction
# -------------------------------
input_data = pd.DataFrame({
    "views": [views],
    "likes": [likes],
    "comments": [comments],
    "Engagement_Rate": [engagement_rate],
    "watch_time_minutes": [watch_time_minutes],
    "video_length_minutes": [video_length_minutes],
    "subscribers": [subscribers],
    "category": [category],   # mapped numeric
    "device": [device],       # mapped numeric
    "country": [country],     # mapped numeric
    "year": [year],
    "month": [month],
    "day": [day],
    "day_of_week": [day_of_week],
    "hour": [hour],
    "is_weekend": [is_weekend],
})

st.write("### Input Summary")
st.dataframe(input_data)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Revenue"):
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.success(f"Estimated Ad Revenue: **${prediction:.2f} USD**")
