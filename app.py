import streamlit as st
import pickle
import numpy as np
import pandas as pd
import json

model = pickle.load(open("/Users/nandapop/Documents/Bootcamp/Ml_web_streamlit/models/classificador_random_forest_with_encoders.sav", "rb"))
df = pd.read_csv("/Users/nandapop/Documents/Bootcamp/Ml_web_streamlit/data/world_AQI.csv")
df_encoded = pd.read_csv("/Users/nandapop/Documents/Bootcamp/Ml_web_streamlit/data/df_enconded.csv")

classifier = model['classifier']
country_encoder = model['country_encoder']
city_encoder = model['city_encoder']

#st.write("Loaded model components:", model.keys())

aqi_category_mapping = {
    0: {'label': 'Good', 'color': 'green'},
    1: {'label': 'Moderate', 'color': 'yellow'},
    2: {'label': 'Unhealthy for Sensitive Groups', 'color': 'orange'},
    3: {'label': 'Unhealthy', 'color': 'red'},
    4: {'label': 'Very Unhealthy', 'color': 'purple'},
    5: {'label': 'Hazardous', 'color': 'maroon'}
}

co_aqi_category_mapping = {
    0: {'label': 'Good', 'color': 'green'},
    1: {'label': 'Moderate', 'color': 'yellow'},
    2: {'label': 'Unhealthy for Sensitive Groups', 'color': 'orange'},
    3: {'label': 'Unhealthy', 'color': 'red'},
    4: {'label': 'Very Unhealthy', 'color': 'purple'}
}

ozone_aqi_category_mapping = {
    0: {'label': 'Good', 'color': 'green'},
    1: {'label': 'Moderate', 'color': 'yellow'},
    2: {'label': 'Unhealthy for Sensitive Groups', 'color': 'orange'},
    3: {'label': 'Unhealthy', 'color': 'red'},
    4: {'label': 'Very Unhealthy', 'color': 'purple'}
}

no2_aqi_category_mapping = {
    0: {'label': 'Good', 'color': 'green'},
    1: {'label': 'Moderate', 'color': 'yellow'}
}

pm25_aqi_category_mapping = {
    0: {'label': 'Good', 'color': 'green'},
    1: {'label': 'Moderate', 'color': 'yellow'},
    2: {'label': 'Unhealthy for Sensitive Groups', 'color': 'orange'},
    3: {'label': 'Unhealthy', 'color': 'red'},
    4: {'label': 'Very Unhealthy', 'color': 'purple'},
    5: {'label': 'Hazardous', 'color': 'maroon'}
}

def get_locations():
    df['Country'] = df['Country'].fillna('Unknown').astype(str)
    df['City'] = df['City'].fillna('Unknown').astype(str)

    countries_cities = {}

    for country in df['Country'].unique():
        cities = sorted(df[df['Country'] == country]['City'].unique().tolist())
        countries_cities[country] = cities
    return countries_cities

st.title("Select a Country and City for AQI Prediction")

locations = get_locations()

selected_country = st.selectbox("Select a country", options=list(locations.keys()))

if selected_country:
    cities = locations[selected_country]
    selected_city = st.selectbox("Select a city ",options=cities)    

def predict(selected_country, selected_city):
    encoded_country = country_encoder.transform([selected_country])[0]
    encoded_city = city_encoder.transform([selected_city])[0]
    
    data = df_encoded[(df_encoded['Country'] == encoded_country) & 
                                (df_encoded['City'] == encoded_city)]

    if data.empty:
        st.error(f"No data found for city {selected_city}, country {selected_country}")
        return 
   
    aqi_value = data['AQI Value'].values[0]
    co_aqi_value = data['CO AQI Value'].values[0]
    ozone_aqi_value = data['Ozone AQI Value'].values[0]
    no2_aqi_value = data['NO2 AQI Value'].values[0]
    pm25_aqi_value = data['PM2.5 AQI Value'].values[0]
    
    co_aqi_category_value = data['CO AQI Category'].values[0]
    ozone_aqi_category_value = data['Ozone AQI Category'].values[0]
    no2_aqi_category_value = data['NO2 AQI Category'].values[0]
    pm25_aqi_category_value = data['PM2.5 AQI Category'].values[0]
    
    feature_names = [
        'AQI Value',
        'CO AQI Value',
        'CO AQI Category',
        'Ozone AQI Value',
        'Ozone AQI Category',
        'NO2 AQI Value',
        'NO2 AQI Category',
        'PM2.5 AQI Value',
        'PM2.5 AQI Category'
    ]
    
    feature_values = [[
        aqi_value,
        co_aqi_value,
        co_aqi_category_value,
        ozone_aqi_value,
        ozone_aqi_category_value,
        no2_aqi_value,
        no2_aqi_category_value,
        pm25_aqi_value,
        pm25_aqi_category_value
    ]]
    features = np.array([[encoded_country, encoded_city, aqi_value, co_aqi_value, ozone_aqi_value, no2_aqi_value, pm25_aqi_value, co_aqi_category_value, ozone_aqi_category_value]])
    features_df = pd.DataFrame(feature_values, columns=feature_names)

    #st.write("Features for prediction:", features.shape)
    #st.write("Model's expected features:", classifier.feature_names_in_)
    
    try:
        prediction = classifier.predict(features_df)
        predicted_aqi_category = prediction[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return
    
    category_info = aqi_category_mapping.get(predicted_aqi_category, {'label': 'Unknown', 'color': 'black'})
    category_label = category_info['label']
    category_color = category_info['color']
    st.write(f"AQI Category:")
    st.markdown(f"<span style='color:{category_color}; font-weight:bold;'>{category_label}</span>", unsafe_allow_html=True)
    
    co_aqi_category_info = co_aqi_category_mapping.get(co_aqi_category_value, {'label': 'Unknown', 'color': 'black'})
    co_aqi_category_label = co_aqi_category_info['label']
    co_aqi_category_color = co_aqi_category_info['color']
    st.write(f"CO AQI Category:")
    st.markdown(f"<span style='color:{co_aqi_category_color}; font-weight:bold;'>{co_aqi_category_label}</span>", unsafe_allow_html=True)

    ozone_aqi_category_info = ozone_aqi_category_mapping.get(ozone_aqi_category_value, {'label': 'Unknown', 'color': 'black'})
    ozone_aqi_category_label = ozone_aqi_category_info['label']
    ozone_aqi_category_color = ozone_aqi_category_info['color']
    st.write(f"Ozone AQI Category:")
    st.markdown(f"<span style='color:{ozone_aqi_category_color}; font-weight:bold;'>{ozone_aqi_category_label}</span>", unsafe_allow_html=True)

    no2_aqi_category_info = no2_aqi_category_mapping.get(no2_aqi_category_value, {'label': 'Unknown', 'color': 'black'})
    no2_aqi_category_label = no2_aqi_category_info['label']
    no2_aqi_category_color = no2_aqi_category_info['color']
    st.write(f"NO2 AQI Category:")
    st.markdown(f"<span style='color:{no2_aqi_category_color}; font-weight:bold;'>{no2_aqi_category_label}</span>", unsafe_allow_html=True)

    pm25_aqi_category_info = pm25_aqi_category_mapping.get(pm25_aqi_category_value, {'label': 'Unknown', 'color': 'black'})
    pm25_aqi_category_label = pm25_aqi_category_info['label']
    pm25_aqi_category_color = pm25_aqi_category_info['color']
    st.write(f"PM2.5 AQI Category:")
    st.markdown(f"<span style='color:{pm25_aqi_category_color}; font-weight:bold;'>{pm25_aqi_category_label}</span>", unsafe_allow_html=True)
        
if st.button("Predict AQI"):
    if selected_country and selected_city:
        predict(selected_country, selected_city)