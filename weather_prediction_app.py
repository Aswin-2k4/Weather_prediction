import streamlit as st
import joblib

# ---------- Encoding maps ----------
cloud_map = {
    'Clear': [1, 0, 0, 0],
    'Cloudy': [0, 1, 0, 0],
    'Overcast': [0, 0, 1, 0],
    'Partly cloudy': [0, 0, 0, 1]
}

location_map = {
    'Coastal': [1, 0, 0],
    'Inland': [0, 1, 0],
    'Mountain': [0, 0, 1]
}

season_map = {
    'Autumn': [1, 0, 0, 0],
    'Spring': [0, 1, 0, 0],
    'Summer': [0, 0, 1, 0],
    'Winter': [0, 0, 0, 1]
}


st.title("Weather predictor")
cloud_nature=st.radio('Condition of cloud',['Select an option','Clear','Cloudy','Overcast','Partly cloudy'])
if cloud_nature!='Select an option':
   st.write(f"Cloud condition chosen : {cloud_nature}")

location=st.radio('Location',['Select an option','Coastal','Inland','Mountain'])
if location!='Select an option':
   st.write(f"Location chosen : {location}")

season=st.radio('Season',['Select an option','Autumn','Spring','Summer','Winter'])
if season!='Select an option':
   st.write(f"Season chosen : {season}")

temp=st.slider('Temperature in celsius',-20,50)

humidity=st.number_input('Humidity')
wind_speed=st.number_input('Wind Speed')
precipitation=st.number_input('Precipitation in % ')

atm_pressure=st.slider('Atmospheric pressure',800,1300)
uv_index=st.slider('UV index',0,20)
visibility=st.slider('Visibility (km) ',1,20)

if cloud_nature != 'Select an option' and location != 'Select an option' and season != 'Select an option':

    cloud_encoded = cloud_map[cloud_nature]
    location_encoded = location_map[location]
    season_encoded = season_map[season]

    final_input = (
        cloud_encoded +
        location_encoded +
        season_encoded +
        [temp, humidity, wind_speed, precipitation,
         atm_pressure, uv_index, visibility]
    )

if st.button('Predict'):
    if cloud_nature == 'Select an option' or location == 'Select an option' or season == 'Select an option':
        st.warning("Please select all categorical options")
    else:
        model = joblib.load('Model/random_forest.pkl')
        output = model.predict([final_input])
        if output==0:
            st.success("Cloudy")
        elif output==1:
            st.success("Rainy")
        elif output==2:
            st.success("Snowy")
        elif output==3:
            st.success("Sunny")

