import streamlit as st
from preprocessing import feature_engineering, preprocess_data
import pandas as pd
import pickle

st.title("ASG 04 MD - Callista Gianna Then - Spaceship Titanic Model Deployment") 

#model sm encoding
with open("logistic_model_optimized.pkl", "rb") as f:model = pickle.load(f)
with open("label_encoders.pkl", "rb") as f:encoders = pickle.load(f)

#input
Age = st.number_input("Age", value=25)
HomePlanet = st.selectbox("HomePlanet", ["Earth","Europa","Mars"])
CryoSleep = st.selectbox("CryoSleep", [True, False])
Destination = st.selectbox("Destination", ["TRAPPIST-1e","55 Cancri e","PSO J318.5-22"])
VIP = st.selectbox("VIP", [True, False])
RoomService = st.number_input("RoomService", 0)
FoodCourt = st.number_input("FoodCourt", 0)
ShoppingMall = st.number_input("ShoppingMall", 0)
Spa = st.number_input("Spa", 0)
VRDeck = st.number_input("VRDeck", 0)
Deck = st.text_input("Deck", "B")
Side = st.selectbox("Side", ["P","S"])



# PassengerId placeholder
PassengerId = st.text_input("Passenger ID", "G_001")


input_data = pd.DataFrame({
    "PassengerId": [PassengerId],
    "Name": ["Unknown"],  
    "Age": [Age],
    "HomePlanet": [HomePlanet],
    "CryoSleep": [CryoSleep],
    "Destination": [Destination],
    "VIP": [VIP],
    "RoomService": [RoomService],
    "FoodCourt": [FoodCourt],
    "ShoppingMall": [ShoppingMall],
    "Spa": [Spa],
    "VRDeck": [VRDeck],
    "Cabin": ["Unknown/0/Unknown"],
    "Deck": [Deck],
    "Side": [Side]
})


input_data = feature_engineering(input_data)

#encoding per row 
for col, encoder in encoders.items():
    if col in input_data.columns:
        input_data[col] = encoder.transform(input_data[col].astype(str))


for col in model.feature_names_in_:
    if col not in input_data.columns:
        input_data[col] = 0 


input_data = input_data[model.feature_names_in_]

if st.button("Predict"):
    prediction = model.predict(input_data)
    
    #tadi hanya buat debugging
    #prob = model.predict_proba(input_data)
    #st.write(prob)
    
    if prediction[0] == 1:
        st.success("Passenger Transported")
    else:
        st.error("Passenger Not Transported")