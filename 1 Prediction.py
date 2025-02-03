import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
st.title('Dataset')
dataframe = pd.read_csv('./dataset.csv')
dataframe

st.title('Dataset After Label Encoding')
label_enc=preprocessing.LabelEncoder()
location_enc=label_enc.fit_transform(dataframe['Location'])
measure_enc=label_enc.fit_transform(dataframe['Measure'])
gender_encoded=label_enc.fit_transform(dataframe['Gender'])
cause_encoded=label_enc.fit_transform(dataframe['Cause'])
X=np.array(list(zip(location_enc,dataframe['Year'],measure_enc,gender_encoded,cause_encoded,dataframe['PM Concentration'],dataframe['Rate Per 100000 Population'])))
dataframe = pd.DataFrame(data = X, 
                        columns = ["Location", "Year", "Measure", "Gender", "Cause", "PM Concentration", "Rate Per 100000 Population"])
dataframe

X = dataframe.iloc[:,0:6]
y = dataframe['Rate Per 100000 Population']



# Take Input
st.title('User Input')
loc_options = {
    "Afghanistan": 0,
    "Albania": 1,
    "Algeria": 2,
    "Andorra": 3,
    "Angola": 4,
    "Antigua and Barbuda": 5,
    "Argentina": 6,
    "Armenia": 7,
    "Australia": 8,
    "Austria": 9,
    "Azerbaijan": 10,
    "Bahamas": 11,
    "Bahrain": 12,
    "Bangladesh": 13,
    "Barbados": 14,
    "Belarus": 15,
    "Belgium": 16,
    "Belize": 17,
    "Benin": 18,
    "Bhutan": 19,
    "Bolivia (Plurinational State of)": 20,
    "Bosnia and Herzegovina": 21,
    "Botswana": 22,
    "Brazil": 23,
    "Brunei Darussalam": 24,
    "Bulgaria": 25,
    "Burkina Faso": 26,
    "Burundi": 27,
    "Cabo Verde": 28,
    "Cambodia": 29,
    "Cameroon": 30,
    "Canada": 31,
    "Central African Republic": 32,
    "Chad": 33,
    "Chile": 34,
    "China": 35,
    "Colombia": 36,
    "Comoros": 37,
    "Congo": 38,
    "Cook Islands": 39,
    "Costa Rica": 40,
    "Côte d'Ivoire": 41,
    "Croatia": 42,
    "Cuba": 43,
    "Cyprus": 44,
    "Czechia": 45,
    "Democratic People's Republic of Korea": 46,
    "Democratic Republic of the Congo": 47,
    "Denmark": 48,
    "Djibouti": 49,
    "Dominica": 50,
    "Dominican Republic": 51,
    "Ecuador": 52,
    "Egypt": 53,
    "El Salvador": 54,
    "Equatorial Guinea": 55,
    "Eritrea": 56,
    "Estonia": 57,
    "Eswatini": 58,
    "Ethiopia": 59,
    "Fiji": 60,
    "Finland": 61,
    "France": 62,
    "Gabon": 63,
    "Gambia": 64,
    "Georgia": 65,
    "Germany": 66,
    "Ghana": 67,
    "Greece": 68,
    "Grenada": 69,
    "Guatemala": 70,
    "Guinea": 71,
    "Guinea-Bissau": 72,
    "Guyana": 73,
    "Haiti": 74,
    "Honduras": 75,
    "Hungary": 76,
    "Iceland": 77,
    "India": 78,
    "Indonesia": 79,
    "Iran (Islamic Republic of)": 80,
    "Iraq": 81,
    "Ireland": 82,
    "Israel": 83,
    "Italy": 84,
    "Jamaica": 85,
    "Japan": 86,
    "Jordan": 87,
    "Kazakhstan": 88,
    "Kenya": 89,
    "Kiribati": 90,
    "Kuwait": 91,
    "Kyrgyzstan": 92,
    "Lao People's Democratic Republic": 93,
    "Latvia": 94,
    "Lebanon": 95,
    "Lesotho": 96,
    "Liberia": 97,
    "Libya": 98,
    "Lithuania": 99,
    "Luxembourg": 100,
    "Madagascar": 101,
    "Malawi": 102,
    "Malaysia": 103,
    "Maldives": 104,
    "Mali": 105,
    "Malta": 106,
    "Marshall Islands": 107,
    "Mauritania": 108,
    "Mauritius": 109,
    "Mexico": 110,
    "Micronesia (Federated States of)": 111,
    "Monaco": 112,
    "Mongolia": 113,
    "Montenegro": 114,
    "Morocco": 115,
    "Mozambique": 116,
    "Myanmar": 117,
    "Namibia": 118,
    "Nauru": 119,
    "Nepal": 120,
    "Netherlands": 121,
    "New Zealand": 122,
    "Nicaragua": 123,
    "Niger": 124,
    "Nigeria": 125,
    "Niue": 126,
    "Norway": 127,
    "Oman": 128,
    "Pakistan": 129,
    "Palau": 130,
    "Palestine": 131,
    "Panama": 132,
    "Papua New Guinea": 133,
    "Paraguay": 134,
    "Peru": 135,
    "Philippines": 136,
    "Poland": 137,
    "Portugal": 138,
    "Qatar": 139,
    "Republic of Korea": 140,
    "Republic of Moldova": 141,
    "Romania": 142,
    "Russian Federation": 143,
    "Rwanda": 144,
    "Saint Kitts and Nevis": 145,
    "Saint Lucia": 146,
    "Saint Vincent and the Grenadines": 147,
    "Samoa": 148,
    "San Marino": 149,
    "Sao Tome and Principe": 150,
    "Saudi Arabia": 151,
    "Senegal": 152,
    "Serbia": 153,
    "Seychelles": 154,
    "Sierra Leone": 155,
    "Singapore": 156,
    "Slovakia": 157,
    "Slovenia": 158,
    "Solomon Islands": 159,
    "Somalia": 160,
    "South Africa": 161,
    "South Sudan": 162,
    "Spain": 163,
    "Sri Lanka": 164,
    "Sudan": 165,
    "Suriname": 166,
    "Sweden": 167,
    "Switzerland": 168,
    "Syrian Arab Republic": 169,
    "Tajikistan": 170,
    "Thailand": 171,
    "Timor-Leste": 172,
    "Togo": 173,
    "Tonga": 174,
    "Trinidad and Tobago": 175,
    "Tunisia": 176,
    "Turkey": 177,
    "Turkmenistan": 178,
    "Tuvalu": 179,
    "Uganda": 180,
    "Ukraine": 181,
    "United Arab Emirates": 182,
    "United Kingdom": 183,
    "United Republic of Tanzania": 184,
    "United States of America": 185,
    "Uruguay": 186,
    "Uzbekistan": 187,
    "Vanuatu": 188,
    "Venezuela (Bolivarian Republic of)": 189,
    "Viet Nam": 190,
    "Yemen": 191,
    "Zambia": 192,
    "Zimbabwe": 193,
}

measure_options = {
    "Death": 1,
    "DALYs": 0,
    "YLLs": 2,
}

gender_options = {
    "Female": 0,
    "Male": 1,
}

cause_options = {
    "Cardiovascular Diseases": 0,
    "Chronic Respiratory Diseases": 1,
    "Respiratory Infections and Tuberculosis": 2,
}

col1, col2 = st.columns(2)

with col1:
    selected_option = st.selectbox("#### Location", list(loc_options.keys()))
    location = loc_options[selected_option]
    
    selected_option = st.selectbox("#### Gender", list(gender_options.keys()))
    gender = gender_options[selected_option]
    
    year = st.slider("#### Year", 2010, 2030, 2020)


with col2:
    selected_option = st.selectbox("#### Measure", list(measure_options.keys()))
    measure = measure_options[selected_option]

    selected_option = st.selectbox("#### Cause", list(cause_options.keys()))
    cause = cause_options[selected_option]

    pmconc = st.slider("#### Concentration of Particulate Matter", 0.0, 100.0, 50.0)


model = st.selectbox("#### Model",('Decision Tree Regressor', 'Gradient Boosting Regressor', 'Light GBM','Linear Regression', 'MLP Regressor', 'Random Forest Regressor',))

if st.button('Predict'):
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

    data = list(zip([location], [year], [measure], [gender], [cause], [pmconc]))
    array = np.array(data)

    pagol = LinearRegression()

    if (model=="Decision Tree Regressor"):
        pagol = DecisionTreeRegressor()
    elif (model=="Gradient Boosting Regressor"):
        pagol = GradientBoostingRegressor()  
    elif (model=="Light GBM"):
        pagol = lgb.LGBMRegressor()   
    elif (model=="Linear Regression"):
        pagol = LinearRegression()    
    elif (model=="MLP Regressor"):
        pagol = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu')
    elif (model=="Random Forest Regressor"):
        pagol = RandomForestRegressor()

    pagol.fit(x_train, y_train)
    predictions = pagol.predict(x_test)
    inputt = pagol.predict(array)
    #prediction = pagol.predict_proba(array)
    mse = metrics.mean_squared_error(y_test, predictions)
    mae = metrics.mean_absolute_error(y_test, predictions)
    r2 = metrics.r2_score(y_test, predictions)
    
    st.write("#### Prediction (Rate Per 100000 Population):", inputt[0])
    st.write("#### Mean Squared Error (MSE):", mse)
    st.write("#### Mean Absolute Error (MAE):", mae)
    st.write("#### R2 Score:", r2)

