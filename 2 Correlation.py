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
import matplotlib.pyplot as plt
import plotly.express as px

st.title("Dataset Scatterplot")

dataframe = pd.read_csv('./dataset.csv')

fig1 = px.scatter(
    dataframe.query("Measure=='Deaths'"),
    x="PM Concentration",
    y="Rate Per 100000 Population",
    color="Gender"
)

fig2 = px.scatter(
    dataframe.query("Measure=='DALYs'"),
    x="PM Concentration",
    y="Rate Per 100000 Population",
    color="Gender"
)

fig3 = px.scatter(
    dataframe.query("Measure=='YLLs'"),
    x="PM Concentration",
    y="Rate Per 100000 Population",
    color="Gender"
)

tab1, tab2, tab3 = st.tabs(["Deaths", "DALYs", "YLLs"])
with tab1:
    st.plotly_chart(fig1, theme="streamlit", use_container_width=True)
with tab2:
    st.plotly_chart(fig2, theme="streamlit", use_container_width=True)
with tab3:
    st.plotly_chart(fig3, theme="streamlit", use_container_width=True)


fig1 = px.scatter(
    dataframe.query("Measure=='Deaths'"),
    x="PM Concentration",
    y="Rate Per 100000 Population",
    color="Cause"
)

fig2 = px.scatter(
    dataframe.query("Measure=='DALYs'"),
    x="PM Concentration",
    y="Rate Per 100000 Population",
    color="Cause"
)

fig3 = px.scatter(
    dataframe.query("Measure=='YLLs'"),
    x="PM Concentration",
    y="Rate Per 100000 Population",
    color="Cause"
)

tab1, tab2, tab3 = st.tabs(["Deaths", "DALYs", "YLLs"])
with tab1:
    st.plotly_chart(fig1, theme="streamlit", use_container_width=True)
with tab2:
    st.plotly_chart(fig2, theme="streamlit", use_container_width=True)
with tab3:
    st.plotly_chart(fig3, theme="streamlit", use_container_width=True)


fig1 = px.line(dataframe, y="Rate Per 100000 Population", x="PM Concentration", color="Location", line_group="Location",
              line_shape="spline", render_mode="svg",
             color_discrete_sequence=px.colors.qualitative.G10,)

st.plotly_chart(fig1, theme="streamlit", use_container_width=True)