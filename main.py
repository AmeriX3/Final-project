
import streamlit as st
<<<<<<< HEAD
=======
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics
>>>>>>> a5a1d984cac9d199402f952f73b7b9314c10d516
import sc 

brand = st.selectbox(
     'Select car Brand',
     sc.Brands)

model = st.selectbox("Select Model",
sc.Models(brand)
)
year = st.selectbox("Select year",sc.Years(model))

doors = st.selectbox("Select amount of doors",sc.doors(model))

odo= st.slider("Select Mileage (km)",0,100000,step=1000)

st.metric(label="Price", value=sc.Price(model,brand,doors,year,odo))

