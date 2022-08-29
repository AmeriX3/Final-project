import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
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
from bs4 import BeautifulSoup

def Brands():
    cars= pd.read_csv("cars.csv")
    lis= cars["car_brand"].unique()
    lis.sort()
    lis = tuple(lis)
    return lis

def Models(s):
    if s:
        cars= pd.read_csv("cars.csv")
        lis= cars[cars["car_brand"]==s]["car_model"].unique()
        lis.sort()
        lis = tuple(lis)
    else:
        cars= pd.read_csv("cars.csv")
        lis= cars["car_model"].unique()
        lis.sort()
        lis = tuple(lis)
        return lis

def Years(s):
    if s:
        cars= pd.read_csv("cars.csv")
        lis= cars[cars["car_model"]==s]["car_year"].unique()
        lis.sort()
        lis = tuple(lis)
    else:
        cars= pd.read_csv("cars.csv")
        lis= cars["car_year"].unique()
        lis.sort()
        lis = tuple(lis)
        return lis

def doors(s):
    if s:
        cars= pd.read_csv("cars.csv")
        lis= cars[cars["car_model"]==s]["car_door"].unique()
        lis.sort()
        lis = tuple(lis)
        return lis

