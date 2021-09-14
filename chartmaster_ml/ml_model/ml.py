import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
# from xgboost import XGBRegressor
from sklearn.multioutput import RegressorChain
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv('/Users/simonjeong/Desktop/2020-05-01_00_00_00_2021-06-30_00_00_00(1MIN, BINANCEFTS_PERP_BTC_USDT)1.csv')
# df1 = pd.read_csv('/Users/simonjeong/Desktop/TradingData2.csv')

df= pd.concat([df]).drop_duplicates(keep='last')

y = df[['PivotLows', 'MaxDuration', 'SlPerc', 'SlCooldown', 'TpSingle']]
x = df[['Slope_Volume1','Slope_Volume2','Slope_Volume3','Volatility','VolumeIndex','Time','DayOfWeek','Month']]

train_X, val_X, train_y, val_y = train_test_split(x, y, random_state = 0)

model = RandomForestRegressor()
model.fit(train_X, train_y)

def setPredictvalue(volume1, volume2, volume3, volatility, volumeIndex, time, days, months):
    val_X = [[volume1, volume2, volume3, volatility, volumeIndex, time, days, months]]
    y_pred = model.predict(val_X)
    return y_pred[0]