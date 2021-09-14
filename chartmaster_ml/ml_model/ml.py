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

# y = df['PivotLows']
y = df[['PivotLows', 'MaxDuration', 'SlPerc', 'SlCooldown', 'TpSingle']]
x = df[['Slope_Volume1','Slope_Volume2','Slope_Volume3','Volatility','VolumeIndex','Time','DayOfWeek','Month']]

# print(x, y)
train_X, val_X, train_y, val_y = train_test_split(x, y, random_state = 0)

# # Categorical Encoding
# label_X_train = train_X.copy()
# label_X_valid = val_X.copy()
# val_X = [[volume1, volume2, volume3, volume4, diff, *ordinal_encoder.transform([[days, 'March']])[0]]]

# object_cols = ['DayOfWeek', 'Month']
# ordinal_encoder = OrdinalEncoder()
# label_X_train[object_cols] = ordinal_encoder.fit_transform(train_X[object_cols])

model = RandomForestRegressor()
model.fit(train_X, train_y)


def setPredictvalue(volume1, volume2, volume3, volatility, volumeIndex, time, days, months):
    # label_X_valid[object_cols] = 
    # print(ordinal_encoder.transform([[days, months]]))
    val_X = [[volume1, volume2, volume3, volatility, volumeIndex, time, days, months]]
    y_pred = model.predict(val_X)
    return y_pred[0]


# {
# "volume1" : 45957.8191963809,
# "volume2" : 46120.7766334909,
# "volume3" : 46244.93240234,
# "volume4" : 46297.619363374
# }
# print(y_pred)


# def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
#     model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
#     model.fit(train_X, train_y)
#     preds_val = model.predict(val_X)
#     mae = mean_squared_error(val_y, preds_val)
#     return(mae)

# for max_leaf_nodes in [5, 50, 400, 450, 500, 600, 700, 5000]:
#     my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
#     print("Max leaf nodes: %d  \t\t Mean Squared Error:  %d" %(max_leaf_nodes, my_mae))


