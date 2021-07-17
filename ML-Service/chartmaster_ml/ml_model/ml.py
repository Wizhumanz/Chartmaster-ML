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

df = pd.read_csv('/Users/simonjeong/Desktop/TradingData.csv')
# df.head()

# y = df['PivotLows']
y = df[['PivotLows', 'MaxDuration', 'SlPerc', 'SlCooldown', 'TpSingle']]
x = df[['EMA1','EMA2','EMA3','EMA4']]

# print(x, y)
train_X, val_X, train_y, val_y = train_test_split(x, y, random_state = 0)

model = RandomForestRegressor()
model.fit(train_X, train_y)

def setPredictvalue(ema1, ema2, ema3, ema4):
    val_X = [[ema1, ema2, ema3, ema4]]
    y_pred = model.predict(val_X)
    return y_pred[0]


# {
# "ema1" : 45957.8191963809,
# "ema2" : 46120.7766334909,
# "ema3" : 46244.93240234,
# "ema4" : 46297.619363374
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


