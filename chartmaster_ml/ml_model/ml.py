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

df = pd.read_csv('/Users/simonjeong/Desktop/TradingData.csv')
# df.head()

# y = df['PivotLows']
y = df[['PivotLows', 'MaxDuration', 'SlPerc', 'SlCooldown', 'TpSingle']]
x = df[['Slope_EMA1','Slope_EMA2','Slope_EMA3','Slope_EMA4','Distance_Btwn_Emas','DayOfWeek', 'Month']]

# print(x, y)
train_X, val_X, train_y, val_y = train_test_split(x, y, random_state = 0)
label_X_train = train_X.copy()
label_X_valid = val_X.copy()

object_cols = ['DayOfWeek', 'Month']
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(train_X[object_cols])

model = RandomForestRegressor()
model.fit(label_X_train, train_y)

def setPredictvalue(ema1, ema2, ema3, ema4, diff, days, months):
    # label_X_valid[object_cols] = 
    # print(ordinal_encoder.transform([[days, months]]))
    val_X = [[ema1, ema2, ema3, ema4, diff, *ordinal_encoder.transform([[days, months]])[0]]]
    print(val_X)
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


