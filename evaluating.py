import seaborn as sns
import pandas as pd
import matplotlib as plt
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb

warnings.filterwarnings("ignore")

diamonds = sns.load_dataset("diamonds")

#print(diamonds.head())

#print(diamonds.shape)

#print(diamonds.describe(exclude=np.number))

X, y = diamonds.drop('price', axis=1), diamonds['price']

#Special things with XG that doesn't require one-hot encoding

cats = X.select_dtypes(exclude=np.number).columns.tolist()

for col in cats:
    X[col] = X[col].astype('category')

#print(X.dtypes)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)

# This defines the hyperparameters, objective is what the model should aim for and 

params = {"objective": "reg:squarederror", "tree_method": "hist"} #if you ever get a NVIDIA GPU, use gpu_hist instead

# n is epcohs probably, 
n = 10000

evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]

model = xgb.train(
    params=params,
    dtrain=dtrain_reg,
    num_boost_round=n,
    evals=evals,
    verbose_eval=10,
    early_stopping_rounds=50, #very important, optimizes epochs for you
)
