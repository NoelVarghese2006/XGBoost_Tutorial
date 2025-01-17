import seaborn as sns
import pandas as pd
import matplotlib as plt
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb


warnings.filterwarnings("ignore")

diamonds = sns.load_dataset("diamonds")

#print(diamonds.head())

#print(diamonds.shape)

#print(diamonds.describe(exclude=np.number))

X, y = diamonds.drop('cut', axis=1), diamonds[['cut']]

y_encoded = OrdinalEncoder().fit_transform(y)

#Special things with XG that doesn't require one-hot encoding

cats = X.select_dtypes(exclude=np.number).columns.tolist()

for col in cats:
    X[col] = X[col].astype('category')

#print(X.dtypes)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, random_state=1, stratify=y_encoded)

dtrain_clf = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_clf = xgb.DMatrix(X_test, y_test, enable_categorical=True)

# This defines the hyperparameters, objective is what the model should aim for and 

params = {"objective": "multi:softprob", "tree_method": "hist", "num_class": 5} #if you ever get a NVIDIA GPU, use gpu_hist instead

# n is epcohs probably
n = 1000
results = xgb.cv (
    params,
    dtrain_clf,
    num_boost_round=n,
    nfold=5,
    early_stopping_rounds=20,
    metrics=['mlogloss', 'auc', 'merror']
)
#print(results.head())
print(results['test-auc-mean'].max())