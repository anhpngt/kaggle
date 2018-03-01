from math import sqrt

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def doDataEngineering(train_df, test_df):
    n_samples = int(len(train_df['LotArea'])*0.75)
    trX = train_df[['LotArea', 'OverallQual', 'OverallCond']][:n_samples]
    trY = train_df['SalePrice'][:n_samples]
    teX = train_df[['LotArea', 'OverallQual', 'OverallCond']][n_samples:]
    teY = train_df['SalePrice'][n_samples:]
    return trX, trY, teX, teY 

if __name__=='__main__':
    # import data
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    print(train_df.info())
    
    trX, trY, teX, teY = doDataEngineering(train_df, test_df)
    print(np.shape(trX))
    print(np.shape(trY))
    
    clf = LinearRegression()
    clf.fit(trX, trY)
    
    prediction = clf.predict(teX)
    error = sqrt(mean_squared_error(teY, prediction))
    print("Error score:", error)
    
    
    