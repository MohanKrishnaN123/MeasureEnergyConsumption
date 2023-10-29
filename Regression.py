import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.metrics import mean_squared_error

def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

def process(fileHead, fileName, date):
    df = pd.read_csv(fileName)
    df = df.set_index('Datetime')
    df.index = pd.to_datetime(df.index)
    train = df.loc[df.index < date]
    test = df.loc[df.index >= date]
    df = create_features(df)
    train = create_features(train)
    test = create_features(test)
    features = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
    target = fileHead
    x_train = train[features]
    y_train = train[target]
    x_test = test[features]
    y_test = test[target]
    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree', n_estimators=1000, early_stopping_rounds=50, objective='reg:linear', max_depth=3, learning_rate=0.01)
    reg.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], verbose=100)
    test['prediction'] = reg.predict(x_test)
    score = np.sqrt(mean_squared_error(test[fileHead], test['prediction']))
    print(f'RMSE Score on Test set: {score:0.2f}')
    test['error'] = np.abs(test[target] - test['prediction'])
    test['date'] = test.index.date
    test.groupby(['date'])['error'].mean().sort_values(ascending=False).head(10)

if __name__ == '__main__':
    file_header = ['AEP_MW','COMED_MW','DAYTON_MW','DEOK_MW','DOM_MW','DUQ_MW','EKPC_MW','FE_MW','NI_MW']
    data = [['../IBM Project/Dataset/AEP_hourly.csv','02-01-2017'],
            ['../IBM Project/Dataset/COMED_hourly.csv','08-01-2017'],
            ['../IBM Project/Dataset/DAYTON_hourly.csv','10-01-16'],
            ['../IBM Project/Dataset/DEOK_hourly.csv','06-01-17'],
            ['../IBM Project/Dataset/DOM_hourly.csv','01-01-17'],
            ['../IBM Project/Dataset/DUQ_hourly.csv','06-01-16'],
            ['../IBM Project/Dataset/EKPC_hourly.csv','08-01-17'],
            ['../IBM Project/Dataset/FE_hourly.csv','02-01-17'],
            ['../IBM Project/Dataset/NI_hourly.csv','07-01-09']]

    print("1.AEP\n2.COMED\n3.DAYTON\n4.DEOK\n5.DOM\n6.DUQ\n7.EKPC\n8.FE\n9.NI\n")
    n = int(input())
    if(n in range(1,10)):
        process(file_header[n-1],data[n-1][0],data[n-1][1])
    else:
        print("Invalid Input!")