import pandas as pd
import numpy as np

import scipy.stats as st
from datetime import datetime
from geographiclib.geodesic import Geodesic

from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import optuna

df = pd.read_csv('/content/drive/MyDrive/dataset/nyc_taxi_trip/train.csv')
test = pd.read_csv('/content/drive/MyDrive/dataset/nyc_taxi_trip/test.csv')
sample = pd.read_csv('/content/drive/MyDrive/dataset/nyc_taxi_trip/sample_submission.csv')
df_osrm = pd.read_csv('/content/drive/MyDrive/dataset/nyc_taxi_trip/train_augmented.csv')
test_osrm = pd.read_csv('/content/drive/MyDrive/dataset/nyc_taxi_trip/test_augmented.csv')
print(df.head(3))

df = pd.merge(df, df_osrm[['id','distance']], on='id', how='left')
test = pd.merge(test, test_osrm[['id','distance']], on='id', how='left')

df = df.drop(['id','dropoff_datetime'], axis=1)
test = test.drop(['id'], axis=1)

df = df.drop_duplicates()

df = df.dropna()

df = df[df['trip_duration']<=18000]

map_yn = {'Y':1,
          'N':0}

df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map(map_yn)
test['store_and_fwd_flag'] = test['store_and_fwd_flag'].map(map_yn)

df.loc[df['vendor_id']==1, 'vendor_id'] = 0
df.loc[df['vendor_id']==2, 'vendor_id'] = 1
test.loc[test['vendor_id']==1, 'vendor_id'] = 0
test.loc[test['vendor_id']==2, 'vendor_id'] = 1
print(df.head(3))

df_passenger = pd.get_dummies(df['passenger_count'], prefix='passenger', drop_first=True)
test_passenger = pd.get_dummies(test['passenger_count'], prefix='passenger', drop_first=True)
df = pd.concat([df, df_passenger], axis=1)
test = pd.concat([test, test_passenger], axis=1)
test['passenger_7'] = 0
test['passenger_8'] = 0

df = df.drop('passenger_count', axis=1)
test = test.drop('passenger_count', axis=1)
print(df.head(3))

def date_prep(df):
  df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
  df['hour'] = df['pickup_datetime'].dt.hour + (df['pickup_datetime'].dt.minute/60)
  df['day'] = df['pickup_datetime'].dt.dayofweek
  df['week'] = df['pickup_datetime'].dt.week
  df['month'] = df['pickup_datetime'].dt.month
  return df
  
df = date_prep(df)
test = date_prep(test)
print(df.head(3))

df['haversine'] = 6371*2*np.arcsin(np.sqrt(np.sin((\
                      np.radians(df['dropoff_latitude'])-np.radians(df['pickup_latitude']))/2)**2\
                      + np.cos(np.radians(df['pickup_latitude']))\
                      *np.cos(np.radians(df['dropoff_latitude']))\
                      *np.sin((np.radians(df['dropoff_longitude'])-np.radians(df['pickup_longitude']))/2)**2))

test['haversine'] = 6371*2*np.arcsin(np.sqrt(np.sin((\
                      np.radians(test['dropoff_latitude'])-np.radians(test['pickup_latitude']))/2)**2\
                      + np.cos(np.radians(test['pickup_latitude']))\
                      *np.cos(np.radians(test['dropoff_latitude']))\
                      *np.sin((np.radians(test['dropoff_longitude'])-np.radians(test['pickup_longitude']))/2)**2))
print(df.head(3))

hd = pd.DataFrame(['2016-01-01', '2016-01-18', '2016-02-15', '2016-05-30',
        '2016-06-04', '2016-09-05', '2016-10-10'], columns=['holiday'])

def weekend_holiday(df):
  df['date'] = df['pickup_datetime'].dt.date.astype(str)
  df = df.merge(hd, left_on='date', right_on='holiday', how='left')

  df.loc[~df['holiday'].isnull(),'holiday'] = 1
  df.loc[df['holiday'].isnull(),'holiday'] = 0
  df['weekend'] = 0
  df.loc[df['pickup_datetime'].dt.dayofweek==5, 'weekend'] = 1
  df.loc[df['pickup_datetime'].dt.dayofweek==6, 'weekend'] = 1
  
  df = df.drop(['date'], axis=1)

  return df

df = weekend_holiday(df)
test = weekend_holiday(test)

df = df.drop('pickup_datetime', axis=1)
test = test.drop('pickup_datetime', axis=1)
print(df.head(3))

def bearing(df):
  df['bearing'] = df.apply(lambda x: Geodesic.WGS84.Inverse(x['pickup_latitude'],
                                                            x['pickup_longitude'],
                                                            x['dropoff_latitude'],
                                                            x['dropoff_longitude'])['azi1'],
                                               axis=1)
  return df

df = bearing(df)
test= bearing(test)
print(df.head(3))

X = df.drop('trip_duration', axis=1)
y = df['trip_duration']
test = test[X.columns.tolist()]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

best = {'gamma': 0.4037136463522653,
        'random_state': 333,
        'reg_lambda': 0.19936046586055467,
        'reg_alpha': 0.7752043175237793,
        'min_child_weight': 10,
        'max_depth': 9,
        'n_estimators': 90,
        'learning_rate': 0.18093250150898615,
        'colsample_bytree': 0.8327319308157345}
xgb = XGBRegressor(**best)
xgb.fit(X, y)
filename = 'my_model.sav'
pickle.dump(xgb, open(filename,'wb'))
print(df.head(3))