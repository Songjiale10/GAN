import mean as mean
import pandas as pd
import datetime

import torch.utils.data

from Dataset import PRSA_dataset
data=pd.read_csv('PRSA_data_2010.1.1-2014.12.31.csv')
print(data.info())

data['time']=data.apply(lambda x:datetime.datetime(year=x['year'],
                                                   month=x['month'],
                                                   day=x['day'],
                                                   hour=x['hour']),
                         axis=1)
data.set_index(keys='time',inplace=True)
data.drop(columns=['No','year','month','day','hour'],inplace=True)
#print(data.head())
data.fillna(method='bfill',inplace=True)
print(data.cbwd.unique())
data=data.join(pd.get_dummies(data.cbwd))

del data['cbwd']
#print(data)
train_data=data.iloc[:35000]
test_data=data.iloc[35000:]

mean=train_data.mean(axis=0)
std=train_data.std(axis=0)
train_data=(train_data-mean)/std
test_data=(test_data-mean)/std
#print(train_data.head())

train_ds=PRSA_dataset(train_data)
test_ds=PRSA_dataset(test_data)
print(len(train_ds),len(test_ds))

BATCH_SIZE=256
hidden_size=64
train_dl=torch.utils.data.DataLoader(
                                    train_ds,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True
)
test_dl=torch.utils.data.DataLoader(
    test_ds,
    batch_size=BATCH_SIZE
)
