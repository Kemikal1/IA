import math
import numpy as np
import copy
from sklearn.linear_model import LinearRegression,Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.utils import shuffle
from sklearn import preprocessing


#load data
training_data = np.load('data_lab6/training_data.npy')

prices = np.load('data_lab6/prices.npy')
# print the first 4 samples
print('The first 4 samples are:\n ', training_data[:4])
print('The first 4 prices are:\n ', prices[:100])
# shuffle
training_data, prices = shuffle(training_data, prices, random_state=0)

def normalize(data1,data2):
    scaler=preprocessing.StandardScaler()
    scaler.fit(data1)
    scaled_data1=scaler.transform(data1)
    scaled_data2=scaler.transform(data2)

    return  scaled_data1,scaled_data2

num_samples_fold=len(training_data)//3
tr_data1,prices1=training_data[:num_samples_fold],prices[:num_samples_fold]
tr_data2,prices2=training_data[num_samples_fold:2*num_samples_fold],prices[num_samples_fold:num_samples_fold *2]
tr_data3,prices3=training_data[num_samples_fold*2:num_samples_fold*3],prices[num_samples_fold*2:num_samples_fold*3]


def step(tr_data,tr_lbs,ts_data,ts_lbs):
    tr_data,ts_data=normalize(tr_data,ts_data)
    model=LinearRegression()
    model.fit(tr_data,tr_lbs)
    mae=mean_absolute_error(ts_lbs,model.predict(ts_data))
    mse=mean_squared_error(ts_lbs,model.predict(ts_data))
    return mae,mse

def step_ridge(tr_data,tr_lbs,ts_data,ts_lbs,alpha):
    tr_data, ts_data = normalize(tr_data, ts_data)
    model = Ridge(alpha)
    model.fit(tr_data, tr_lbs)
    mae = mean_absolute_error(ts_lbs, model.predict(ts_data))
    mse = mean_squared_error(ts_lbs, model.predict(ts_data))
    return mae, mse
###
#3 run-uri pentru regresie liniara
###


mae1,mse1=step(np.concatenate((tr_data1,tr_data3)),np.concatenate((prices1,prices3)),tr_data2,prices2)
mae2,mse2=step(np.concatenate((tr_data1,tr_data2)),np.concatenate((prices1,prices2)),tr_data3,prices3)
mae3,mse3=step(np.concatenate((tr_data2,tr_data3)),np.concatenate((prices2,prices3)),tr_data1,prices1)
mae=(mae1+mae2+mae3)/3
mse=(mse1+mse2+mse3)/3
print(tr_data1)
print("Mae si mse pentru regresia liniara",mae,mse)

alph=[1,10,100,1000]
mae_fin=mae*10
mse_fin=mse*10
alpha_best=0
for alpha in alph:
    mae1, mse1 = step_ridge(np.concatenate((tr_data1, tr_data3)), np.concatenate((prices1, prices3)), tr_data2, prices2,alpha)
    mae2, mse2 = step_ridge(np.concatenate((tr_data1, tr_data2)), np.concatenate((prices1, prices2)), tr_data3, prices3,alpha)
    mae3, mse3 = step_ridge(np.concatenate((tr_data2, tr_data3)), np.concatenate((prices2, prices3)), tr_data1, prices1,alpha)
    mae = (mae1 + mae2 + mae3) / 3
    mse = (mse1 + mse2 + mse3) / 3
    print (mae,mse)

    if (mae+mse)/2<(mae_fin+mse_fin)/2:
        mae_fin=mae
        mse_fin=mse
        alpha_best=alpha


print("Mae si mse pentru ridge cu cel mai bun alpha",mae_fin,mse_fin,"alpha=",alpha_best)


model=Ridge(alpha_best)
model.fit(training_data,prices)
print("coeficientii",model.coef_)
coef=model.coef_

print("cel mai semnificativ atribut",np.argmax(np.abs(coef))+1)
coef[np.argmax(np.abs(coef))]=0
print("al doilea cel mai semnificativ atribut",np.argmax(np.abs(coef))+1)
print("cel mai putin semnificativ atribut",np.argmin(np.abs(coef))+1)

