import pandas as pd
import math as mt
import numpy as np
import tensorflow as tf
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

EPOCH = 10
BACK = 1
MUN = 10
DENSE = 1


def make_dataset(this_data, back):
    
    dataX, dataY = [], []
    #shift data by back and insert to Y
    for i in range(len(this_data)-back-1):
        a = this_data[i:(i+back), 0]
        dataX.append(a)
        dataY.append(this_data[i + back, 0])
        
    return np.array(dataX), np.array(dataY)

def interpolate(my_data):
    
    result = pd.DataFrame({'timestamp': [],  'municipality_id': [],  'usage': []})    
    
    for j in np.arange(0,len(my_data),180):
        
        cur_date = my_data.iloc[j]['timestamp'][6:10]
        #initialize app_data 
        app_data = {'timestamp': [],  'municipality_id': [],  'usage': []}
        app_rows = pd.DataFrame(app_data)
        
        if((cur_date != "6-22") and (cur_date != "8-05")):            
            app_rows = my_data.iloc[j:j+180] #use the data as it is             
        else: 
            #set previous week data for interpolation
            if(j >=7*180):  
                #get the previos week data
                copier_time = my_data.iloc[j-7*180:j-5*180]['timestamp'].tolist()
                copier_mun = my_data.iloc[j-7*180:j-5*180]['municipality_id'].tolist()
                copier_use  = my_data.iloc[j-7*180:j-5*180]['usage'].tolist()
                
            for i in range(0,2*180):
                if(cur_date == "6-22"):                
                    if(i<180):              
                        mytime = "2017-06-20"+copier_time[i][-9:]                        
                    else:    
                        mytime = "2017-06-21"+copier_time[i][-9:]
                else:                
                    if(i<180):              
                        mytime = "2017-08-03"+copier_time[i][-9:]
                    else:     
                        mytime = "2017-08-04"+copier_time[i][-9:]
                        
                mun = copier_mun[i]
                use  = copier_use[i] 
                 #data of previous week days to be appended   
                cur_row = {'timestamp':mytime,  'municipality_id': mun, 'usage':use}
                app_rows = app_rows.append(cur_row,ignore_index=True)
            
            #data of today is to be appended
            app_rows = app_rows.append(my_data.iloc[j:j+180],ignore_index=True)
         
        #append to the result 
        result = result.append(app_rows,ignore_index=True)  
        app_data = {'timestamp': [],  'municipality_id': [],  'usage': []}
        app_rows = pd.DataFrame(app_data)
        #clear app_rows Dataframe for the next iteration
        
    result = result.reset_index()
    
    return result
  
  
  #read, trim and interpolate data
my_input = pd.read_csv('municipality_bus_utilization.csv')
trimmed_input = my_input.drop('total_capacity',axis=1)
trimmed_input = interpolate(trimmed_input)

#for debug
#for index, row in trimmed_input.iterrows():
#    print(row['timestamp'], row['municipality_id'], row['usage'])

#sort on first municipality_id then timestamp
sorted_input = trimmed_input.sort_values(["municipality_id", "timestamp"])

rmse = [[0]*MUN,[0]*MUN]

for i in range(0,MUN): #for each municipality
    
    cur_mun = sorted_input.loc[sorted_input['municipality_id'] == i]
    cur_mun = cur_mun.reset_index()
    
    data = {'timestamp': [],  'municipality_id': [],  'usage': []}
    del_mun = pd.DataFrame(data)
    
    for j in range(0,len(cur_mun)):
        if(j%2 == 1):            #take even rows
            #take the max usage of two consecutive half hours
            cur_max = max(cur_mun.iloc[j-1]['usage'],cur_mun.iloc[j]['usage'])
            cur_row = \
            {'timestamp': cur_mun.iloc[j]['timestamp'],  'municipality_id': cur_mun.iloc[j]['municipality_id'],  'usage':cur_max}
            del_mun = del_mun.append(cur_row,ignore_index=True)
            
    #partition over time to train and test
    train = del_mun.loc[del_mun['timestamp'] < "2017-08-05"].reset_index()    
    test = del_mun.loc[del_mun['timestamp'] >= "2017-08-05"].reset_index()
    
        
    ## ##################################
    # train autoregression and predict last two week usages
    #take only usage data for train 
    train_raw = train.drop(['timestamp','municipality_id'],axis=1)
    train_AR_data = train_raw['usage'].to_list()    
    
    #take only usage data for test
    test_raw = test.drop(['timestamp','municipality_id'],axis=1)
    test_AR_data = test_raw['usage'].to_list()
    
    #AR MODEL    
    AR_model = AutoReg(train_AR_data, lags=50,old_names=False)
    AR_model_fit = AR_model.fit()
    AR_predictions = AR_model_fit.predict(start=len(train_AR_data), \
                                    end=len(train_AR_data)+len(test_AR_data)-1, dynamic=False)
    
    #the root mean square error for the AR model
    rmse[0][i] = mt.sqrt(mean_squared_error(test_AR_data, AR_predictions))
    
    ## ##############################################
    #train RNN and predict last two week usages
    # normalize the dataset    
    trainset =train.drop(['timestamp','municipality_id','index'],axis=1).values
    trainset = trainset.astype('float32')
    
    testset = test.drop(['timestamp','municipality_id','index'],axis=1).values
    testset = testset.astype('float32')
    compare_test = testset
        
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_RNN_data = scaler.fit_transform(trainset)    
    test_RNN_data = scaler.fit_transform(testset)    
  
    trainX, trainY = make_dataset(train_RNN_data, BACK)
    testX, testY   = make_dataset(test_RNN_data, BACK)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX  = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    
    #RNN MODEL
    print('For municipality %d the RNN is running' % i)
    #1 input, a hidden layer with 64 neurons, and an output layer that makes a single value prediction. 
    #The default sigmoid activation function is used. The network is trained for EPOCHS
    RNN_model = tf.keras.models.Sequential([tf.keras.layers.LSTM(64, input_shape=(1, BACK)),
    tf.keras.layers.Dense(units=DENSE)])
    RNN_model.compile(loss='mean_squared_error')
    RNN_model.fit(trainX, trainY, epochs=EPOCH, batch_size=1)
    
    RNN_predictions = RNN_model.predict(testX)
    RNN_predictions = scaler.inverse_transform(RNN_predictions)    
    
    min_len = min(len(compare_test),len(RNN_predictions[:,0]))
    
    #the root mean square error for the RNN model
    rmse[1][i] =mt.sqrt(mean_squared_error(compare_test[:min_len], RNN_predictions[:,0][:min_len]))
    
    ## #############################################3
    # plot results for each municipality
    pyplot.figure(i)
    pyplot.plot(test_AR_data)
    pyplot.plot(AR_predictions, color='red')
    pyplot.plot(RNN_predictions, color='green')
    pyplot.title('Municipality %d with RMSE(AR)=%.1f and RMSE(RNN)=%.1f' % (i, rmse[0][i],rmse[1][i]))
    pyplot.legend(["test data", "predict (AR)", "predict(RNN)"], loc ="upper left")
    pyplot.ylabel('usage')
    pyplot.savefig("output %d.jpg" %i)
 
    # print average error results for AR/RNN methods respectively
print('The total RMSE for AR Method is %.1f' % np.mean(rmse[0]))
print('The total RMSE for RNN Method is %.1f' % np.mean(rmse[1]))

with open('RMSE results.txt', 'w') as f:
    f.write('The mean RMSE for AR Method is %.1f' % np.mean(rmse[0]))
    f.write('\n')
    f.write('The mean RMSE for RNN Method is %.1f' % np.mean(rmse[1]))
    f.write('\n')
