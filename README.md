# Bus-Utilization
Forecasting future bus utilization in municipalities

forecast.py is the main code where one baseline and one complex algorithm Autoregression (AR) and Recurrent Neural Network (RNN) have been implemented. The root mean square error results (RMSE's) for each municiality n is written in file with the name output n.
For both AR and RNN, the average RMSE over the municipalities have been calculated seperately and written in a RMSE results.txt file.  
For interpolation the data from the previous week of the same day is utilized to monitor daily characteristic of usage. 

