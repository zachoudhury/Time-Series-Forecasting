
#Time series forecasing for mean discharge rate (cft/s) of Comal Springs' on 
#daily data from 2018 to 2022 using TensorFlow framework and LSTM

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' #This line of code was required to prevent the program from aborting and restarting kernel (souce stack overflow)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf


#Loading csv file
"""
file_path=tf.keras.utils.get_file(origin='https://drive.google.com/uc?export=download&id=1pjmXkrRyiqaHDsjtd_g1YWpHB056kvJP')
csv_path, _ =os.path.splitext(file_path)
df=pd.read_csv(csv_path)
"""
os.chdir("E:/Machine Learning/AssignTS_forecast") #Change working directory
df=pd.read_csv("Comal_Springs.csv")

#Making date column the index column and changing format to datetime
df.index=pd.to_datetime(df['Date'],format='%m/%d/%Y')  

#Plotting time series
fig, ax = plt.subplots(figsize=(8, 6))
year_locator = mdates.YearLocator()
ax.xaxis.set_major_locator(year_locator) # Locator for major axis only.
ax.plot(df['Date'],df['Discharge_cfps'],'k')
plt.title("Mean discharge rate of Comal Springs")
plt.xlabel("Year")
plt.ylabel("Discharge Rate (cft/s)")
plt.show()

#Defining function to convert given dataframe into numpy array with "window_size" inputs and 1 label

def df_to_X_y(df,window_size):
	df_as_np=df.to_numpy()
	X=[]
	y=[]
	for i in range(len(df_as_np)-window_size):
		row=[[a] for a in df_as_np[i:i+6]]
		X.append(row) 
		label=df_as_np[i+6]
		y.append(label)
	return np.array(X), np.array(y)

WINDOW_SIZE=6
X,y=df_to_X_y(df['Discharge_cfps'],WINDOW_SIZE)
print("Shape of input", X.shape)
print("Shape of output", y.shape)


#Splitting dataset into train-validation-test sets at a ratio of 8:1:1
X_train, y_train = X[:1456], y[:1456]
X_val, y_val = X[1456:1638], y[1456:1638]
X_test, y_test = X[1638:], y[1638:]

print("Shape of training data: ", X_train.shape,y_train.shape)
print("Shape of validation data: ", X_val.shape,y_val.shape)
print("Shape of test data: ", X_test.shape,y_test.shape)

##Building up the model

#Importing libraries
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import InputLayer, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

#Defining layers
model_CS=Sequential()
model_CS.add(InputLayer((6,1)))
model_CS.add(LSTM(64,activation='relu'))
model_CS.add(Dense(8,'relu'))
model_CS.add(Dense(1))

print("Model summary:", model_CS.summary())

#Defining checkpoint to save the state of the system
cp=ModelCheckpoint('model_CS/', save_best_only=True) 

#Configuring the model for training
model_CS.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])

#Train and fit the model
model_CS.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, callbacks=[cp])


#Loading model
model_CS=load_model('model_CS/')

#Predicting test series (prediction vs actual)
test_predictions=model_CS.predict(X_test).flatten()
test_results=pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test})
print("Test Results", test_results)

#Plot the predicted values and actual values for the test data
plt.plot(test_results['Test Predictions'],'r')
plt.plot(test_results['Actuals'],'k')
plt.legend(['Predicted','Actual'],loc="lower right")
plt.title("Mean discharge rate of Comal Springs (predicted and actual)")
plt.xlabel("nth day from Test Set")
plt.ylabel("Discharge Rate (cft/s)")
plt.grid()
plt.show()