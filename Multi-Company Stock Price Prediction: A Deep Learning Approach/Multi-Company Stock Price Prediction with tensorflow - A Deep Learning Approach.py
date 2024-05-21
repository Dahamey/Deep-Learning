#!/usr/bin/env python
# coding: utf-8

# <h1 style="text-align:center;">Multi-Company Stock Price Prediction: A Deep Learning Approach</h1>

# * <u>**Author**</u> **:**[Younes Dahami](https://www.linkedin.com/in/dahami/)

# ![](big_tech.jpeg)

# In this notebook, we'll construct a "Stock Price Prediction" project employing `TensorFlow`. Analyzing stock market prices involves a **timeseries** approach, well-suited for implementation with a Recurrent Neural Network (RNN).
# 
# To execute this, we'll utilize TensorFlow, an open-source Python framework renowned for its powerful capabilities in Deep Learning and Machine Learning tasks.

# # 1) Importing the libraries and dataset

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler 
import tensorflow as tf
from tensorflow import keras

import os
import opendatasets as od
from datetime import datetime

import warnings 
warnings.filterwarnings("ignore") 


# Let's download the dataset as pandas dataframe :

# In[2]:


# download the dataset from kaggle
url = "https://www.kaggle.com/datasets/rohitjain454/all-stocks-5yr"
od.download(url, data_dir="./all-stocks-5yr")


# In[3]:


df = pd.read_csv("./all-stocks-5yr/all_stocks_5yr.csv")
df.head()


# In[4]:


# shape 
print(df.shape)

# Sample of 6 rows
df.sample(6)


# In[5]:


df.info()


# Since the given data consists of a `date` feature, that is an "object" data type, let's convert the column `date` into DateTime data type :

# In[6]:


df["date"] = pd.to_datetime(df["date"])
df.info()


# # 2) Exploratory Data Analysis
# 
# Let's delve into data analysis by visualizing and manipulating data. For this endeavor, we'll focus on renowned companies such as **Nvidia, Google, Apple, Facebook,** and more.
# 
# Our initial step involves visualizing the distribution of open and closed stock prices over a span of 5 years for selected companies.

# In[7]:


df.columns


# In[8]:


df_grouped = df.groupby("Name")[["open", "close"]].mean()
df_grouped


# In[9]:


df_grouped.index.isin(["NVDA"]).sum()


# In[10]:


# Nvidia
df_grouped.loc["NVDA"]


# In[11]:


# date vs open & date vs close
plt.figure(figsize=(20,10))

companies = ["AAPL", "AMD", "FB", "GOOGL", "AMZN", "NVDA", "EBAY", "CSCO", "IBM"]

for index, company in enumerate(companies, start=1) :
    # 9 plots : 3 by 3
    plt.subplot(3,3, index)
    
    # dataframe of company
    c = df[df["Name"] == company]
    
    # plotting "close" of the company
    plt.plot(c["date"], c["close"], c = "r", label = "close", marker = "+")
    
    # plotting "open" of the company
    plt.plot(c["date"], c["open"], c ="g", label = "open", marker = "*")
    plt.title(company)
    plt.legend()
    plt.tight_layout()


# Now let’s plot the volume of trade for these 9 stocks as well as a function of time.

# In[12]:


plt.figure(figsize=(20,10))

for index, company in enumerate(companies, 1) :
    plt.subplot(3,3,index)
    c = df[df["Name"] == company]
    
    plt.plot(c["date"], c["volume"], c = "blue", marker = "o")
    plt.title(f"{company} Volume")
    plt.tight_layout()


# Now let’s analyze the data for Apple Stocks from 2013 to 2018 :

# In[13]:


apple_df = df[df["Name"] == "AAPL"]

apple_2013_2018 = apple_df.loc[(apple_df['date'] > datetime(2013,1,1)) & (apple_df["date"]<datetime(2018,1,1))]

# Plotting Apple close between 2013 and 2018
plt.plot(apple_2013_2018["date"], apple_2013_2018["close"])
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Apple Stock Prices")

plt.show()


# # 3) Building the model
# 
# ## 3-1) Splitting into trainining and validation datasets
# 
# Now let’s select a subset of the whole data as the training data, the remaining data left will be for the validation part.

# In[24]:


apple_close = apple_df.filter(["close"])
dataset_apple = apple_close.values

# 85% training  
training_apple = int(np.ceil(len(dataset_apple) * .85))
print(f"The number of obesrvation for the training is {training_apple}, meanwhile the remaining\
    {len(dataset_apple)-training_apple} will be for validation")


# ## 3-2) Scalling

# Now that we have the training data length, the next thing to do is applying scaling and preparing features and labels (`X_train` and `y_train`) :

# In[15]:


# Let's scale the data : making the values between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset_apple)

# Training data (85%)~1071 obs
train_data = scaled_data[0:int(training_apple), :]

# Preparing the features X and label y
X_train = []
y_train = []

for i in range(60, len(train_data)) :
    # X_train contains closes fom i-60 to i-1 (i.e. the last 60 closes)
    X_train.append(train_data[i-60:i, 0])
    # y_train contains the close of the i-th close
    y_train.append(train_data[i, 0])

# convert from list to numpy array
X_train, y_train = np.array(X_train), np.array(y_train)    
print(X_train.shape)
print(y_train.shape)


# Reshaping from (1011, 60) ----> (1011, 60, 1)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print(X_train.shape)


# # 3) Build Gated RNN-LSTM network using TensorFlow
# 
# Using TensorFlow, we can create LSTM-gated RNN cells. LSTM is used in Recurrent Neural Networks for sequence models and time series data. LSTM is used to avoid the vanishing gradient issue which widely occurres in training RNN.
# 
# **To stack multiple LSTM in TensorFlow it is mandatory to use `return_sequences = True`.** Since our data is time series varying we apply no activation to the output layer and it remains as 1 node. 

# In[16]:


model = keras.models.Sequential()

# Input layer : (60,1) = (T,C)
model.add(keras.layers.Input(shape = (X_train.shape[1], 1)))

# LSTM 1 layer  : (T,C)=(60,1)  @ (None, 1,64)--->(B,60,1)  @ (B, 1,64)---> (None,60,64)=(B,T,C)
model.add(keras.layers.LSTM(units = 64, 
                            return_sequences = True))

# LSTM 2 layer : 
model.add(keras.layers.LSTM(units=64))

# Dense layer :
model.add(keras.layers.Dense(32))

# Regularization : deactivating 50% of nodes
model.add(keras.layers.Dropout(0.5))

# ouput layer with 1 node
model.add(keras.layers.Dense(1))


# In[17]:


model.summary()


# # 4) Model Compilation and Training
# 
# While compiling a model we provide these three essential parameters :
# 
# * **optimizer :** This is the method that helps to optimize the cost function by using gradient descent.
# 
# * **loss :** The loss function by which we monitor whether the model is improving with training or not.
# 
# * **metrics :** This helps to evaluate the model by predicting the training and the validation data.

# In[18]:


# Predicting the stock price (continuous variable) using data from the last 60 closes to predict 61st
model.compile(optimizer= "adam",
             loss = "mean_squared_error")


# In[19]:


# Fitting the model
history = model.fit(X_train, y_train, epochs = 12)


# # 5) Predictions
# 
# For predictions, we require testing data, so we first create the testing data and then proceed with the model prediction. 

# In[20]:


test_data = scaled_data[training_apple - 60:, :] 
X_test = [] 
y_test = dataset_apple[training_apple:, :] 


for i in range(60, len(test_data)) : 
    X_test.append(test_data[i-60:i, 0]) 

X_test = np.array(X_test) 
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) 
  
# predict the testing data 
predictions = model.predict(X_test) 

# retrieve the values
predictions = scaler.inverse_transform(predictions) 


# In[21]:


predictions.shape


# In[22]:


# evaluation metrics 
mse = np.mean(((predictions - y_test) ** 2)) 
print("MSE", mse) 
print("RMSE", np.sqrt(mse)) 


# Now that we have predicted the testing data, let us visualize the final results. 

# In[23]:


train = apple_df[:training_apple]
test = apple_df[training_apple:]
test["Predictions"] = predictions

plt.figure(figsize=(20,10))
plt.plot(train["date"], train["close"])
plt.plot(test["date"], test[["close", "Predictions"]])

plt.title("Apple Stock Close Price")
plt.xlabel("Date")
plt.ylabel("Close")
plt.legend(["Train", "Test", "Predictions"])


# # Change Log
# 
# | Date (YYYY-MM-DD) | Version | Changed By      | Change Description      |
# | ----------------- | ------- | -------------   | ----------------------- |
# | 2023-11-25       | 1.0     | Younes Dahami   |  initial version |
# |
# 

# In[ ]:




