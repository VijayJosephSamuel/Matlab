
# coding: utf-8

# In[1]:


# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


dataset_train = pd.read_csv('newtrain.csv')


# In[3]:


dataset_train = dataset_train.drop(columns=['Unnamed: 0'])


# In[4]:


training_set = dataset_train.T


# In[5]:


trainp = training_set.values


# In[6]:


trainp


# In[8]:


# trainp = trainp[~pd.isnull(trainp)]


# In[19]:





# In[7]:


A = []
    


# In[8]:


for i in range(0,4459):
    A.append(trainp[i][~pd.isnull(trainp[i])])
    # print(i)


# In[9]:


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))


# In[10]:


A


# In[11]:


training = []
for i in range(0,4459):
    training.append(sc.fit_transform(A[i][1:len(A[i])].reshape(-1, 1)))
    # print(i)


# In[12]:


training[0]


# In[13]:


X_train = []
y_train = []


# In[14]:


for i in range(0, 4459):
    for j in range(60,len(training[i])):
         X_train.append(training[i][j-60:j, 0])
         y_train.append(training[i][j, 0])


# In[15]:


X_train, y_train = np.array(X_train), np.array(y_train)


# In[16]:


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# In[17]:


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[18]:


# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[19]:


# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


# In[1]:
regressor.save('my_model.h5')



