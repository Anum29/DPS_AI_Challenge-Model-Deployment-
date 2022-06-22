#!/usr/bin/env python
# coding: utf-8

# In[73]:


## Importing the required libraries
import pandas as pd
import matplotlib.pyplot as plt


# ## Data  Preprocessing and Visualization

# In[2]:


## Reading a csv file in pandas data frame

df = pd.read_csv("data.csv")


# In[3]:


## first 5 values
df.head()


# In[4]:


## reading data  till the year 2021 and accumulated accidents 
df1= df[(df.Year<=2021) & (df.Month=='Summe')]


# In[5]:


df1.head()


# In[6]:


## Accumulating accidents by year
df2 = df1.groupby('Year', )['Value'].sum().to_dict() 


# In[7]:


df2 = pd.DataFrame(list(df2.items()),columns = ['year','value']) 


# In[8]:


df2.head()


# In[9]:


# create data
x = df2["year"]
y = df2["value"]


# In[10]:


# plot total accidents by year
plt.bar(x, y)
plt.xlabel("Year")
plt.ylabel("Total accidents")
plt.show()


#  2001 and 2019 has the most accidents

# In[11]:


plt.plot(x, y)  
plt.show()


# In[12]:


df3 = df1.groupby(['Year', 'Category'])['Value'].nlargest(5)


# In[13]:


df3 = df1.groupby(['Year', 'Category'])['Value'].sum()


# In[14]:


type(df3)


# In[15]:


df3


# In[16]:


df1.groupby(['Year', 'Category'])['Value'].sum().unstack().plot(kind='bar',stacked = True)


# In[17]:


plt.show()


# Maximum accidents are through Verkehrsunfalle category

# In[18]:


## Data preprocessing for target value prediction
df= df[(df.Year<=2021) & (df.Category=='Alkoholunfälle') & (df.AccidentType=='insgesamt') & (df.Month!='Summe')]


# In[19]:


df


# In[20]:


df.head(30)


# In[21]:


df['Month'] = df['Month'].str[4:]


# In[22]:


df


# In[23]:


# check if there are any missing values
df.isnull().sum()


# In[24]:


df.groupby(['Year'])['Value'].sum().plot(kind='bar',stacked = True)


# In[25]:


plt.show()


# In[26]:


df.groupby(['Year', 'Month'])['Value'].sum().unstack().plot(kind='bar',stacked = True)


# In[27]:


plt.show()


# Data is now ready for model development!

# # Model Development

# In[28]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error


# ## Train-test split

# In[29]:


## train-test split
train = df[(df.Year<=2020)]
test = df[(df.Year>=2021)]


# In[30]:


train.head()


# In[31]:


train.count()


# In[32]:


train_X = train.iloc[:, [2,3]]
train_y = train.iloc[:, [4]]


# In[33]:


train_y


# In[34]:


sc_out = MinMaxScaler(feature_range=(0, 1))
scaler_output = sc_out.fit_transform(train_y[['Value']])
scaler_output =pd.DataFrame(scaler_output)
train_y=scaler_output


# In[35]:


train_y


# In[36]:


test_X = test.iloc[:, [2,3]]  
test_y = test.iloc[:, [4]]


# In[37]:


sc_out = MinMaxScaler(feature_range=(0, 1))
scaler_output = sc_out.fit_transform(test_y[['Value']])
scaler_output =pd.DataFrame(scaler_output)
test_y=scaler_output


# In[38]:


test_y


# ## Random Forest

# In[39]:


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
# Train the model on training data
rf.fit(train_X, train_y);
score = rf.score(train_X, train_y)  
print("Training score: ", score)


import sklearn.metrics as metrics
from math import sqrt

pred = rf.predict(test_X)
mae = metrics.mean_absolute_error(test_y,pred)
rms = sqrt(mean_squared_error(test_y,pred))

print("MAE:", mae)
print("MSE:", mean_squared_error(test_y,pred))
print("RMSE:", rms)


# In[40]:


pred=pred.reshape(-1,1)


# In[41]:


pred


# In[42]:


test_y


# In[43]:


pred=pred.reshape(-1,1)
Predictions = sc_out.inverse_transform(pred)
print(Predictions)


# ## Lasso Regression

# In[44]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


# In[45]:


import numpy as np
params = {'alpha': (np.logspace(-8, 8, 100))} # It will check from 1e-08 to 1e+08
lasso = Lasso(normalize=True)
lasso_model = GridSearchCV(lasso, params, cv = 10)
lasso_model.fit(train_X, train_y)


# In[46]:


print(lasso_model.best_params_)
print(lasso_model.best_score_)


# In[47]:


from sklearn import linear_model
from sklearn.metrics import *
lasso_model = linear_model.Lasso(alpha=0.00048626015800653534, normalize = True)
lasso_model.fit(train_X, train_y)
score = lasso_model.score(train_X, train_y)  
print("Training score: ", score)


import sklearn.metrics as metrics
from math import sqrt
prediction = lasso_model.predict(test_X)
mae = metrics.mean_absolute_error(test_y,prediction)
rms = sqrt(mean_squared_error(test_y,prediction))
r2score = r2_score(test_y, prediction)
 
print("MAE:", mae)
print("MSE:", mean_squared_error(test_y,prediction))
print("RMSE:", rms)


# In[48]:


pred = lasso_model.predict(test_X)
pred=pred.reshape(-1,1)


# In[49]:


Predictions = sc_out.inverse_transform(pred)
print(Predictions)


# ## Ridge Regression

# In[50]:


from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

params = {'alpha': (np.logspace(-8, 8, 100))} # It will check from 1e-08 to 1e+08
ridge = Ridge(normalize=True)
ridge_model = GridSearchCV(ridge, params, cv = 10)
ridge_model.fit(train_X, train_y)
print(ridge_model.best_params_)
print(ridge_model.best_score_)


# In[51]:


r_model = Ridge(alpha=0.18738174228603868, fit_intercept=True, normalize=True)
 
r_model.fit(train_X, train_y)
score = r_model.score(train_X, train_y)  
print("Training score: ", score)


prediction = r_model.predict(test_X)
mae = metrics.mean_absolute_error(test_y,prediction)
rms = sqrt(mean_squared_error(test_y,prediction))
r2score = r2_score(test_y, prediction)
print("MAE:", mae)
print("MSE:", mean_squared_error(test_y,prediction))
print("RMSE:", rms)
print("R2score", r2score)


# In[52]:


pred = r_model.predict(test_X)
pred=pred.reshape(-1,1)


# In[53]:


Predictions = sc_out.inverse_transform(pred)
print(Predictions)


# ## Support Vector Regression

# In[54]:


from sklearn import preprocessing
from sklearn import utils
from sklearn.metrics import *
lab_enc = preprocessing.LabelEncoder()
train_y = lab_enc.fit_transform(train_y)


# In[55]:


from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVR

gsc = GridSearchCV(

    estimator=SVR(kernel='rbf'),
    
    param_grid={
    
        'C': [0.1, 1, 100, 1000],
        
        'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
        
        'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
        
    },
    
    cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
gsc.fit(train_X, train_y)


# In[56]:


best_params = gsc.best_params_
best_params


# In[57]:


from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

rng = np.random.RandomState(0)
regr = make_pipeline(StandardScaler(), SVR(C=1000, epsilon=5, gamma=0.001))
regr.fit(train_X, train_y)


# In[58]:


score = regr.score(train_X, train_y)  
print("Training score: ", score)


pred = regr.predict(test_X)
mae = metrics.mean_absolute_error(test_y,pred)
rms = sqrt(mean_squared_error(test_y,pred))
print("MAE:", mae)
print("MSE:", mean_squared_error(test_y,pred))
print("RMSE:", rms)


# In[59]:


pred = r_model.predict(test_X)
pred=pred.reshape(-1,1)


# In[60]:


Predictions = sc_out.inverse_transform(pred)
print(Predictions)


# # Selecting the best model

#  Random forest has the best accuracy of all regression models. Let's fit the regression line on the entire data 

# In[61]:


X =  df.iloc[:, [2,3]]
y = df.iloc[:, [4]]


# In[62]:


# scaling input features
sc_in = MinMaxScaler(feature_range=(0, 1))
scaled_input = sc_in.fit_transform(df[['Year', 'Month']])
scaled_input =pd.DataFrame(scaled_input)
scaled_input
X=scaled_input.rename(columns={0:'Year', 1:'Month'})

X


# In[63]:


# scaling output features
sc_out = MinMaxScaler(feature_range=(0, 1))
scaler_output = sc_out.fit_transform(df[['Value']])
scaler_output =pd.DataFrame(scaler_output)
y=scaler_output.rename(columns={0:'Value'})

y


# In[64]:


from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
model = RandomForestRegressor(n_estimators = 100, random_state = 42)
# Train the model on training data
model.fit(X, y)


# # Model Deployment

# In[65]:


import pickle
# saving the model
pickle.dump(model, open('model.sav', 'wb'))
pickle.dump(sc_out, open('out_scaler.sav', 'wb'))
pickle.dump(sc_in, open('scaler.sav', 'wb'))


# In[66]:


import tarfile
tar = tarfile.open("model.tar.gz", "w:gz")
tar.add('model.sav')
tar.close()


# In[67]:


import boto3


# In[68]:


def upload_to_s3(channel, file):
    s3 = boto3.resource('s3')
    data = open(file, "rb")
    key = channel + file
    bucket = 'dps-ai-challenge-model'
    s3.Bucket(bucket).put_object(Key=key, Body=data)

upload_to_s3('model-weights/', 'out_scaler.sav')
upload_to_s3('model-weights/', 'scaler.sav')
upload_to_s3('model-weights/', 'model.tar.gz')
upload_to_s3('model-weights/', 'model.sav')


# In[ ]:


from sagemaker.sklearn import SKLearnModel
from sagemaker import get_execution_role

role = get_execution_role() 
model = SKLearnModel(model_data='s3://dps-ai-challenge-model/model-weights/model.tar.gz',
                             role=role, 
                             entry_point='inference.py',
                             framework_version ='0.23-1',
                            py_version='py3')
predictor = model.deploy(instance_type='ml.t2.medium', initial_instance_count=1)


# ## Testing the Model

# In[74]:


import json
  
# Opening JSON file
f = open('input.json')
print(f)
# returns JSON object as 
# a dictionary
data = json.load(f)


# In[75]:


import boto3
runtime= boto3.client('runtime.sagemaker')
payload = json.dumps(data)
print(payload)
#event is the test request you are sending
response = runtime.invoke_endpoint(EndpointName='sagemaker-scikit-learn-2022-06-20-03-59-08-850',
                                   ContentType='application/json',
                                   Body=payload)
result = json.loads(response['Body'].read().decode()) #decode response


# In[76]:


result


# In[ ]:




