#!/usr/bin/env python
# coding: utf-8

# In[101]:


import matplotlib.pyplot as plt
import warnings
import numpy as np
from sklearn import datasets, preprocessing, model_selection, svm, metrics
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from math import log


# In[31]:


dataset = pd.read_csv("C:/Users/KIM/Desktop/in_total_practive.csv", index_col = 'date')
#rain 전처리
dataset['rain']=dataset['rain'].fillna(0)
dataset['count'] = pd.to_numeric(dataset['count'])


# In[32]:


dataset['rain']


# In[33]:


X = dataset.iloc[:1564, 0:3].values #2019/12/31 까지
y = dataset.iloc[:1564,3].values


# In[34]:


X


# In[36]:


#전처리
from sklearn.preprocessing import StandardScaler

#binary 변수 처리...

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))


# In[45]:


X.shape #(1564,3)


# In[38]:


y


# In[51]:


#test, train set으로 나누기 (일단 임의로)
n = int(1564*0.7)
print(n)
train_X = X[:n,:]
test_X = X[n:,:]

train_y = y[:n]
test_y = y[n:]


# In[58]:


train_X.shape #(1094,3)
test_y.shape


# # SVR

# In[54]:


#SVR

regressor = SVR(kernel = 'rbf') #rbf, linear, poly...grid search로 최적의 파라미터 찾기
regressor.fit(train_X, train_y)


# In[59]:


#predict
y_pred = regressor.predict(test_X) 
y_pred = sc_y.inverse_transform(y_pred) #정규화된 값에서 다시 원래 값으로 


# In[61]:


y_pred


# In[ ]:


#plot
X_grid = np.arange(min(X), max(X), 0.01) #this step required because data is feature scaled.
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# # Grid Search

# In[62]:


#parameter 
param = {'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),'C' : [1,5,10, 50, 100],'degree' : [2,3,4],'coef0' : [0.01,10,0.5],'gamma' : ['scale'], 'max_iter': [1000000]}





# In[64]:


#모델 정의
modelsvr = SVR()

grids = GridSearchCV(modelsvr,param,cv=5)


# In[65]:


#모델 fit
grids.fit(train_X,train_y)
print('best_params:', grids.best_params_) #best_params: {'C': 5, 'coef0': 0.01, 'degree': 3, 'gamma': 'scale', 'kernel': 'poly', 'max_iter': 1000000}


# In[68]:


#뽑힌 parameter아래 대입
model = svm.SVR(kernel="poly", C=100, degree=3, epsilon=0.1,max_iter=1000000, coef0=10, gamma = 'scale')
model.fit(train_X, train_y)
pred_y= model.predict(test_X)
y_pred = sc_y.inverse_transform(y_pred)


# In[102]:


# 평가 지표 : AIC, r^2 , mae, mape, smape, mse

#for AIC

# calculate aic for regression
def aic(n, mse, num_params): #n:len(y)
    aic = n * log(mse) + 2 * num_params
    return aic


def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

def smape(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)



# In[103]:



r_square = metrics.r2_score(test_y, pred_y)
mae=mean_absolute_error(test_y, pred_y)
mape = mape(test_y, pred_y)
smape = smape(test_y, pred_y)
mse = mean_squared_error(test_y, pred_y, squared = False)
AIC= aic(len(test_y), mse, 3)# 설명변수 개수 넣기

print("AIC={:.5f}".format(AIC))
print("r_square={:.5f}".format(r_square))
print("mae={:.5f}".format(mae))
print("mape={:.5f}".format(mape))
print("smape={:.5f}".format(smape))
print("mse={:.5f}".format(mse))


# In[ ]:




