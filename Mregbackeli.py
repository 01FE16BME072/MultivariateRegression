import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn import cross_validation
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression

dataframe = pd.read_csv('startup.csv')
#print(dataframe.head())
X = dataframe.iloc[:,:-1].values
Y = dataframe.iloc[:,4].values

label = LabelEncoder()
X[:,3] = label.fit_transform(X[:,3])
onehot = OneHotEncoder(categorical_features = [3])
X = onehot.fit_transform(X).toarray()
X = X.astype(np.int32)
X = X[:,1:]

X  = np.append(arr = np.ones((50,1)).astype(int),values = X,axis = 1)
X_Opt = X[:,[0,1,2,3,4]]
reg = sm.OLS(endog = Y ,exog = X_Opt).fit()
print(reg.summary())

X_Opt = X[:,[0,1,3,4]]
reg = sm.OLS(endog = Y ,exog = X_Opt).fit()
print(reg.summary())

X_Opt = X[:,[0,3,4]]
reg = sm.OLS(endog = Y ,exog = X_Opt).fit()
print(reg.summary())

X_Opt = X[:,[0,3]]
reg = sm.OLS(endog = Y ,exog = X_Opt).fit()
print(reg.summary())

regline = LinearRegression()
regline.fit(X_Opt,Y)

dataframe2 = pd.read_csv('startuptest.csv')

Xp = dataframe2.iloc[:,:].values

print(regline.predict(Xp))


