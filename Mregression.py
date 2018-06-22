import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
from matplotlib import pyplot as plt
import operator

dataframe = pd.read_csv('startup.csv')

#print(dataframe.head())
X = dataframe.iloc[:,:-1].values
Y = dataframe.iloc[:,4].values

label = LabelEncoder()
X[:,3] = label.fit_transform(X[:,3])
onehot = OneHotEncoder(categorical_features = [3])
X = onehot.fit_transform(X).toarray()

X = X[:,1:]

X = X.astype(np.int32)

print(X)

X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X,Y,test_size = 0.2)

plt.scatter(X[:,2],Y,color = 'blue')
plt.scatter(X[:,3],Y,color = 'red')
plt.scatter(X[:,4],Y,color = 'yellow')
#plt.scatter(X[:,5],Y,color = 'yellow')
plt.show()

reg = LinearRegression()
regLine = reg.fit(X_train,Y_train)

pred = reg.predict(X_test)

print(pred)



