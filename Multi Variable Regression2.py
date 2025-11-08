import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

data=pd.read_csv('iris.csv')
print(data)
print(data.info())
data['species'].replace({"setosa":0,"versicolor":1,"virginia":2})
data['species']

X=np.array([1,2,3,4,5,6,7,8,9]).reshape(-1,1)
y=np.array([2,5,4,6,7,3,8,9,1])
model=LinearRegression()
model.fit(X,y)
plt.scatter(X,y)
plt.plot(X, model.predict(X))
plt.title('Simple Linear Regression w/ Iris')
plt.show()

X_multi=pd.DataFrame({
    'feature1':[1,2,3,4,5,6,7,8,9],
    'feature2':[9,8,7,6,5,4,3,2,1]})
y_multi=np.array([5,6,7,8,9,10,11,12,13])
model_multi=LinearRegression()
model_multi.fit(X_multi,y_multi)
print('Multi-variable regression w/ Iris',model_multi.coef_)

np.random.seed(0)
X_poly=np.linspace(0,10,50).reshape(-1, 1)
y_poly=0.5*X_poly**2+2*X_poly+3*np.random.randn(50,1)*5
poly=PolynomialFeatures(degree=2)
X_poly_trans=poly.fit_transform(X_poly)
model_poly=LinearRegression()
model_poly.fit(X_poly_trans,y_poly)
plt.scatter(X_poly,y_poly)
plt.plot(X_poly,model_poly.predict(X_poly_trans))
plt.title('Polynomial Regression w/ Iris')
plt.show()