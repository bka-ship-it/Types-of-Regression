import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

data=pd.read_csv('wine.csv')
print(data.head())

X=data[['Alcohol','Malicacid','Ash','Magnesium']]
Y=data['class']

print(X.head())
print(Y.head())

X_train,Y_train,X_test,Y_test=train_test_split(X,Y,test_size=0.2,random_state=5)

standard_scaler=StandardScaler()
standard_scaler.fit_transform(X_train)
label_encoder=LabelEncoder()
label_encoder.fit_transform(Y_train)

classifier=KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train,Y_train)

standard_scaler.transform(X_test)
y_pred=classifier.predict(X_test)
label_encoder.transform(Y_test)

matrix=confusion_matrix(Y_test,y_pred)
sns.heatmap(matrix,annot=True,fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.show()

print(classification_report(Y_test,y_pred))