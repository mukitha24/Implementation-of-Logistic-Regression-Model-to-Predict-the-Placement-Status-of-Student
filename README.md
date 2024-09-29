# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data. 

2.Print the placement data and salary data. 

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results. 
 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:MUKITHA V M 
RegisterNumber: 212223040119 
*/
```
```
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1 = data1.drop(["sl_no", "salary"], axis = 1)#removes the specified row or column data1.head()
data1.isnull().sum()
data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0 )

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)

from sklearn.metrics import classification_report
classification_report1 =classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```
## Output:
The Logistic Regression Model to Predict the Placement Status of Student:

## Placement Data:
![328956910-7e0b7779-182c-445e-911b-f12f101c44b7](https://github.com/user-attachments/assets/4e203c33-838e-4d51-adb5-92f8f16ad421)
## Y_prediction array:
![328956951-26888a45-a07e-4b6e-ae4f-e3be30767823](https://github.com/user-attachments/assets/efbd8633-43ce-4d84-8645-8b8d8da0cb20)
## Accuracy value:
![328957000-de0dc4eb-0911-43ab-800a-9222001952b3](https://github.com/user-attachments/assets/c9a0240f-b756-4c03-945f-74d96b8104ef)
## Confusion array:
![328957049-3628787b-7dd3-40c8-b518-cf54f31efab0](https://github.com/user-attachments/assets/b740c4ea-27d4-4f0f-bbff-2796320af753)
## Classification Report:
![328957085-249f2803-8c4c-4fb1-8815-f6537c76805c](https://github.com/user-attachments/assets/bd805e5a-3b58-4f9a-a02d-9cc9a9914e14)
## prediction of LR:
![328957138-9b6898b3-84e1-4871-9172-a1855cca0c26](https://github.com/user-attachments/assets/bedc1062-517f-4675-ba2b-5ee43e3d6847)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
