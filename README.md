## Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student


## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.


## Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook


## Algorithm
1.import pandas module.

2.Read the required csv file using pandas .

3.Import LabEncoder module.

4.From sklearn import logistic regression.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.print the required values.

8.End the program.


## Program:
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: JOSIAH IMMANUEL A
RegisterNumber: 212223043003
~~~

import pandas as pd

data = pd.read_csv("Placement_Data.csv")

print("1. Placement data")
print(data.head())


data1 = data.copy()
data1= data1.drop(["sl_no","salary"],axis=1)

print("2. Salary Data")
print(data1.head())

print("3. Checking the null() function")
print(data1.isnull().sum())

print("4. Data Duplicate")
print(data1.duplicated().sum())

from sklearn.preprocessing import LabelEncoder
lc = LabelEncoder()

data1["gender"] = lc.fit_transform(data1["gender"])
data1["ssc_b"] = lc.fit_transform(data1["ssc_b"])
data1["hsc_b"] = lc.fit_transform(data1["hsc_b"])
data1["hsc_s"] = lc.fit_transform(data1["hsc_s"])
data1["degree_t"]=lc.fit_transform(data["degree_t"])
data1["workex"] = lc.fit_transform(data1["workex"])
data1["specialisation"] = lc.fit_transform(data1["specialisation"])
data1["status"]=lc.fit_transform(data1["status"])


print("5. Print data")
print(data1)

y = data1["status"]
print("6. Data-status")
x = data1.iloc[:,:-1]
print(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")

print("7. y_prediction array")
print(lr.fit(x_train,y_train))
y_pred = lr.predict(x_test)
print(y_pred)


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print("8. Accuracy")
print(accuracy)


from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print("9. Confusion array")
print(confusion)


from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)

print("10. Classification report")
print(classification_report1)

prediction = [1,67,1,91,1,1,58,2,0,55,1,58.80]
print(lr.predict([prediction])) 


prediction = [1,80,1,90,1,1,90,1,0,85,1,85]
print("11. Prediction of LR")
print(lr.predict([prediction]))
~~~
*/

## Output:

![Screenshot 2025-04-12 102414](https://github.com/user-attachments/assets/817311b9-b51a-4424-8776-3d668b6d81e5)


![Screenshot 2025-04-12 102437](https://github.com/user-attachments/assets/2548c2cf-119f-4c4f-ab33-441636cd3717)


![Screenshot 2025-04-12 102454](https://github.com/user-attachments/assets/baf08e94-4396-46ea-a2f5-dbe48b070851)


![Screenshot 2025-04-12 102523](https://github.com/user-attachments/assets/1dad7bb4-18f9-45f7-8396-f72dd13a0fad)


![Screenshot 2025-04-12 102536](https://github.com/user-attachments/assets/460c1f81-5a95-4d7d-a6f6-706be66000a1)


![Screenshot 2025-04-12 102553](https://github.com/user-attachments/assets/7008df37-3e89-4940-9e96-2c34e90ed9e9)


![Screenshot 2025-04-12 102607](https://github.com/user-attachments/assets/2fc5ae36-1237-4093-b964-d684f86f4a36)


![Screenshot 2025-04-12 102627](https://github.com/user-attachments/assets/a9cd30a0-12c9-40a8-be3e-0477a32496cf)


![Screenshot 2025-04-12 102653](https://github.com/user-attachments/assets/61f8eb40-c2b5-4895-a696-560f59f870ad)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming
