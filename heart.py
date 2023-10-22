import pandas as pd
data=pd.read_csv("data.csv")
print(data)
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
from sklearn.svm import SVC
cls=SVC()
cls.fit(x_train,y_train)
pred=cls.predict(x_test)
from sklearn.metrics import accuracy_score,recall_score,precision_score
acc=accuracy_score(y_test,pred)
p=precision_score(y_test,pred)
r=recall_score(y_test,pred)
print(acc)
import matplotlib.pyplot as plt
w= ["accuracy","precision","recall"]
h= [acc,p,r]
plt.bar(w,h)
plt.show()