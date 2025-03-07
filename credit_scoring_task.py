# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score,recall_score,precision_score

#load the dataset
df = pd.read_csv('/content/data_train.csv')
df

#Data Preprocessing
df.describe()

df.head()

df.shape

df['Score_point'].value_counts()

df.info()

df.isnull().sum()

x=df.drop(['Score_point'],axis=1)
y=df['Score_point']
print(x.head())
print(y.head())

# Split into train and test sets
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=2)

print(train_x.head())
print(train_y.head())

# Standardize the features
sc = StandardScaler()
X_train = sc.fit_transform(train_x)
X_test = sc.transform(test_x)

"""# Logistic Regression Model"""

#Train the logistic regression model
from sklearn.linear_model import LogisticRegression
model_lr=LogisticRegression(max_iter=1000)
model_lr.fit(train_x,train_y)
y_pred_train=model_lr.predict(train_x)
y_pred_test=model_lr.predict(test_x)

print(y_pred_train)

print(y_pred_test)

#accuracy of the logistic regression model
accuracy_train=accuracy_score(train_y,y_pred_train)
print("accuracy of the logistic regression model on your train", accuracy_train)
accuracy_test = accuracy_score(test_y, y_pred_test)
print("accuracy of the logistic regression model on your test", accuracy_test)

confusion_matrix(train_y,y_pred_train)

confusion_matrix(test_y, y_pred_test)

predictions = model_lr.predict_proba(X_test)
predictions

# predict f1_score,precision,recall

f1 = f1_score(test_y, y_pred_test, average='weighted')
print("f1_score of the logistic regression model on your test", f1)

precision = precision_score(test_y, y_pred_test, average='weighted')
print("precision of the logistic regression model on your test", precision)

recall = recall_score(test_y, y_pred_test, average='weighted')
print("recall of the logistic regression model on your test", recall)

metrics = {
    'Accuracy (Train)': accuracy_train,
    'Accuracy (Test)': accuracy_test,
    'F1-Score': f1,
    'Precision': precision,
    'Recall': recall
}

metrics_df = pd.DataFrame([metrics])
metrics_df
