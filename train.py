import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle


df = pd.read_csv('cellular_churn_greece.csv')
X = df.drop('churned', axis=1)
y = df['churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
clf = LogisticRegression()
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

print('Accuracy score on test set:', clf.score(X_test, y_test))

pickle.dump(clf, open('churn_model.pkl', 'wb'))
