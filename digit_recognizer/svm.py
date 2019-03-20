
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm

df = pd.read_csv('./train.csv')
x = df.iloc[:10000, 1:]
y = df.iloc[:10000, :1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

x_train[x_train > 0] = 1
x_test[x_test > 0] = 1

clf = svm.SVC()
clf.fit(x_train, np.array(y_train).flatten())
print(clf.score(x_test, y_test))
