#from sklearn.datasets import load_iris
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder

iris = sns.load_dataset("iris")


X = iris.iloc[:,0:4].values
y = iris.iloc[:,4].values

print(X)
print(y)

encoder =  LabelEncoder()
y1 = encoder.fit_transform(y)

Y = pd.get_dummies(y1).values

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                    test_size=0.2, 
                                                    random_state=1) 


