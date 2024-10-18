import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
#1.
iris=pd.read_csv('data/iris.csv')
print(f"columns:{iris.columns}")

X=iris.drop(['Id','Species'], axis=1)
y=iris['Species']

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

#2.
model=RandomForestClassifier()
model.fit(X_train, y_train)

y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)

print(f"Accuracy of the model:{accuracy}")

joblib.dump(model, "model/iris_model.pkl")
