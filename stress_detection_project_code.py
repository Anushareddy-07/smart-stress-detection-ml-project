import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('stress_detection_dataset.csv')
X = data[['Working_Hours','Sleep_Hours','Task_Pressure','Mood_Level','Break_Frequency','Screen_Time']]
y = data['Stress_Level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

sample = [[10,5,8,3,2,11]]
print("Predicted Stress:", model.predict(sample)[0])