import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import joblib


weather_data=pd.read_csv('Dataset/weather_classification_data_cleaned.csv')
weather_dataframe=pd.DataFrame(weather_data)


X=weather_dataframe.drop(['Weather Type'],axis=1)
y=weather_dataframe['Weather Type']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=3)


model= RandomForestClassifier(
    n_estimators=250,
    max_depth=12,
    min_samples_leaf=5,
    max_features="sqrt",
    bootstrap=True,
    class_weight="balanced",
    random_state=2,
    n_jobs=-1
)
model.fit(X_train,y_train)

print("Random Forest Train Accuracy : ",model.score(X_train,y_train))
print("Random Forest Test Accuracy : ",model.score(X_test,y_test))

joblib.dump(model,'Model/random_forest.pkl')
print("Model saved successfully")

y_pred = model.predict(X_test)
print(y_pred)
cm_test = confusion_matrix(y_test, y_pred)
print(cm_test)

