import  joblib

model = joblib.load('main.pkl')

print(model.predict([[15]]))