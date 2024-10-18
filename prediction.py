import joblib


def predict(features):
    model=joblib.load("model/iris_model.pkl")
    return model.predict([features])[0]
