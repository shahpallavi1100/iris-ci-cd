from model_train import train_model
import joblib
import os

def test_training_accuracy():
    acc = train_model()
    assert acc > 0.8

def test_model_file_created():
    assert os.path.exists("iris_model.pkl")

def test_prediction():
    model = joblib.load("iris_model.pkl")
    prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])
    assert prediction[0] in [0, 1, 2]
