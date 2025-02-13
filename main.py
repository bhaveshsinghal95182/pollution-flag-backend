from fastapi import FastAPI
import tensorflow as tf
import numpy as np

app = FastAPI()
model = tf.keras.models.load_model("car_pollution_model.keras")

@app.get("/")
async def root():
    return {"api started"}

@app.post("/predict")
async def predict(size: int, co2: float, nox: float, pm25: float, co: float):
    features = np.array([[size, co2, nox, pm25, co]])
    prediction = np.argmax(model.predict(features))
    return {"pollution_flag": int(prediction)}

# Run: uvicorn main:app --reload
