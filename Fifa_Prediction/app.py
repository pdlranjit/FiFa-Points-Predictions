from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app=FastAPI()
model=joblib.load('Fifa_Prediction/models/fifa_models.pkl')

class  TeamInput(BaseModel):
   
    previous_rank:int
    rank:int
    previous_points:float

@app.post("/predict")
def predict(data:TeamInput):
    X=np.array([[data.previous_rank,data.rank,data.previous_points]])
    predicted_points=model.predict(X)[0]
    return{
        "predicted_points":round(float(predicted_points),2)
    }
