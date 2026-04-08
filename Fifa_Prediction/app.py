from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import logging

#_________loggin setup
logging.basicConfig(level=logging.INFO,
                    filename="app.log",
                    format="%(asctime)s -%(levelname)s - %(message)s"
                    )



app=FastAPI()
model=joblib.load('Fifa_Prediction/models/fifa_models.pkl')
scaler = joblib.load('Fifa_Prediction/models/scaler.pkl')

class  TeamInput(BaseModel):
   
    previous_rank:int
    rank:int
    previous_points:float

@app.post("/predict")
def predict(data:TeamInput):
    try:
        #incoming request
        logging.info(f"Incoming request:{data}")

        if data.rank<=0:
            raise HTTPException(status_code=400,
                                detail="rank must be greater than 0"
                                )
        if data.previous_rank <= 0:
            raise HTTPException(
                status_code=400,
                detail='previous rank must be greater than 0'
            )
        if data.previous_points< 0:
            raise HTTPException(
                status_code=400,
                detail="Previous point cant be nagative"
            )
        

        X=np.array([[data.previous_rank,data.rank,data.previous_points]])
        X = scaler.transform(X)
        predicted_points=model.predict(X)[0]
        result=round(float(predicted_points),2)

    #log result
        logging.info(f"Prediction result: {result}")

     
        return{
           "status":"sucess",
           "predicted_points":result
        }
    except HTTPException:
     raise # it handle manual error like rank negative  previous rank less than 0
    
    except Exception as e:
      logging.error(f"Prediction error:{str(e)}")
      raise HTTPException(
        status_code=500,
        detail=f"Prediction failed:{str(e)}"
      )