import pickle
from fastapi import FastAPI
from pydantic import BaseModel

class Defect(BaseModel):
    defect_type: str
    defect_location: str
    severity: str
    inspection_method: str
    product_id: str
    month: int
    day_of_week: int

class PredictResponse(BaseModel):
    repair_cost: float

app = FastAPI(title='repair-cost-prediction')

# Load model + DictVectorizer
with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

@app.post('/predict', response_model=PredictResponse)
def predict(defect: Defect):
    defect_dict = [defect.dict()]  # convert to list of dicts
    X = dv.transform(defect_dict)
    y_pred = model.predict(X)[0]
    return PredictResponse(repair_cost=float(y_pred))

if __name__ == '__main__': 
    uvicorn.run(app, host='0.0.0.0', port=3000)