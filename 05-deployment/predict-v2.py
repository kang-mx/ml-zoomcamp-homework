import pickle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float


class PredictResponse(BaseModel):
    conversion_probability: float
    will_convert: bool


app = FastAPI(title='conversion-prediction')

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


@app.post('/predict', response_model=PredictResponse)
def predict(client: Client):
    # Convert client object to dict and wrap in a list
    client_dict = [client.dict()]

    # Predict probability
    prob = pipeline.predict_proba(client_dict)[0, 1]
    will_convert = prob >= 0.5

    return PredictResponse(
        conversion_probability=float(prob),
        will_convert=will_convert
    )


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=3000)
