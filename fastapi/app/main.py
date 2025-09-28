from fastapi import FastAPI
import os


from app.models import NerRequest, NerResult
from app.nermodel import NerModel, SpacyNerModel

script_directory = os.path.dirname(os.path.abspath(__file__))
configuration = os.getenv("CONFIGURATION")

if configuration == "Development":
    script_directory = os.path.join(script_directory, os.path.pardir, os.path.pardir, "ModelIntegration")

model_path = os.path.join(script_directory, 'model')
app = FastAPI()
model: NerModel = SpacyNerModel(model_path)

@app.post("/api/predict", response_model=list[NerResult])
async def root(request: NerRequest):
    result = model.predict(request.input)

    return result
