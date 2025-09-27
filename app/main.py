from fastapi import FastAPI
import os

from app.models import NerRequest, NerResult
from app.nermodel import NerModel, SpacyNerModel

script_directory = os.path.dirname(os.path.abspath(__file__))
model_path = script_directory + "/nermodels/custom_ru_core_news_lg_with_9_labels_50_epochs"

app = FastAPI()
model: NerModel = SpacyNerModel(model_path)

@app.post("/api/predict", response_model=list[NerResult])
async def root(request: NerRequest):
    result = model.predict(request.input)

    return result
