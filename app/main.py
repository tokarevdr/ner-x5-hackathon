from fastapi import FastAPI, Depends
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import aioredis
import json
import os

from app.models import NerRequest, NerResult

redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))

app = FastAPI()
redis = aioredis.Redis(host=redis_host, port=redis_port, decode_responses=True, db=0)


async def get_cache():
    return redis


@app.post("/api/predict", response_model=list[NerResult])
async def root(request: NerRequest, cache: aioredis.Redis = Depends(get_cache)):
    cache_key = request.input
    cached_item = await cache.get(cache_key)

    if cached_item:
        data = json.loads(cached_item)
        return [NerResult(**item) for item in data]
    
    json_results = jsonable_encoder([NerResult(start_index=0, end_index=5, entity="B-TYPE"),
                                     NerResult(start_index=7, end_index=11, entity="I-TYPE"),
                                     NerResult(start_index=13, end_index=13, entity="B-VOLUME"),
                                     NerResult(start_index=15, end_index=18, entity="I-VOLUME")])
    json_response = JSONResponse(content=json_results)

    await cache.set(cache_key, json_response.body)

    return json_response
