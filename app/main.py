from fastapi import FastAPI, Depends
import asyncio
import aioredis
import os

redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))

app = FastAPI()
redis = aioredis.Redis(host=redis_host, port=redis_port, decode_responses=True, db=0)

async def get_cache():
    return redis


@app.post("/api/predict/{item_id}")
async def root(item_id: str, cache: aioredis.Redis = Depends(get_cache)):
    cache_key = f"item_{item_id}"
    cached_item = await cache.get(cache_key)

    if cached_item:
        return {"item_id": item_id, "item": cached_item, "cached": True}
    
    item = f"Item {item_id}"

    await cache.set(cache_key, item)

    return {"item_id": item_id, "item": item, "cached": False}

