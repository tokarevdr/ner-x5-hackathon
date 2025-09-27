from pydantic import BaseModel, NonNegativeInt


class NerRequest(BaseModel):
    input: str


class NerResult(BaseModel):
    start_index: NonNegativeInt
    end_index: NonNegativeInt
    entity: str