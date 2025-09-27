from abc import ABC
from app.models import NerResult

class NerModel(ABC):
    def __init__(self, model_path: str):
        ...
        
    def predict(self, input: str) -> list[NerResult]:
        ...

import spacy
class SpacyNerModel(NerModel):
    def __init__(self, model_path: str):
        super()
        self._model = spacy.load(model_path)

    def predict(self, input: str) -> list[NerResult]:
        doc = self._model(input)

        return [NerResult(start_index=ent.start_char, end_index=ent.end_char, entity=ent.label_) for ent in doc.ents]