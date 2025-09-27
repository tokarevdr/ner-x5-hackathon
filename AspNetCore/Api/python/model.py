import spacy
import logging
fine_tuned_model = None
logger = logging.getLogger(__name__)

def initialize(path: str) -> None:
    global fine_tuned_model
    logger.info(f"Begin initialize model with path: {path}")
    fine_tuned_model = spacy.load(path)
    logger.info("Model initialized")

def predict(query: str) -> list[tuple[int, int, str]]:
    
    doc = fine_tuned_model(query)
    return [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
