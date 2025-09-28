import spacy
import logging
# const = None
fine_tuned_model = None
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize(path: str) -> None:
    # global const
    global fine_tuned_model
    logger.info(f"Begin initialize model with path: {path}")
    # const = 4
    fine_tuned_model = spacy.load(path)
    logger.info("Model initialized")

def predict(query: str) -> list[tuple[int, int, str]]:
    
    doc = fine_tuned_model(query)
    return [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
    # results:list[tuple[int, int, str]] = []
    # for i in range(const):
    #     results.append((i, i+1, f"{i}"))
    # return results