import spacy
import logging
import os

fine_tuned_model = None
logger = logging.getLogger(__name__)

def setup_hf_login(token:str):
    try:
        from huggingface_hub import login
        if token is None:
            token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        if token:
            login(token=token)
            print("✅ Авторизация HF настроена")
            return True
        else:
            print("⚠️ Токен HF не найден")
            return False
    except ImportError:
        print("❌ huggingface_hub не установлен. Установите: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ Ошибка авторизации HF: {e}")
        return False


def load_spacy_from_hf(repo_name, token=None):
    try:
        from huggingface_hub import snapshot_download
        import spacy

        if not setup_hf_login(token):
            return None

        model_path = snapshot_download(repo_id=repo_name, token=token)
        nlp = spacy.load(model_path)
        print(f"✅ spaCy модель загружена из: {repo_name}")
        return nlp

    except Exception as e:
        print(f"❌ Ошибка загрузки spaCy модели: {e}")
        raise e

def initialize(repo_name: str, token:str) -> None:
    global fine_tuned_model
    logger.info(f"Begin initialize model from repo: {repo_name}")
    fine_tuned_model = load_spacy_from_hf(repo_name, token=token)
    logger.info("Model initialized")

def predict(query: str) -> list[tuple[int, int, str]]:
    doc = fine_tuned_model(query)
    return [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
