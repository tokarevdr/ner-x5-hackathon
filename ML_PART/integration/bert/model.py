from transformers import AutoTokenizer, AutoModelForTokenClassification
from huggingface_hub import snapshot_download
import logging
import os
import torch
from TorchCRF import CRF

fine_tuned_model = None
loaded_tokenizer = None
logger = logging.getLogger(__name__)


class NERModelWithCRF(torch.nn.Module):
    def __init__(self, num_labels, model_checkpoint="DeepPavlov/rubert-base-cased"):
        super().__init__()
        self.bert = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            num_labels=num_labels
        )
        self.crf = CRF(num_labels)
        self.model_checkpoint = model_checkpoint

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        emissions = outputs.logits

        if labels is not None:
            labels_crf = labels.clone()
            labels_crf[labels == -100] = 0
            loss = -self.crf(emissions, labels_crf, mask=attention_mask.type(torch.uint8))
            return loss
        else:
            return self.crf.viterbi_decode(emissions, mask=attention_mask.type(torch.uint8))

    def get_emissions(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids, attention_mask=attention_mask)
            return outputs.logits

    def save_pretrained(self, save_directory):
        self.bert.save_pretrained(save_directory)
        crf_path = os.path.join(save_directory, "crf_layer.pt")
        torch.save(self.crf.state_dict(), crf_path)

        metadata = {
            "model_type": "bert-crf",
            "model_checkpoint": self.model_checkpoint,
            "num_labels": self.bert.config.num_labels,
            "crf_layer": "crf_layer.pt"
        }

        import json
        with open(os.path.join(save_directory, "model_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Полная модель сохранена в: {save_directory}")

    @classmethod
    def from_pretrained(cls, save_directory, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        import json
        with open(os.path.join(save_directory, "model_metadata.json"), "r") as f:
            metadata = json.load(f)

        model = cls(
            num_labels=metadata["num_labels"],
            model_checkpoint=metadata["model_checkpoint"]
        )

        model.bert = AutoModelForTokenClassification.from_pretrained(save_directory)

        # Исправление: загружаем CRF с правильным устройством
        crf_path = os.path.join(save_directory, "crf_layer.pt")
        model.crf.load_state_dict(torch.load(crf_path, map_location=device))

        # Перемещаем всю модель на устройство
        model.to(device)
        return model

def token_labels_to_char_spans(offsets, token_labels):
    """
    offsets: list of (start,end)
    token_labels: list like ['B-BRAND','I-BRAND','O',...]
    Возвращает список char-spans в формате:
       [(start,end,'B-BRAND'), ..., (start,end,'O'), ...]
    Правила:
      - Non-O spans возвращаются как единственный B-<TYPE> спан (начало => 'B-', внутри => объединяется)
      - O-спаны возвращаются как 'O'
      - Объединяем токены в один char-span только если смежны (next.start == cur.end).
    """
    spans = []
    cur = None  # [start, end, base_label or 'O']
    for (off, lab) in zip(offsets, token_labels):
        t_s, t_e = off
        if t_s == t_e:
            # skip special tokens
            continue
        if lab == "O":
            if cur is None:
                cur = [t_s, t_e, "O"]
            else:
                if cur[2] == "O" and t_s == cur[1]:
                    # extend contiguous O span
                    cur[1] = t_e
                else:
                    # push previous and start new O span
                    spans.append((cur[0], cur[1], "B-" + cur[2] if cur[2] != "O" else "O") if cur[2] != "O" else (cur[0], cur[1], "O"))
                    cur = [t_s, t_e, "O"]
        else:
            # labels like B-X or I-X (robust to plain X)
            if lab.startswith("B-"):
                base = lab.split("-", 1)[1]
                if cur is not None:
                    # push previous
                    spans.append((cur[0], cur[1], "B-" + cur[2] if cur[2] != "O" else "O") if cur[2] != "O" else (cur[0], cur[1], "O"))
                cur = [t_s, t_e, base]
            elif lab.startswith("I-"):
                base = lab.split("-", 1)[1]
                if cur is not None and cur[2] == base and t_s == cur[1]:
                    cur[1] = t_e
                else:
                    # I- without B- : начинаем новый span (robust)
                    if cur is not None:
                        spans.append((cur[0], cur[1], "B-" + cur[2] if cur[2] != "O" else "O") if cur[2] != "O" else (cur[0], cur[1], "O"))
                    cur = [t_s, t_e, base]
            else:
                # plain label like 'TYPE' -> treat as B-<TYPE>
                base = lab
                if cur is not None:
                    spans.append((cur[0], cur[1], "B-" + cur[2] if cur[2] != "O" else "O") if cur[2] != "O" else (cur[0], cur[1], "O"))
                cur = [t_s, t_e, base]

    if cur is not None:
        if cur[2] == "O":
            spans.append((cur[0], cur[1], "O"))
        else:
            spans.append((cur[0], cur[1], "B-" + cur[2]))
    return spans

class HFWrapper:
    def __init__(self, model, tokenizer, id2label):
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label  # Добавляем id2label для преобразования ID в метки

    def __call__(self, text):
        class Doc:
            def __init__(self, ents):
                self.ents = ents

        class Ent:
            def __init__(self, start, end, label):
                self.start_char = start
                self.end_char = end
                self.label_ = label

        # Токенизация с truncation и max_length (из CONFIG)
        tokenized = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=128,  # Из CONFIG["max_length"]
            return_tensors="pt",
            return_offsets_mapping=True
        )
        input_ids = tokenized["input_ids"].to(self.model.bert.device)
        attention_mask = tokenized["attention_mask"].to(self.model.bert.device)

        # Инференс модели
        with torch.no_grad():
            pred = self.model(input_ids, attention_mask)[0]  # Список ID меток от viterbi_decode

        # Преобразуем ID меток в текстовые метки
        token_labels = [self.id2label[id] for id in pred]

        # Конвертируем в спаны с помощью token_labels_to_char_spans
        spans = token_labels_to_char_spans(tokenized["offset_mapping"][0].tolist(), token_labels)

        # Создаём entities
        ents = [Ent(s, e, l) for s, e, l in spans]
        return Doc(ents)



def setup_hf_login(token=None):
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


def load_bert_from_hf(repo_name, token=None, device=None):
    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not setup_hf_login(token):
            return None, None, None

        local_dir = snapshot_download(repo_id=repo_name, token=token)

        # Исправление: передаем device в from_pretrained
        model = NERModelWithCRF.from_pretrained(local_dir, device=device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(local_dir)

        import json
        with open(os.path.join(local_dir, "training_config.json"), "r") as f:
            config = json.load(f)

        print(f"✅ BERT модель загружена из: {repo_name}")
        return model, tokenizer, config

    except Exception as e:
        print(f"❌ Ошибка загрузки BERT модели: {e}")
        import traceback
        traceback.print_exc()  # Добавляем полную трассировку для отладки
        return None, None, None


def initialize(repo_name: str, token: str) -> None:
    global fine_tuned_model
    global loaded_tokenizer
    logger.info(f"Begin initialize model from repo: {repo_name}")

    # Определяем устройство заранее
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Используется устройство: {device}")

    fine_tuned_model, loaded_tokenizer, _ = load_bert_from_hf(
        repo_name,
        token=token,
        device=device
    )

    # Проверяем, что модель загрузилась
    if fine_tuned_model is not None and loaded_tokenizer is not None:
        logger.info("✅ Model initialized successfully")
        print("✅ Model initialized successfully")
    else:
        logger.error("❌ Failed to initialize model")
        print("❌ Failed to initialize model")


def predict(query: str) -> list[tuple[int, int, str]]:
    global fine_tuned_model
    global loaded_tokenizer

    # Проверяем, что модель и токенизатор загружены
    if fine_tuned_model is None or loaded_tokenizer is None:
        print("❌ Модель не инициализирована. Сначала вызовите initialize()")
        return []

    print(f"🔍 Токенизатор: {type(loaded_tokenizer)}, Модель: {type(fine_tuned_model)}")

    wrapper = HFWrapper(
        fine_tuned_model,
        loaded_tokenizer,
        id2label={i: label for i, label in enumerate([
            "O", "B-TYPE", "I-TYPE", "B-BRAND", "I-BRAND",
            "B-VOLUME", "I-VOLUME", "B-PERCENT", "I-PERCENT"
        ])}
    )
    doc = wrapper(query)
    return [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
