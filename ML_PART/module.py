"""module.py

Файл для всяких функций, которые везде буду использовать
"""
import os
import torch
import pandas as pd


def calculate_ner_metrics(true_entities, pred_entities):
    """
    Calculate TP, FP, FN for each entity type (TYPE, BRAND, VOLUME, PERCENT) based on BIO tagging.
    """
    entity_types = ['TYPE', 'BRAND', 'VOLUME', 'PERCENT']
    metrics = {entity: {'TP': 0, 'FP': 0, 'FN': 0} for entity in entity_types}

    def group_entities(entities):
        grouped = {entity: [] for entity in entity_types}
        valid_entities = []
        for start, end, label in entities:
            if '-' in label:
                prefix, entity_type = label.split('-', 1)
                if entity_type in entity_types:
                    valid_entities.append((start, end, prefix, entity_type))

        valid_entities.sort(key=lambda x: x[0])
        current_entity = None
        current_type = None
        current_start = None
        current_end = None

        for start, end, prefix, entity_type in valid_entities:
            if prefix == 'B':
                if current_entity is not None:
                    grouped[current_type].append((current_start, current_end))
                current_entity = entity_type
                current_type = entity_type
                current_start = start
                current_end = end
            elif prefix == 'I' and current_entity == entity_type:
                current_end = end
            else:
                if current_entity is not None:
                    grouped[current_type].append((current_start, current_end))
                current_entity = None
                current_type = None
                current_start = None
                current_end = None

        if current_entity is not None:
            grouped[current_type].append((current_start, current_end))
        return grouped

    true_grouped = group_entities(true_entities)
    pred_grouped = group_entities(pred_entities)

    for entity_type in entity_types:
        true_spans = set(true_grouped[entity_type])
        pred_spans = set(pred_grouped[entity_type])
        metrics[entity_type]['TP'] = len(true_spans & pred_spans)
        metrics[entity_type]['FP'] = len(pred_spans - true_spans)
        metrics[entity_type]['FN'] = len(true_spans - pred_spans)

    return metrics


def calculate_macro_f1(entity_pairs, max_processing_time=1.0):
    entity_types = ['TYPE', 'BRAND', 'VOLUME', 'PERCENT']
    total_metrics = {entity: {'TP': 0, 'FP': 0, 'FN': 0} for entity in entity_types}

    results = []
    for i, pair in enumerate(entity_pairs, start=1):
        metrics = calculate_ner_metrics(pair[0], pair[1])
        results.append(metrics)

    for metrics in results:
        for entity_type in entity_types:
            total_metrics[entity_type]['TP'] += metrics[entity_type]['TP']
            total_metrics[entity_type]['FP'] += metrics[entity_type]['FP']
            total_metrics[entity_type]['FN'] += metrics[entity_type]['FN']

    f1_scores = []
    f1_per_entity = {entity: 0.0 for entity in entity_types}

    for entity_type in entity_types:
        tp = total_metrics[entity_type]['TP']
        fp = total_metrics[entity_type]['FP']
        fn = total_metrics[entity_type]['FN']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        f1_per_entity[entity_type] = f1
        if tp + fp + fn > 0:
            f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    return (macro_f1, f1_per_entity['TYPE'], f1_per_entity['BRAND'], f1_per_entity['VOLUME'], f1_per_entity['PERCENT'])


def evaluate_model(model, eval_data):
    entity_pairs = []
    for text, annotations in eval_data:
        doc = model(text)
        pred_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        true_entities = annotations['entities']
        entity_pairs.append((true_entities, pred_entities))

    macro_f1, f1_type, f1_brand, f1_volume, f1_percent = calculate_macro_f1(entity_pairs)
    return {
        'f1_macro': macro_f1,
        'f1_TYPE': f1_type,
        'f1_BRAND': f1_brand,
        'f1_VOLUME': f1_volume,
        'f1_PERCENT': f1_percent
    }


def check_repo_exists(repo_name, token=None):
    try:
        from huggingface_hub import HfApi
        if not setup_hf_login(token):
            return False

        api = HfApi()
        repo_info = api.repo_info(repo_id=repo_name)
        print(f"✅ Репозиторий существует: {repo_name}")
        print(f"   URL: https://huggingface.co/{repo_name}")
        return True
    except Exception as e:
        print(f"❌ Репозиторий не существует или недоступен: {repo_name}")
        print(f"   Ошибка: {e}")
        return False


def process_submission(trained_model, input_file='submission.csv', output_file='submission_response.csv'):
    df = pd.read_csv(input_file, sep=';')
    results = []

    for text in df['sample']:
        doc = trained_model(text)
        entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        results.append(entities)

    output_df = pd.DataFrame({
        'sample': df['sample'],
        'annotation': results
    })
    output_df.to_csv(output_file, sep=';', index=False)

def parse_span_str(span_str):
    """
    Парсит строковое представление спанов, например:
    "[(0, 4, 'B-TYPE'), (5, 12, 'I-TYPE')]"
    Возвращает list of tuples (int,int,str).
    """
    if isinstance(span_str, (list, tuple)):
        return span_str
    if not span_str or not isinstance(span_str, str):
        return []
    try:
        parsed = ast.literal_eval(span_str)
        # ensure ints
        out = []
        for s in parsed:
            if len(s) >= 3:
                out.append((int(s[0]), int(s[1]), str(s[2])))
        return out
    except Exception as e:
        raise ValueError(f"Can't parse span string: {e}")

def merge_prefixed_char_spans(spans):
    """
    Вход: spans - list of (start,end,label) где label может быть:
      - 'B-TYPE', 'I-TYPE', 'O' или просто 'TYPE' (robust)
    Возвращает список объединённых базовых спанов:
      [(start,end,'TYPE'), ...] или [(start,end,'O'), ...]
    Правило объединения: последовательности B-/I- с одинаковым типом и
    прилегающими границами (next.start == cur.end) объединяются.
    O-спаны объединяются только если непрерывны (смежны).
    """
    if not spans:
        return []
    # sort by start
    spans_sorted = sorted(spans, key=lambda x: int(x[0]))
    merged = []
    i = 0
    n = len(spans_sorted)
    while i < n:
        s0, s1, lab = spans_sorted[i]
        s0 = int(s0); s1 = int(s1)
        if lab == 'O':
            cur_s, cur_e = s0, s1
            j = i + 1
            while j < n and spans_sorted[j][2] == 'O' and int(spans_sorted[j][0]) == cur_e:
                cur_e = int(spans_sorted[j][1]); j += 1
            merged.append((cur_s, cur_e, 'O'))
            i = j
            continue

        # handle labels with B- or I- or plain
        if isinstance(lab, str) and (lab.startswith('B-') or lab.startswith('I-')):
            base = lab.split('-', 1)[1]
        else:
            base = lab  # already base
        # start new span at s0..s1
        cur_s, cur_e = s0, s1
        j = i + 1
        while j < n:
            ns0, ns1, nlab = spans_sorted[j]
            ns0 = int(ns0); ns1 = int(ns1)
            # accept continuation if it's I-base and contiguous, or plain base contiguous
            if (isinstance(nlab, str) and nlab.startswith('I-') and nlab.split('-',1)[1] == base and ns0 == cur_e) \
               or (nlab == base and ns0 == cur_e):
                cur_e = ns1
                j += 1
            else:
                break
        merged.append((cur_s, cur_e, base))
        i = j
    return merged


def tokenize_and_align_labels(text, spans_prefixed, tokenizer, add_special_tokens=True, truncation=True, max_length=None):
    """
    text: str
    spans_prefixed: list of (start,end,label) where label may be 'B-TYPE','I-TYPE','O'
    tokenizer: HuggingFace tokenizer with use_fast=True (must provide offset_mapping)
    Возвращает dict:
      {
        'tokens': [...],
        'input_ids': [...],
        'offsets': [(s,e), ...],
        'token_labels': ['B-TYPE','I-TYPE','O', ...]   # BIO per token
      }
    Логика:
      1) Сначала объединяем префиксные char-спаны в базовые entity spans (merge_prefixed_char_spans)
      2) Для каждого токена находим span с максимальным overlap. Если overlap==0 => 'O'
      3) Помечаем токен как B-<TYPE>, если токен содержит начало span (t_start <= span_start < t_end),
         иначе как I-<TYPE> (если частично или полностью внутри).
    """
    # 1) merge char spans to base spans
    merged_spans = merge_prefixed_char_spans(spans_prefixed)

    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=add_special_tokens,
        truncation=truncation,
        max_length=max_length
    )
    offsets = enc["offset_mapping"]
    input_ids = enc["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    token_labels = []
    for (t_start, t_end) in offsets:
        if t_start == t_end:
            # special token ([CLS],[SEP]) — пометим 'O' (для обучения можете заменить на -100)
            token_labels.append("O")
            continue

        best_span = None
        best_overlap = 0
        for (s_start, s_end, s_lab) in merged_spans:
            # s_lab is base label or 'O'
            overlap = min(t_end, s_end) - max(t_start, s_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_span = (s_start, s_end, s_lab)

        if best_span is None or best_overlap <= 0:
            token_labels.append("O")
        else:
            s_start, s_end, s_lab = best_span
            if s_lab == 'O':
                token_labels.append("O")
            else:
                # decide B vs I:
                if t_start <= s_start < t_end:
                    token_labels.append("B-" + s_lab)
                elif s_start <= t_start < s_end:
                    token_labels.append("I-" + s_lab)
                else:
                    # fallback
                    token_labels.append("I-" + s_lab)

    return {
        "tokens": tokens,
        "input_ids": input_ids,
        "offsets": offsets,
        "token_labels": token_labels,
    }


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


def build_label_maps_from_examples(all_prefixed_spans):
    """
    all_prefixed_spans: iterable of spans-lists (raw from CSV)
    Возвращает label2id, id2label covering all 'B-X','I-X' and 'O'.
    """
    bases = set()
    for spans in all_prefixed_spans:
        merged = merge_prefixed_char_spans(spans)
        for s,e,lab in merged:
            if lab == 'O':
                continue
            bases.add(lab)
    labels = ["O"]
    for b in sorted(bases):
        labels.append("B-" + b)
        labels.append("I-" + b)
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


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


def save_spacy_to_hf(nlp, repo_name, token=None):
    try:
        from huggingface_hub import HfApi
        import tempfile
        import json

        if not setup_hf_login(token):
            return False

        api = HfApi()
        try:
            api.repo_info(repo_id=repo_name)
            print(f"✅ Репозиторий найден: {repo_name}")
        except Exception:
            print(f"🆕 Создаем новый репозиторий: {repo_name}")
            api.create_repo(repo_id=repo_name, private=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            nlp.to_disk(temp_dir)

            # Создаем README.md
            readme_content = f"""---
language: ru
license: mit
tags:
- named-entity-recognition
- ner
- spacy
- russian
---

# Модель для извлечения сущностей: {repo_name}

Модель для извлечения сущностей TYPE, BRAND, VOLUME, PERCENT из текстов продуктов.

## Использование

```python
from module import load_spacy_from_hf

nlp = load_spacy_from_hf("{repo_name}")
```
"""
            with open(os.path.join(temp_dir, "README.md"), "w", encoding="utf-8") as f:
                f.write(readme_content)

            api.upload_folder(
                folder_path=temp_dir,
                repo_id=repo_name,
                repo_type="model"
            )

        print(f"✅ spaCy модель сохранена в HF Hub: {repo_name}")
        return True

    except Exception as e:
        print(f"❌ Ошибка сохранения spaCy на HF: {e}")
        return False


def list_my_repos(token=None):
    try:
        from huggingface_hub import HfApi
        if not setup_hf_login(token):
            return

        api = HfApi()
        repos = api.list_repos()
        print("📂 Ваши репозитории на HF Hub:")
        for repo in repos:
            print(f"   - {repo.id}")
    except Exception as e:
        print(f"❌ Ошибка получения списка репозиториев: {e}")


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
        return None


try:
    import torch
    from transformers import AutoModelForTokenClassification
    from TorchCRF import CRF


    class NERModelWithCRF(torch.nn.Module):
        def __init__(self, num_labels, model_checkpoint="DeepPavlov/rubert-base-cased"):
            super().__init__()
            self.bert = AutoModelForTokenClassification.from_pretrained(
                model_checkpoint,
                num_labels=num_labels
            )
            self.crf = CRF(num_labels)
            self.model_checkpoint = model_checkpoint

        # В файле module.py, внутри класса NERModelWithCRF, метод forward

        def forward(self, input_ids, attention_mask, labels=None):
            outputs = self.bert(input_ids, attention_mask=attention_mask)
            emissions = outputs.logits

            if labels is not None:
                # print(labels.min(), labels.max())  # Закомментировано, как у вас

                # Клонирование и замена -100 на 0
                labels_crf = labels.clone()
                labels_crf[labels == -100] = 0  # 0 - индекс для "O"

                loss = -self.crf(emissions, labels_crf, mask=attention_mask.type(torch.uint8))
                return loss
            else:
                # Замена decode на viterbi_decode
                return self.crf.viterbi_decode(emissions, mask=attention_mask.type(torch.uint8))

        def get_emissions(self, input_ids, attention_mask):
            """Дополнительный метод для получения emissions"""
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
        def from_pretrained(cls, save_directory):
            import json
            with open(os.path.join(save_directory, "model_metadata.json"), "r") as f:
                metadata = json.load(f)

            model = cls(
                num_labels=metadata["num_labels"],
                model_checkpoint=metadata["model_checkpoint"]
            )

            model.bert = AutoModelForTokenClassification.from_pretrained(save_directory)
            crf_path = os.path.join(save_directory, "crf_layer.pt")
            model.crf.load_state_dict(torch.load(crf_path))
            return model

except ImportError:
    print("⚠️ Torch/Transformers не установлены, NERModelWithCRF недоступен")


def save_bert_to_hf(model, tokenizer, config, repo_name, token=None, private=True):
    try:
        from huggingface_hub import HfApi
        import tempfile
        import json

        if not setup_hf_login(token):
            return False

        api = HfApi()
        try:
            api.repo_info(repo_id=repo_name)
            print(f"✅ Репозиторий найден: {repo_name}")
        except Exception:
            print(f"🆕 Создаем новый репозиторий для BERT: {repo_name}")
            api.create_repo(repo_id=repo_name, private=private)

        with tempfile.TemporaryDirectory() as temp_dir:
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(temp_dir)
            else:
                model.save_pretrained(temp_dir)

            tokenizer.save_pretrained(temp_dir)

            with open(os.path.join(temp_dir, "training_config.json"), "w") as f:
                json.dump(config, f, indent=2)

            # Создаем README.md для BERT модели
            readme_content = f"""---
language: ru
license: mit
tags:
- named-entity-recognition
- ner
- bert
- crf
- russian
---

# Модель для извлечения сущностей: {repo_name}

Модель для извлечения сущностей TYPE, BRAND, VOLUME, PERCENT из текстов продуктов.

## Использование

```python
from module import load_bert_from_hf

model, tokenizer, config = load_bert_from_hf("{repo_name}")
```
"""
            with open(os.path.join(temp_dir, "README.md"), "w", encoding="utf-8") as f:
                f.write(readme_content)

            api.upload_folder(
                folder_path=temp_dir,
                repo_id=repo_name,
                repo_type="model"
            )

        print(f"✅ BERT модель сохранена в HF Hub: {repo_name}")
        return True

    except Exception as e:
        print(f"❌ Ошибка сохранения BERT модели: {e}")
        return False


def load_bert_from_hf(repo_name, token=None, device=None):
    try:
        from transformers import AutoTokenizer
        from huggingface_hub import snapshot_download

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not setup_hf_login(token):
            return None, None, None

        local_dir = snapshot_download(repo_id=repo_name, token=token)
        model = NERModelWithCRF.from_pretrained(local_dir)
        model.to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(local_dir)

        import json
        with open(os.path.join(local_dir, "training_config.json"), "r") as f:
            config = json.load(f)

        print(f"✅ BERT модель загружена из: {repo_name}")
        return model, tokenizer, config

    except Exception as e:
        print(f"❌ Ошибка загрузки BERT модели: {e}")
        return None, None, None


def process_submission_bert(model, tokenizer, input_file='submission.csv', output_file='submission_response_bert.csv', id2label=None):
    bert_wrapper = HFWrapper(model, tokenizer, id2label=id2label)
    df = pd.read_csv(input_file, sep=';')
    results = []

    for text in df['sample']:
        try:
            doc = bert_wrapper(text)
            entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
            results.append(entities)
        except Exception as e:
            print(f"⚠️ Ошибка обработки текста '{text[:50]}...': {e}")
            results.append([])

    output_df = pd.DataFrame({
        'sample': df['sample'],
        'annotation': results
    })
    output_df.to_csv(output_file, sep=';', index=False)
    print(f"✅ Результаты сохранены в: {output_file}")
    print(f"📊 Обработано примеров: {len(results)}")



