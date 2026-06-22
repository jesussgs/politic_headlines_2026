from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from transformers import TrainingArguments, Trainer
import os
import torch.nn as nn
from transformers import AutoModel


# ---------- Config ----------

MODEL_NAME = "IIC/RigoBERTa-2.0"
#MODEL_NAME = "BSC-LT/MrBERT-es"
#MODEL_NAME = "FacebookAI/xlm-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

METHOD = MODEL_NAME
TEST_SIZE = 0.2

SEED = 42
#SEED = 0

CLEAN_DATASET = False

#TRAIN_CSV = "train_public.csv"
TRAIN_CSV = "/train_corpora/train_corpora.csv"
DEV_CSV = "/development_political/dev_public.csv"
OUTPUT_SUBMISSION = "./results.csv"
IMAGES_DIR = Path("images")
SAVE_METRICS_JSON = True

TITLE_COLS = [f"title_{i}" for i in range(1, 11)]
TOKENS_ALL = [f"t{i}" for i in range(1, 11)]

NDCG_K = 10
ALPHA = 0.9
N_COLS = 10  # t1..t10

"""## Download dataset

We download the development phase dataset and extract it locally.
"""

"""## Loading and analysis

Functions required to load the training dataset and perform an initial analysis to inspect its contents.

"""

def validate_columns(df: pd.DataFrame) -> None:
    required = ["id", "article_body", "image_hash"] + TITLE_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}\nPresentes: {list(df.columns)}")

df_train = pd.read_csv(TRAIN_CSV)
df_test= pd.read_csv(DEV_CSV)
validate_columns(df_train)
validate_columns(df_test)

train_df = df_train.copy()
test_df = df_test.copy()


"""## Auxiliary functions

Functions required for the correct execution of the pipeline.

"""

def tokens(x: Any) -> List[str]:
    if x is None:
        return []
    s = str(x).strip()
    return s.split() if s else []

def stable_seed(global_seed: int, row_key: str) -> int:
    h = hashlib.sha256(f"{global_seed}|{row_key}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)

def get_source_text_task1(row: pd.Series) -> str:
    return str(row.get("article_body", "") or "").strip()

def get_source_text_task2(row: pd.Series) -> str:
    # baseline naïve: usar image_hash como "señal"
    body = str(row.get("article_body", "") or "").strip()
    img = str(row.get("image_hash", "") or "").strip()
    return (img + "\n\n" + body).strip() if img else body

def get_titles(row: pd.Series) -> List[str]:
    return [str(row.get(c, "") or "") for c in TITLE_COLS]

def is_valid_rank(rank_tokens: List[str]) -> bool:
    return len(rank_tokens) == 10 and set(rank_tokens) == set(TOKENS_ALL)

def find_image_path(images_dir: Path, image_hash: str) -> Optional[Path]:
    if not image_hash or (isinstance(image_hash, float) and np.isnan(image_hash)):
        return None
    h = str(image_hash).strip()
    if not h:
        return None

    # Probar extensiones comunes
    exts = [".jpg", ".jpeg", ".png", ".webp"]
    for ext in exts:
        p = images_dir / f"{h}{ext}"
        if p.exists():
            return p

    # Si por algún motivo ya viene con extensión
    p = images_dir / h
    if p.exists():
        return p

    return None

def _minmax_01(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn)

from datasets import Dataset

TITLE_COLS = [f"title_{i}" for i in range(1, 11)]

def gold_index_from_ytrue(y_true: str) -> int:
    tok = str(y_true).strip().split()[0]  # "t9"
    return int(tok[1:]) - 1


def crop_article_head_tail(text: str, total_chars: int = 5000, tail_chars: int = 1500) -> str:
    text = str(text or "").strip()
    if len(text) <= total_chars:
        return text

    tail_chars = min(tail_chars, total_chars // 2)
    head_chars = total_chars - tail_chars

    head = text[:head_chars].strip()
    tail = text[-tail_chars:].strip()

    return head + "\n...\n" + tail
    
def build_listwise_dataset_from_df(df, task: int, article_chars: int = 5000, use_head_tail = False) -> Dataset:
    rows = []
    for _, row in df.iterrows():
        # task 1: solo texto
        #article = str(row.get("article_body", "") or "")[:article_chars]
        article = str(row.get("article_body", "") or "").strip()
        titles = [str(row.get(c, "") or "") for c in TITLE_COLS]
        gold = gold_index_from_ytrue(row["y_true"])

        rows.append({
            "article": article,
            "titles": titles,
            "label": gold,   # int 0..9
        })
    return Dataset.from_list(rows)

class ListwiseCollator:
    def __init__(self, tokenizer, max_length=384):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        # features: list de dicts con "article", "titles" (len=10), "label"
        B = len(features)
        all_input_ids = []
        all_attn = []
        all_type_ids = []
        labels = []

        for f in features:
            art = f["article"]
            titles = f["titles"]
            enc = self.tokenizer(
                [art] * len(titles),
                titles,
                truncation="only_first",
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            all_input_ids.append(enc["input_ids"])          # (10,L)
            all_attn.append(enc["attention_mask"])          # (10,L)
            if "token_type_ids" in enc:
                all_type_ids.append(enc["token_type_ids"])  # (10,L)
            labels.append(int(f["label"]))

        batch = {
            "input_ids": torch.stack(all_input_ids, dim=0),        # (B,10,L)
            "attention_mask": torch.stack(all_attn, dim=0),        # (B,10,L)
            "labels": torch.tensor(labels, dtype=torch.long),      # (B,)
        }
        if all_type_ids:
            batch["token_type_ids"] = torch.stack(all_type_ids, dim=0)  # (B,10,L)
        return batch


class EncoderListwiseRanker(nn.Module):
    def __init__(self, model_name=MODEL_NAME, dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.scorer = nn.Linear(hidden, 1)  # score escalar por par

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        # input_ids: (B,10,L) → aplanamos a (B*10,L)
        B, K, L = input_ids.shape
        input_ids = input_ids.view(B*K, L)
        attention_mask = attention_mask.view(B*K, L)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(B*K, L)

        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
        )
        cls = out.last_hidden_state[:, 0, :]          # (B*K,H)
        s = self.scorer(self.drop(cls)).squeeze(-1)   # (B*K,)
        scores = s.view(B, K)                          # (B,10)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(scores, labels)

        return {"loss": loss, "scores": scores}

def compute_metrics(eval_pred):
    scores, labels = eval_pred  # scores: (B, K), labels: (B,)
    scores = np.array(scores)
    labels = np.array(labels)

    pred_top1 = np.argmax(scores, axis=1)
    top1_acc = (pred_top1 == labels).mean()

    ranks = []
    for s, y in zip(scores, labels):
        order = np.argsort(-s)  # descendente
        rank = np.where(order == y)[0][0] + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    mrr = np.mean(1.0 / ranks)
    hit3 = np.mean(ranks <= 3)
    hit5 = np.mean(ranks <= 5)

    return {
        "top1_acc": float(top1_acc),
        "mrr": float(mrr),
        "hit@3": float(hit3),
        "hit@5": float(hit5),
    }
    
def train_listwise_encoder(
    df_train: pd.DataFrame,
    task: int,
    out_dir: str = "./rigoberta_listwise",
    article_chars: int = 3500,
    max_length: int = 384,
):
    ds = build_listwise_dataset_from_df(df_train, task=task, article_chars=article_chars, use_head_tail=True).shuffle(seed=SEED)
    split = ds.train_test_split(test_size=0.15, seed=SEED)
    ds_train, ds_eval = split["train"], split["test"]

    model = EncoderListwiseRanker(MODEL_NAME)

    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=8,   # 8 for BETO, 2 for RigoBerta, 1 for EuroBert
        per_device_eval_batch_size=8,
        #auto_find_batch_size=True,
        num_train_epochs=10,
        #warmup_ratio=0.1,
        weight_decay=0.005,
        bf16=True,
        tf32=True,
        logging_steps=200,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_top1_acc",   # o eval_mrr / eval_top1_acc
        greater_is_better=True,             # True si usas mrr o acc
        report_to="none",
        remove_unused_columns=False,
    )

    collator = ListwiseCollator(tokenizer, max_length=max_length)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    os.makedirs(out_dir, exist_ok=True)
    torch.save(trainer.model.state_dict(), f"{out_dir}/listwise_ranker.pt")
    tokenizer.save_pretrained(out_dir)

    # también guarda el nombre base del encoder en un txt
    with open(f"{out_dir}/base_model.txt", "w", encoding="utf-8") as f:
        f.write(MODEL_NAME)
    return out_dir


encoder_dir = train_listwise_encoder(train_df, task=1, out_dir="/output_political/rigoberta_listwise", max_length=512, article_chars=5000)
