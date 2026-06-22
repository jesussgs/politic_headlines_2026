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
import torch.nn as nn
from datasets import Dataset
from transformers import CLIPProcessor, CLIPModel
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

# =========================
# Config
# =========================
SEED = 42
TRAIN_CSV = "/train_corpora/train_corpora.csv"
IMAGES_DIR = Path("/train_corpora/images")
TEST_SIZE = 0.20

TITLE_COLS = [f"title_{i}" for i in range(1, 11)]
TOKENS_ALL = [f"t{i}" for i in range(1, 11)]
N_COLS = 10
ALPHA = 0.9

# Misma configuración de inferencia que en tus runs
MAX_LENGTH = 512
ARTICLE_CHARS = 5000

DEFAULT_MAX_LENGTH = 512
MRBERT_MAX_LENGTH = 1024

def resolve_text_max_length(model_name: str, model_dir: str) -> int:
    """
    Usa 1024 para modelos MrBERT-es y 512 para el resto.
    Detecta tanto por nombre lógico del ensemble como por ruta.
    """
    key = f"{model_name} {model_dir}".lower()

    if "mrbert" in key or "mrbertes" in key or "mrbért" in key:
        return MRBERT_MAX_LENGTH

    return DEFAULT_MAX_LENGTH

# =========================
# Utilidades generales
# =========================
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
    
def validate_columns(df: pd.DataFrame) -> None:
    required = ["id", "article_body", "image_hash", "y_true"] + TITLE_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}\nPresentes: {list(df.columns)}")


def gold_index_from_ytrue(y_true: str) -> int:
    tok = str(y_true).strip().split()[0]
    return int(tok[1:]) - 1


def _parse_rank_list(x: Any) -> List[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    s = str(x).strip()
    if not s:
        return []

    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(t).strip() for t in arr if str(t).strip()]
        except Exception:
            pass

    s = s.replace("\t", " ").replace("\n", " ").replace("\r", " ").replace(";", " ")
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
    else:
        parts = [p.strip() for p in s.split()]
    return [p for p in parts if p]


def _token_to_col(tok: Any) -> Optional[int]:
    if tok is None or (isinstance(tok, float) and pd.isna(tok)):
        return None
    s = str(tok).strip()
    if len(s) < 2:
        return None
    prefix = s[0].lower()
    if prefix not in ("t", "d"):
        return None
    try:
        return int(s[1:])
    except Exception:
        return None


def _unique_valid_pred_cols(pred: List[str], n_cols: int) -> List[int]:
    out: List[int] = []
    seen = set()
    for tok in pred:
        n = _token_to_col(tok)
        if n is None or not (1 <= n <= n_cols):
            continue
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def _ndcg_from_ideal(pred_cols: List[int], ideal_cols: List[int], k: int) -> float:
    if not ideal_cols:
        return 0.0

    ideal_rank: Dict[int, int] = {c: i for i, c in enumerate(ideal_cols)}

    def gain_for_col(c: int) -> float:
        r = ideal_rank.get(c, None)
        if r is None:
            return 0.0
        return float(len(ideal_cols) - r)

    dcg = 0.0
    for i, c in enumerate(pred_cols[:k], start=1):
        dcg += gain_for_col(c) / math.log2(i + 1)

    idcg = 0.0
    for i, c in enumerate(ideal_cols[:k], start=1):
        idcg += gain_for_col(c) / math.log2(i + 1)

    if idcg <= 0.0:
        return 0.0
    return max(0.0, min(1.0, dcg / idcg))


def pa_ndcg(pred_tokens: List[str], true_tokens: List[str], k: int = 10, alpha: float = 0.9) -> float:
    if not pred_tokens or not true_tokens:
        return 0.0

    ideal_cols = _unique_valid_pred_cols(true_tokens, N_COLS)
    pred_cols = _unique_valid_pred_cols(pred_tokens, N_COLS)

    if not ideal_cols or not pred_cols:
        return 0.0

    if pred_cols[0] != ideal_cols[0]:
        return 0.0

    primary = ideal_cols[0]
    pred_rest = [c for c in pred_cols if c != primary]
    ideal_rest = [c for c in ideal_cols if c != primary]

    aux = _ndcg_from_ideal(pred_rest, ideal_rest, k=k)
    score = alpha + (1.0 - alpha) * aux
    return max(0.0, min(1.0, score))


def evaluate_ranking_predictions(df: pd.DataFrame, pred_col: str = "pred_task_1") -> Dict[str, float]:
    top1_hits = []
    ranks = []
    pa_scores = []

    for _, row in df.iterrows():
        pred_tokens = str(row[pred_col]).strip().split()
        gold_idx = gold_index_from_ytrue(row["y_true"])
        gold_tok = f"t{gold_idx+1}"

        if len(pred_tokens) != 10:
            continue

        top1_hits.append(float(pred_tokens[0] == gold_tok))
        rank_pos = pred_tokens.index(gold_tok) + 1
        ranks.append(rank_pos)
        pa_scores.append(pa_ndcg(pred_tokens, _parse_rank_list(row["y_true"]), k=10, alpha=ALPHA))

    ranks = np.array(ranks, dtype=np.int32)
    return {
        "top1_acc": float(np.mean(top1_hits)),
        "mrr": float(np.mean(1.0 / ranks)),
        "hit@3": float(np.mean(ranks <= 3)),
        "hit@5": float(np.mean(ranks <= 5)),
        "pa_ndcg@10": float(np.mean(pa_scores)),
    }


# =========================
# Split de validación idéntico al entrenamiento
# =========================
def build_listwise_dataset_from_df(df: pd.DataFrame, article_chars: int = 5000) -> Dataset:
    rows = []
    for row_idx, (_, row) in enumerate(df.iterrows()):
        #article = str(row.get("article_body", "") or "")[:article_chars]
        article = str(row.get("article_body", "") or "").strip()
        titles = [str(row.get(c, "") or "") for c in TITLE_COLS]
        gold = gold_index_from_ytrue(row["y_true"])
        rows.append({
            "row_idx": row_idx,
            "article": article,
            "titles": titles,
            "label": gold,
        })
    return Dataset.from_list(rows)


def get_validation_df(df_train: pd.DataFrame, article_chars: int = 5000, test_size: float = 0.15, seed: int = 0) -> pd.DataFrame:
    ds = build_listwise_dataset_from_df(df_train, article_chars=article_chars).shuffle(seed=seed)
    split = ds.train_test_split(test_size=test_size, seed=seed)
    ds_eval = split["test"]
    eval_indices = list(ds_eval["row_idx"])
    return df_train.iloc[eval_indices].copy().reset_index(drop=True)

    
def normalize_scores(scores: np.ndarray, mode: Optional[str] = "zscore") -> np.ndarray:
    x = scores.astype(np.float32)
    if mode is None:
        return x
    if mode == "minmax":
        mn, mx = float(np.min(x)), float(np.max(x))
        if mx - mn < 1e-12:
            return np.zeros_like(x, dtype=np.float32)
        return (x - mn) / (mx - mn)
    if mode == "zscore":
        mu = float(np.mean(x))
        sd = float(np.std(x))
        if sd < 1e-12:
            return np.zeros_like(x, dtype=np.float32)
        return (x - mu) / sd
    raise ValueError(f"Modo de normalización no soportado: {mode}")

# =========================
# Modelo listwise usado en tus entrenamientos
# =========================
class EncoderListwiseRanker(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        hidden = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.scorer = nn.Linear(hidden, 1)  # sigue en float32

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        B, K, L = input_ids.shape
        input_ids = input_ids.view(B * K, L)
        attention_mask = attention_mask.view(B * K, L)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(B * K, L)

        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
        )

        cls = out.last_hidden_state[:, 0, :]   # puede venir en bf16
        cls = self.drop(cls)

        # Asegurar mismo device y dtype que la cabeza lineal
        cls = cls.to(device=self.scorer.weight.device, dtype=self.scorer.weight.dtype)

        s = self.scorer(cls).squeeze(-1)
        scores = s.view(B, K)

        loss = None
        if labels is not None:
            labels = labels.to(scores.device)
            loss = nn.CrossEntropyLoss()(scores, labels)

        return {"loss": loss, "scores": scores}


class FullReranker:
    def __init__(self, model_dir: str, max_length: int = 512, article_chars: int = 5000, device: str = "cuda:0"):
        self.model_dir = model_dir
        self.max_length = max_length
        self.article_chars = article_chars
        self.device = device if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        with open(Path(model_dir) / "base_model.txt", "r", encoding="utf-8") as f:
            base_model = f.read().strip()

        self.model = EncoderListwiseRanker(model_name=base_model)
        sd = torch.load(Path(model_dir) / "listwise_ranker.pt", map_location="cpu")
        self.model.load_state_dict(sd)
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def score_titles(self, article: str, titles: List[str]) -> np.ndarray:
        art = str(article or "")#[: self.article_chars]
        enc = self.tokenizer(
            [art] * len(titles),
            titles,
            truncation="only_first",
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        input_ids = enc["input_ids"].unsqueeze(0)
        attn = enc["attention_mask"].unsqueeze(0)
        type_ids = enc.get("token_type_ids", None)
        if type_ids is not None:
            type_ids = type_ids.unsqueeze(0)

        out = self.model(
            input_ids=input_ids,
            attention_mask=attn,
            token_type_ids=type_ids,
        )
        return out["scores"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        
        
class TextEnsembleReranker:
    def __init__(
        self,
        model_dirs: Dict[str, str],
        weights: Dict[str, float],
        max_length: int = 512,
        article_chars: int = 5000,
        normalization: str = "zscore",
        max_lengths_by_model: Optional[Dict[str, int]] = None,
    ):
        self.model_dirs = model_dirs
        self.weights = weights
        self.max_length = max_length
        self.article_chars = article_chars
        self.normalization = normalization
        self.max_lengths_by_model = max_lengths_by_model or {}

        self.models = {}

        for name, model_dir in model_dirs.items():
            model_max_length = self.max_lengths_by_model.get(
                name,
                resolve_text_max_length(name, model_dir)
            )

            print(f"[INFO] Loading text model '{name}' with max_length={model_max_length}")

            self.models[name] = FullReranker(
                model_dir=model_dir,
                max_length=model_max_length,
                article_chars=article_chars,
            )

    @torch.no_grad()
    def score_titles_per_model(self, article: str, titles: List[str]) -> Dict[str, np.ndarray]:
        out = {}
        for name, model in self.models.items():
            out[name] = model.score_titles(article, titles)
        return out

    @torch.no_grad()
    def score_titles(self, article: str, titles: List[str]) -> np.ndarray:
        model_scores = self.score_titles_per_model(article, titles)

        final_scores = np.zeros(len(titles), dtype=np.float32)
        for name, scores in model_scores.items():
            s = normalize_scores(scores, mode=self.normalization)
            final_scores += float(self.weights[name]) * s

        return final_scores

    @torch.no_grad()
    def rank_titles(self, article: str, titles: List[str]) -> List[int]:
        scores = self.score_titles(article, titles)
        return np.argsort(-scores, kind="mergesort").tolist()



device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

siglip_model_name = "google/siglip-base-patch16-256-multilingual"
clip_model = AutoModel.from_pretrained(siglip_model_name, dtype=torch.bfloat16).to(device) #, device_map="auto").to(device)
clip_processor = AutoProcessor.from_pretrained(siglip_model_name)
clip_model.eval()

@torch.inference_mode()
def siglip_logits_image_vs_titles(
    clip_model,
    clip_processor,
    device: str,
    image_path: Path,
    titles: List[str],
) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    max_len = clip_model.config.text_config.max_position_embeddings

    inputs = clip_processor(
        text=titles,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    ).to(device)
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = clip_model(**inputs)
    return outputs.logits_per_image[0].detach().float().cpu().numpy().astype(np.float32)
    

@torch.inference_mode()
def clip_multilingual_logits_image_vs_titles(
    clip_model,
    clip_img_model,
    image_path: Path,
    titles: List[str],
) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")

    image_emb = clip_img_model.encode(
        [image],
        convert_to_tensor=True,
        show_progress_bar=False,
    )  # (1, D)

    text_emb = clip_model.encode(
        titles,
        convert_to_tensor=True,
        show_progress_bar=False,
    )  # (10, D)

    sims = util.cos_sim(image_emb, text_emb)[0]   # (10,)
    return sims.detach().float().cpu().numpy().astype(np.float32)
        
def predict_task2_siglip_plus_text_ensemble_gated(
    df_pred: pd.DataFrame,
    images_dir: Path,
    text_ensemble: TextEnsembleReranker,
    clip_model,
    clip_img_model,
    clip_processor,
    device: str,
    w_img: float = 0.20,
    gate_gap: float = 0.10,
    img_when_confident: float = 0.00,
    text_norm_for_fusion: str = "minmax",
    img_norm_for_fusion: str = "minmax",
) -> pd.Series:

    preds = []
    missing_images = 0

    for _, row in tqdm(df_pred.iterrows(), total=len(df_pred), desc="MAKING MULTI-MODAL INFERENCE"):
        article = str(row.get("article_body", "") or "").strip()
        titles = [str(row.get(f"title_{i}", "") or "") for i in range(1, 11)]

        # Ensemble textual
        text_scores_raw = text_ensemble.score_titles(article, titles)
        text_scores = normalize_scores(text_scores_raw, mode=text_norm_for_fusion)

        ord_text = np.argsort(-text_scores)
        gap = float(text_scores[ord_text[0]] - text_scores[ord_text[1]])

        img_path = find_image_path(images_dir, row.get("image_hash", ""))
        if img_path is None:
            missing_images += 1
            order = np.argsort(-text_scores, kind="mergesort").tolist()
            preds.append(" ".join([f"t{i+1}" for i in order]))
            continue

#        img_scores_raw = siglip_logits_image_vs_titles(
#            clip_model=clip_model,
#            clip_processor=clip_processor,
#            device=device,
#            image_path=img_path,
#            titles=titles,
#        )

        img_scores_raw = clip_multilingual_logits_image_vs_titles(
            clip_model=clip_model,
            clip_img_model=clip_img_model,
            image_path=img_path,
            titles=titles,
        )
        
        img_scores = normalize_scores(img_scores_raw, mode=img_norm_for_fusion)

        w_img_eff = img_when_confident if gap >= gate_gap else w_img
        w_text_eff = 1.0 - w_img_eff

        final_scores = (w_text_eff * text_scores) + (w_img_eff * img_scores)

        order = np.argsort(-final_scores, kind="mergesort").tolist()
        preds.append(" ".join([f"t{i+1}" for i in order]))

    if missing_images:
        print(f"[WARN] Missing images for {missing_images} rows. Used text-only ensemble fallback.")

    return pd.Series(preds, index=df_pred.index)
            

def predict_task2_siglip_plus_text_ensemble_agreement_gated(
    df_pred: pd.DataFrame,
    images_dir,
    text_ensemble,              # instancia de TextEnsembleReranker
    clip_model,
    clip_processor,
    device: str,
    w_img: float = 0.10,
    gate_gap: float = 0.10,
    w_img_disagree: float = 0.00,
    img_when_confident: float = 0.00,
    text_norm_for_fusion: str = "minmax",
    img_norm_for_fusion: str = "minmax",
) -> pd.Series:
    """
    Fusión global texto+imagen con doble gating:
      1) por confianza del texto (gap top1-top2)
      2) por acuerdo texto-imagen en el top1
    """

    preds = []
    missing_images = 0

    for _, row in tqdm(df_pred.iterrows(), total=len(df_pred), desc="MAKING MULTI-MODAL INFERENCE"):
        article = str(row.get("article_body", "") or "").strip()
        titles = [str(row.get(f"title_{i}", "") or "") for i in range(1, 11)]

        # Scores textuales del ensemble
        text_scores_raw = text_ensemble.score_titles(article, titles)
        text_scores = normalize_scores(text_scores_raw, mode=text_norm_for_fusion)

        text_order = np.argsort(-text_scores, kind="mergesort").tolist()
        text_top1 = text_order[0]
        text_gap = float(text_scores[text_order[0]] - text_scores[text_order[1]])

        img_path = find_image_path(images_dir, row.get("image_hash", ""))
        if img_path is None:
            missing_images += 1
            preds.append(" ".join([f"t{i+1}" for i in text_order]))
            continue

        # Scores de imagen
        img_scores_raw = siglip_logits_image_vs_titles(
            clip_model=clip_model,
            clip_processor=clip_processor,
            device=device,
            image_path=img_path,
            titles=titles,
        )
        
        img_scores = normalize_scores(img_scores_raw, mode=img_norm_for_fusion)

        img_order = np.argsort(-img_scores, kind="mergesort").tolist()
        img_top1 = img_order[0]

        agree_top1 = (text_top1 == img_top1)

        # Gating
        if text_gap >= gate_gap:
            w_img_eff = img_when_confident
        else:
            w_img_eff = w_img if agree_top1 else w_img_disagree

        w_text_eff = 1.0 - w_img_eff
        final_scores = (w_text_eff * text_scores) + (w_img_eff * img_scores)

        final_order = np.argsort(-final_scores, kind="mergesort").tolist()
        preds.append(" ".join([f"t{i+1}" for i in final_order]))

    if missing_images:
        print(f"[WARN] Missing images for {missing_images} rows. Used text-only ensemble fallback.")

    return pd.Series(preds, index=df_pred.index)

  
def search_best_multimodal_params(
    df_eval: pd.DataFrame,
    images_dir: Path,
    text_ensemble: TextEnsembleReranker,
    clip_model,
    clip_processor,
    device: str,
    w_img_grid: List[float],
    gate_gap_grid: List[float],
    w_img_disagree_grid: List[float],
    img_when_confident_grid: List[float] = [0.0],
    text_norm_for_fusion: str = "minmax",
    img_norm_for_fusion: str = "minmax",
):
    best_params = None
    best_metrics = None
    best_score = -1.0
    img_when_confident = 0.00
    
    for gate_gap in gate_gap_grid:
        for w_img_disagree in w_img_disagree_grid:
            for w_img in w_img_grid:
              tmp_df = df_eval.copy()
              tmp_df["pred_task_2"] = predict_task2_siglip_plus_text_ensemble_agreement_gated(
                  df_pred=tmp_df,
                  images_dir=images_dir,
                  text_ensemble=text_ensemble,
                  clip_model=clip_model,
                  clip_processor=clip_processor,
                  device=device,
                  w_img=w_img,
                  gate_gap=gate_gap,
                  w_img_disagree=w_img_disagree,
                  img_when_confident=img_when_confident,
                  text_norm_for_fusion=text_norm_for_fusion,
                  img_norm_for_fusion=img_norm_for_fusion,
              )
  
              metrics = evaluate_ranking_predictions(tmp_df, pred_col="pred_task_2")
  
              # Priorizamos top1_acc y luego PA-nDCG
              score = metrics["top1_acc"] * 20.0 + metrics["pa_ndcg@10"] * 100.0
  
              print(
                  f"[w_img={w_img:.2f}, gate_gap={gate_gap:.2f}, w_img_disagree={w_img_disagree:.2f}] "
                  f"-> {metrics}"
              )
  
              if score > best_score:
                  best_score = score
                  best_params = {
                      "w_img": w_img,
                      "gate_gap": gate_gap,
                      "w_img_disagree": w_img_disagree,
                      "img_when_confident": img_when_confident,
                      "text_norm_for_fusion": text_norm_for_fusion,
                      "img_norm_for_fusion": img_norm_for_fusion,
                  }
                  best_metrics = metrics

    return best_params, best_metrics
    
    
    
text_model_dirs = {
    "rigoberta": "/output_political/rigoberta_listwise",
    "rigoberta2": "/output_political/rigoberta2_listwise",
    "mrbert": "/output_political/mrbertes_listwise",
    "mrbert2": "/output_political/mrbertes2_listwise",
    "roberta": "/output_political/roberta_listwise",
    "roberta2": "/output_political/mrbertes2_listwise",
}

# USE YOUR WEIGHTS FROM PREVIOUS FILE FOR TASK 1
text_weights = {
    "rigoberta": 0.225,
    "rigoberta2": 0.225,
    "mrbert": 0.225,
    "mrbert2": 0.225,
    "roberta": 0.05,
    "roberta2": 0.05,
}

text_ensemble = TextEnsembleReranker(
    model_dirs=text_model_dirs,
    weights=text_weights,
    max_length=512,
    article_chars=5000,
    normalization="zscore",
)

# Misma validación que durante entrenamiento
df_train = pd.read_csv(TRAIN_CSV)
validate_columns(df_train)
df_eval = get_validation_df(df_train, article_chars=ARTICLE_CHARS, test_size=TEST_SIZE, seed=SEED)
print(f"Validation rows: {len(df_eval)}")
    
best_params, best_metrics = search_best_multimodal_params(
    df_eval=df_eval,   # tu split de validación
    images_dir=IMAGES_DIR,
    text_ensemble=text_ensemble,
    clip_model=clip_model,
    clip_processor=clip_processor,
    device=device,
    w_img_grid=[0.1, 0.2, 0.3, 0.4],
    gate_gap_grid=[0.1, 0.15, 0.20, 0.25],
    w_img_disagree_grid = [0.00],
    img_when_confident_grid=[0.00],
    text_norm_for_fusion="zscore",
    img_norm_for_fusion="zscore",
)

print("Best params:", best_params)
print("Best metrics:", best_metrics)