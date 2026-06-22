from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


# =========================
# Config
# =========================
SEED = 42
TRAIN_CSV = "/train_corpora/train_corpora.csv"
TEST_SIZE = 0.20

TITLE_COLS = [f"title_{i}" for i in range(1, 11)]
TOKENS_ALL = [f"t{i}" for i in range(1, 11)]
N_COLS = 10
ALPHA = 0.9

# Misma configuración de inferencia que en tus runs
MAX_LENGTH = 512
ARTICLE_CHARS = 5000

# Directorios de modelos entrenados
MODEL_DIRS = {
    "rigoberta": "/output_political/rigoberta_listwise",
    "rigoberta2": "/output_political/rigoberta2_listwise",
    "mrbert": "/output_political/mrbertes_listwise",
    "mrbert2": "/output_political/mrbertes2_listwise",
    "roberta": "/output_political/roberta_listwise",
    "roberta2": "/output_political/roberta2_listwise",
}

# Pesos iniciales. Luego se optimizan por grid search.
INITIAL_WEIGHTS = {
    "rigoberta": 0.30,
    "rigoberta2": 0.30,
    "mrbert": 0.10,
    "mrbert2": 0.10,
    "roberta": 0.10,
    "roberta2": 0.10
}

# Si quieres saltarte la búsqueda y usar solo INITIAL_WEIGHTS, pon esto a False
RUN_WEIGHT_SEARCH = True
WEIGHT_GRID_STEP = 0.05

# Normalización de scores por noticia: "zscore", "minmax" o None
SCORE_NORMALIZATION = "zscore"


# =========================
# Utilidades generales
# =========================
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
        article = str(row.get("article_body", "") or "")[:article_chars]
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


# =========================
# Normalización de scores
# =========================
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
class BetoListwiseRanker(nn.Module):
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

        self.model = BetoListwiseRanker(model_name=base_model)
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


# =========================
# Inferencia por modelo y ensemble
# =========================
def predict_scores_for_model(df_pred: pd.DataFrame, model_dir: str, max_length: int = 512, article_chars: int = 5000) -> np.ndarray:
    reranker = FullReranker(model_dir=model_dir, max_length=max_length, article_chars=article_chars)
    all_scores = []

    for _, row in tqdm(df_pred.iterrows(),desc="MAKING INFERENCE"):
        article = str(row.get("article_body", "") or "")
        titles = [str(row.get(f"title_{i}", "") or "") for i in range(1, 11)]
        scores = reranker.score_titles(article, titles)
        all_scores.append(scores)

    return np.stack(all_scores, axis=0)  # (N,10)


def combine_weighted_scores(score_dict: Dict[str, np.ndarray], weights: Dict[str, float], normalization: Optional[str] = "zscore") -> np.ndarray:
    model_names = list(score_dict.keys())
    n_samples = next(iter(score_dict.values())).shape[0]
    combined = []

    for i in range(n_samples):
        acc = np.zeros(10, dtype=np.float32)
        for name in model_names:
            s = score_dict[name][i]
            s = normalize_scores(s, mode=normalization)
            acc += float(weights[name]) * s
        combined.append(acc)

    return np.stack(combined, axis=0)


def scores_to_rank_strings(score_matrix: np.ndarray) -> List[str]:
    preds = []
    for scores in score_matrix:
        order = np.argsort(-scores, kind="mergesort").tolist()
        preds.append(" ".join([f"t{i+1}" for i in order]))
    return preds


def search_best_weights(
    df_eval: pd.DataFrame,
    score_dict: Dict[str, np.ndarray],
    step: float = 0.05,
    normalization: Optional[str] = "zscore",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    names = list(score_dict.keys())
    if len(names) != 3:
        raise ValueError("La búsqueda implementada espera exactamente 3 modelos.")

    best_weights = None
    best_metrics = None
    best_score = -1.0

    grid = np.arange(0.0, 1.0 + 1e-9, step)
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9:
                continue
            if w3 < 0:
                w3 = 0.0
            weights = {names[0]: float(w1), names[1]: float(w2), names[2]: float(w3)}
            if abs(sum(weights.values()) - 1.0) > 1e-6:
                continue

            combined = combine_weighted_scores(score_dict, weights, normalization=normalization)
            tmp_df = df_eval.copy()
            tmp_df["pred_task_1"] = scores_to_rank_strings(combined)
            metrics = evaluate_ranking_predictions(tmp_df, pred_col="pred_task_1")

            score = metrics["top1_acc"] * 20.0 + metrics["pa_ndcg@10"] * 100.0
            if score > best_score:
                best_score = score
                best_weights = weights
                best_metrics = metrics

    if best_weights is None or best_metrics is None:
        raise RuntimeError("No se pudo encontrar una combinación válida de pesos.")

    return best_weights, best_metrics

def search_best_weights_local(
    df_eval,
    score_dict,
    base_weights,
    step=0.025,
    radius=0.10,
    normalization="zscore",
):
    names = list(score_dict.keys())
    n_models = len(names)

    if n_models not in (2, 3, 4, 5, 6):
        raise ValueError(f"Solo se soportan 2, 3, 4, 5 o 6 modelos, pero llegaron {n_models}")

    best_weights = None
    best_metrics = None
    best_score = -1.0

    if n_models == 2:
        w0 = base_weights[names[0]]
        grid0 = np.arange(max(0.0, w0 - radius), min(1.0, w0 + radius) + 1e-9, step)

        for a in grid0:
            b = 1.0 - a
            if b < 0.0 or b > 1.0:
                continue

            weights = {
                names[0]: float(a),
                names[1]: float(b),
            }

            combined = combine_weighted_scores(score_dict, weights, normalization=normalization)
            tmp_df = df_eval.copy()
            tmp_df["pred_task_1"] = scores_to_rank_strings(combined)
            metrics = evaluate_ranking_predictions(tmp_df, pred_col="pred_task_1")

            score = metrics["top1_acc"] * 20.0 + metrics["pa_ndcg@10"] * 100.0
            if score > best_score:
                best_score = score
                best_weights = weights
                best_metrics = metrics

    elif n_models == 3:
        w0 = base_weights[names[0]]
        w1 = base_weights[names[1]]

        grid0 = np.arange(max(0.0, w0 - radius), min(1.0, w0 + radius) + 1e-9, step)
        grid1 = np.arange(max(0.0, w1 - radius), min(1.0, w1 + radius) + 1e-9, step)

        for a in grid0:
            for b in grid1:
                c = 1.0 - a - b
                if c < 0.0 or c > 1.0:
                    continue

                weights = {
                    names[0]: float(a),
                    names[1]: float(b),
                    names[2]: float(c),
                }

                combined = combine_weighted_scores(score_dict, weights, normalization=normalization)
                tmp_df = df_eval.copy()
                tmp_df["pred_task_1"] = scores_to_rank_strings(combined)
                metrics = evaluate_ranking_predictions(tmp_df, pred_col="pred_task_1")

                score = metrics["top1_acc"] * 20.0 + metrics["pa_ndcg@10"] * 100.0
                if score > best_score:
                    best_score = score
                    best_weights = weights
                    best_metrics = metrics

    elif n_models == 4:
        w0 = base_weights[names[0]]
        w1 = base_weights[names[1]]
        w2 = base_weights[names[2]]

        grid0 = np.arange(max(0.0, w0 - radius), min(1.0, w0 + radius) + 1e-9, step)
        grid1 = np.arange(max(0.0, w1 - radius), min(1.0, w1 + radius) + 1e-9, step)
        grid2 = np.arange(max(0.0, w2 - radius), min(1.0, w2 + radius) + 1e-9, step)

        for a in grid0:
            for b in grid1:
                for c in grid2:
                    d = 1.0 - a - b - c
                    if d < 0.0 or d > 1.0:
                        continue

                    weights = {
                        names[0]: float(a),
                        names[1]: float(b),
                        names[2]: float(c),
                        names[3]: float(d),
                    }

                    combined = combine_weighted_scores(score_dict, weights, normalization=normalization)
                    tmp_df = df_eval.copy()
                    tmp_df["pred_task_1"] = scores_to_rank_strings(combined)
                    metrics = evaluate_ranking_predictions(tmp_df, pred_col="pred_task_1")

                    score = metrics["top1_acc"] * 20.0 + metrics["pa_ndcg@10"] * 100.0
                    if score > best_score:
                        best_score = score
                        best_weights = weights
                        best_metrics = metrics

    elif n_models == 5:
        w0 = base_weights[names[0]]
        w1 = base_weights[names[1]]
        w2 = base_weights[names[2]]
        w3 = base_weights[names[3]]

        grid0 = np.arange(max(0.0, w0 - radius), min(1.0, w0 + radius) + 1e-9, step)
        grid1 = np.arange(max(0.0, w1 - radius), min(1.0, w1 + radius) + 1e-9, step)
        grid2 = np.arange(max(0.0, w2 - radius), min(1.0, w2 + radius) + 1e-9, step)
        grid3 = np.arange(max(0.0, w3 - radius), min(1.0, w3 + radius) + 1e-9, step)

        for a in grid0:
            for b in grid1:
                for c in grid2:
                    for d in grid3:
                        e = 1.0 - a - b - c - d
                        if e < 0.0 or e > 1.0:
                            continue

                        weights = {
                            names[0]: float(a),
                            names[1]: float(b),
                            names[2]: float(c),
                            names[3]: float(d),
                            names[4]: float(e),
                        }

                        combined = combine_weighted_scores(score_dict, weights, normalization=normalization)
                        tmp_df = df_eval.copy()
                        tmp_df["pred_task_1"] = scores_to_rank_strings(combined)
                        metrics = evaluate_ranking_predictions(tmp_df, pred_col="pred_task_1")

                        score = metrics["top1_acc"] * 20.0 + metrics["pa_ndcg@10"] * 100.0
                        if score > best_score:
                            best_score = score
                            best_weights = weights
                            best_metrics = metrics

    elif n_models == 6:
        w0 = base_weights[names[0]]
        w1 = base_weights[names[1]]
        w2 = base_weights[names[2]]
        w3 = base_weights[names[3]]
        w4 = base_weights[names[4]]

        grid0 = np.arange(max(0.0, w0 - radius), min(1.0, w0 + radius) + 1e-9, step)
        grid1 = np.arange(max(0.0, w1 - radius), min(1.0, w1 + radius) + 1e-9, step)
        grid2 = np.arange(max(0.0, w2 - radius), min(1.0, w2 + radius) + 1e-9, step)
        grid3 = np.arange(max(0.0, w3 - radius), min(1.0, w3 + radius) + 1e-9, step)
        grid4 = np.arange(max(0.0, w4 - radius), min(1.0, w4 + radius) + 1e-9, step)

        for a in grid0:
            for b in grid1:
                for c in grid2:
                    for d in grid3:
                        for e in grid4:
                            f = 1.0 - a - b - c - d - e
                            if f < 0.0 or f > 1.0:
                                continue

                            weights = {
                                names[0]: float(a),
                                names[1]: float(b),
                                names[2]: float(c),
                                names[3]: float(d),
                                names[4]: float(e),
                                names[5]: float(f),
                            }

                            combined = combine_weighted_scores(score_dict, weights, normalization=normalization)
                            tmp_df = df_eval.copy()
                            tmp_df["pred_task_1"] = scores_to_rank_strings(combined)
                            metrics = evaluate_ranking_predictions(tmp_df, pred_col="pred_task_1")

                            score = metrics["top1_acc"] * 20.0 + metrics["pa_ndcg@10"] * 100.0
                            if score > best_score:
                                best_score = score
                                best_weights = weights
                                best_metrics = metrics

    return best_weights, best_metrics
# =========================
# Main
# =========================
def main() -> None:
    df_train = pd.read_csv(TRAIN_CSV)
    validate_columns(df_train)

    # Misma validación que durante entrenamiento
    df_eval = get_validation_df(df_train, article_chars=ARTICLE_CHARS, test_size=TEST_SIZE, seed=SEED)
    ### PARA TODO TRAIN
    df_eval = df_train.copy()
    print(f"Validation rows: {len(df_eval)}")
    
    # Cargar scores individuales
    score_dict: Dict[str, np.ndarray] = {}
    individual_metrics: Dict[str, Dict[str, float]] = {}

    for name, model_dir in MODEL_DIRS.items():
        print(f"\nCargando modelo: {name} -> {model_dir}")
        scores = predict_scores_for_model(
            df_eval,
            model_dir=model_dir,
            max_length=MAX_LENGTH,
            article_chars=ARTICLE_CHARS,
        )
        score_dict[name] = scores

        tmp_df = df_eval.copy()
        tmp_df["pred_task_1"] = scores_to_rank_strings(scores)
        metrics = evaluate_ranking_predictions(tmp_df, pred_col="pred_task_1")
        individual_metrics[name] = metrics
        print(f"Métricas {name}: {metrics}")

    print("\n=== Resumen individual ===")
    for name, metrics in individual_metrics.items():
        print(name, metrics)

    # Ensemble
    if RUN_WEIGHT_SEARCH:
        best_weights, best_metrics = search_best_weights_local(
            df_eval=df_eval,
            score_dict=score_dict,
            base_weights=INITIAL_WEIGHTS,
            step=0.025,
            radius=0.10,
            normalization="zscore",
        )
        print("\n=== Mejor ensemble encontrado ===")
        print("Pesos:", best_weights)
        print("Métricas:", best_metrics)
        final_weights = best_weights
    else:
        final_weights = INITIAL_WEIGHTS
        combined = combine_weighted_scores(score_dict, final_weights, normalization=SCORE_NORMALIZATION)
        tmp_df = df_eval.copy()
        tmp_df["pred_task_1"] = scores_to_rank_strings(combined)
        best_metrics = evaluate_ranking_predictions(tmp_df, pred_col="pred_task_1")
        print("\n=== Ensemble con pesos fijos ===")
        print("Pesos:", final_weights)
        print("Métricas:", best_metrics)

    # Guardar predicciones finales del ensemble
    combined = combine_weighted_scores(score_dict, final_weights, normalization=SCORE_NORMALIZATION)
    df_eval["pred_task_1"] = scores_to_rank_strings(combined)
    df_eval.to_csv("/output_political/validation_ensemble_predictions.csv", index=False)

    with open("/output_political/validation_ensemble_weights.json", "w", encoding="utf-8") as f:
        json.dump({
            "weights": final_weights,
            "metrics": best_metrics,
            "normalization": SCORE_NORMALIZATION,
        }, f, ensure_ascii=False, indent=2)

    print("\nGuardado:")
    print("- /output_political/validation_ensemble_predictions.csv")
    print("- /output_political/validation_ensemble_weights.json")


if __name__ == "__main__":
    main()
