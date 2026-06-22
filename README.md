# GACOP at PoliticHeadlinES-IberLEF 2026

This repository contains the code used by the GACOP team for the **PoliticHeadlinES-IberLEF 2026** shared task on Spanish political headline reranking.

The task consists of ranking ten candidate headlines for each political news article. The repository includes code for:

- Training listwise text encoders.
- Searching the best text-only ensemble for Task 1.
- Searching the best text-image fusion strategy for Task 2.

## Repository Structure

```text
.
├── train_encoders.py          # Training script for the listwise text encoders
├── infer_task1_politicheadlines.py    # Text-only ensemble search for Task 1
├── infer_task2_politicheadlines.py   # Text-image ensemble search for Task 2
└── README.md
```

## Files

### `train_encoders.py`

This script trains the encoder-only rerankers used in the text-only system.

The model follows a listwise ranking formulation. For each article, the model receives the ten candidate headlines and produces one score per headline. The correct headline is used as the target class, and the model is trained with cross-entropy loss over the complete candidate list.

The script supports transformer encoders from Hugging Face. In the final system, we used models such as:

- `IIC/RigoBERTa-2.0`
- `BSC-LT/MrBERT-es`
- `FacebookAI/xlm-roberta-large`

Each trained model directory contains:

```text
listwise_ranker.pt
base_model.txt
tokenizer files
```

### `infer_task1_politicheadlines.py`

This script evaluates the trained text encoders and searches for the best weighted ensemble for **Task 1**.

The script:

1. Loads the trained listwise rerankers.
2. Computes one score vector per model and article.
3. Normalizes the model scores.
4. Searches for the best ensemble weights.
5. Converts the final scores into ranked headline tokens.
6. Saves the validation predictions and selected weights.

The expected ranking format is:

```text
t3 t1 t7 t2 t5 t6 t4 t8 t9 t10
```

where `t1` to `t10` refer to the ten candidate headlines of each article.

### `infer_task2_politicheadlines.py`

This script extends the text-only ensemble with visual information for **Task 2**.

The script combines:

- The text-only ensemble scores.
- Image-headline compatibility scores from a multilingual SigLIP model.

It implements an agreement-gated fusion strategy. The visual branch contributes only when:

1. The text-only ensemble is uncertain.
2. The text and image branches select the same top-ranked headline.

This conservative strategy reduces the impact of generic or weakly grounded political images.

## Data Format

The scripts expect CSV files with the following columns:

```text
id
article_body
image_hash
title_1
title_2
title_3
title_4
title_5
title_6
title_7
title_8
title_9
title_10
y_true
```

The `y_true` column is required for training and validation. It contains the gold ranking using tokens such as:

```text
t1 t4 t2 t5 t3 t6 t7 t8 t9 t10
```

For Task 2, the images must be stored in an image directory. The scripts look for files using the value of `image_hash` and common extensions such as:

```text
.jpg
.jpeg
.png
.webp
```

## Installation

Create a Python environment and install the required packages:

```bash
pip install torch transformers datasets pandas numpy scikit-learn pillow tqdm sentence-transformers
```

GPU is strongly recommended for both training and inference.

## Training Text Encoders

To train a listwise encoder, configure the model and paths inside `train_encoders.py`:

```python
MODEL_NAME = "IIC/RigoBERTa-2.0"
TRAIN_CSV = "/train_corpora/train_corpora.csv"
```

Then run:

```bash
python train_encoders.py
```

The script saves the trained reranker in the configured output directory. For example:

```text
/output_political/rigoberta_listwise
```

To train several encoders or several seeds, update `MODEL_NAME`, `SEED`, and `out_dir`.

## Task 1: Text-only Ensemble Search

After training the encoders, configure their output directories in `infer_task1_politicheadlines.py`:

```python
MODEL_DIRS = {
    "rigoberta": "/output_political/rigoberta_listwise",
    "rigoberta2": "/output_political/rigoberta2_listwise",
    "mrbert": "/output_political/mrbertes_listwise",
    "mrbert2": "/output_political/mrbertes2_listwise",
    "roberta": "/output_political/roberta_listwise",
    "roberta2": "/output_political/roberta2_listwise",
}
```

Then run:

```bash
python infer_task1_politicheadlines.py
```

The script searches for the best ensemble weights and saves:

```text
/output_political/validation_ensemble_predictions.csv
/output_political/validation_ensemble_weights.json
```

## Task 2: Text-image Fusion Search

To search the best multimodal configuration, configure the text model directories and image directory in `infer_task2_politicheadlines.py`:

```python
TRAIN_CSV = "/train_corpora/train_corpora.csv"
IMAGES_DIR = Path("/train_corpora/images")
```

Then run:

```bash
python infer_task2_politicheadlines.py
```

The script evaluates different multimodal fusion settings, including:

- Image weight.
- Text-confidence threshold.
- Agreement-gated fusion.
- Score normalization strategy.

The final configuration uses the text-only ensemble as the main signal and incorporates SigLIP scores only under conservative conditions.

## Output Format

The final submission file should contain one row per article and the following columns:

```text
id
task_1
task_2
```

Example:

```csv
id,task_1,task_2
001,t3 t1 t7 t2 t5 t6 t4 t8 t9 t10,t3 t1 t7 t2 t5 t6 t4 t8 t9 t10
```

## Metrics

The validation scripts compute several ranking metrics:

- Top-1 accuracy.
- Mean Reciprocal Rank.
- Hit@3.
- Hit@5.
- PA-nDCG@10.

The ensemble search uses a combined validation criterion based on Top-1 accuracy and PA-nDCG@10.

## Citation

If you use this repository, please cite the corresponding PoliticHeadlinES-IberLEF 2026 working notes:

```bibtex
@inproceedings{gacop-politicheadlines-2026,
  title     = {GACOP at PoliticHeadlinES-IberLEF 2026: Listwise Text Ensembles and Agreement-Gated Multimodal Fusion for Spanish Political Headline Reranking},
  author    = {García-Salmerón, Jesús and González-Férez, Pilar and Bernabé, Gregorio},
  booktitle = {Proceedings of the Iberian Languages Evaluation Forum (IberLEF 2026)},
  year      = {2026}
}
```
