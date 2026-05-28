# BERT-Text-Classification

### BERT · Multinomial Naive Bayes · K-Nearest Neighbours
A comparative NLP study that builds and evaluates three end-to-end text classification pipelines on a balanced multi-class news dataset (~2,100 articles across five categories). The project answers a practical question every ML engineer faces: **when does a heavyweight transformer actually beat a classical statistical baseline — and when is it overkill?**

---

## Results at a Glance

| Model | Accuracy | Macro-F1 | AUC (macro) | Training Time |
|---|---|---|---|---|
| Fine-tuned BERT | ~94% | ~0.94 | — | Hours (GPU) |
| TF-IDF + Naive Bayes | ~98% | ~0.98 | ~0.999 | Seconds (CPU) |
| TF-IDF + ANOVA + KNN | ~96% | 0.961 | ~0.998 | Minutes (CPU) |

---

## Categories

| Label | Category |
|---|---|
| 0 | Politics |
| 1 | Sports |
| 2 | Technology |
| 3 | Entertainment |
| 4 | Business |

---

## Project Structure

```
├── Bert_Text_Classification_Project_Modeling.ipynb   # Main notebook (all 3 models)
├── df_file.csv                                        # Dataset
├── best_tfidf_nb_pipeline.pkl                         # Saved MNB model
├── knn_tfidf_text_classifier.pkl                      # Saved KNN model
├── Data_Distribution_plot.png                         # Class distribution chart
├── WordCloud.png                                      # Per-category word clouds
├── feature_selection_comparison.png                   # Chi-Squared vs ANOVA plot
├── knn_evaluation_plots.png                           # KNN confusion matrix + ROC
└── knn_per_class_f1.png                               # Per-class F1 bar chart
```

---

## Models

### 1. Fine-Tuned BERT
A custom `BERTClassifier` built in PyTorch wrapping a frozen `bert-base-uncased` backbone (110M parameters) with a trainable two-layer classification head:

```
CLS token (768d) → Dropout(0.3) → Linear(768→256) → ReLU → Linear(256→5)
```

Only ~198,000 parameters are trained (the classification head). The backbone is frozen to reduce compute and overfitting risk.

**Training setup:**
- Optimiser: AdamW (lr=1e-3, weight decay=1e-2)
- Loss: Cross-entropy with class weights + label smoothing (ε=0.1)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=2)
- Gradient clipping: max norm 1.0
- Early stopping: patience=5 epochs
- Batch size: 16, max sequence length: 512

**Test set per-class recall:**

| Class | Recall | Primary Confusion |
|---|---|---|
| Politics | 0.95 | Business (3.3%) |
| Sports | 0.97 | Politics (1.3%) |
| Technology | 0.90 | Entertainment (7.7%) |
| Entertainment | 0.96 | Business (1.8%) |
| Business | 0.93 | Politics (2.6%) |

---

### 2. TF-IDF + Multinomial Naive Bayes
A lightweight classical pipeline tuned end-to-end via 5-fold stratified `GridSearchCV`.

**Pipeline:**
```
TfidfVectorizer(ngram_range=(1,2), max_df=0.75, min_df=2) → MultinomialNB(α=0.1)
```

**Test set performance:**
- Overall accuracy: ~98%
- Macro-F1: ~0.98
- AUC (One-vs-Rest): 0.998–1.000 across all five classes
- Sports: perfect recall (1.00); Technology: lowest recall (0.96)

Despite its simplicity, this pipeline outperforms BERT — training in seconds on CPU. The reason: news categories have highly distinctive vocabularies, which is exactly what TF-IDF is designed to exploit.

---

### 3. TF-IDF + Feature Selection + KNN
A KNN pipeline with statistical feature selection, tuned via 3-fold inner stratified cross-validation.

**Pipeline:**
```
TfidfVectorizer(ngram_range=(1,2)) → SelectKBest(ANOVA, k=8000) → KNeighborsClassifier(n_neighbors=7, weights='distance', metric='cosine')
```

Two feature selectors were compared — Chi-Squared and ANOVA F-test. ANOVA achieved a higher macro-F1 CV score and was selected. ANOVA is theoretically more appropriate here because TF-IDF values are continuous (Chi-Squared was designed for count/binary features).

**Test set performance:**
- Macro-F1: 0.961
- AUC: ≥0.993 across all classes (Sports=1.000, Politics=0.999, Technology=0.997)

---

## UMAP Embedding Visualisation

Before fine-tuning, CLS-token embeddings from the frozen `bert-base-uncased` model were extracted for all articles and projected to 2D using UMAP (`n_neighbors=30`, `min_dist=0.1`, `metric='cosine'`).

Key findings:
- **Sports** forms a tight, well-separated cluster — consistent with its near-perfect recall across all models
- **Entertainment** is similarly compact
- **Technology** and **Business** partially overlap — reflecting genuine semantic ambiguity (tech company news bridges both categories), which explains their higher cross-class confusion in all three models

---

## Preprocessing

Two separate pipelines are used depending on the model family:

**BERT preprocessing** (minimal — preserves sub-word tokenisation):
- Remove URLs (`http\S+`, `www.\S+`)
- Strip HTML tags
- Collapse repeated whitespace and special characters
- Lowercase

**MNB / KNN preprocessing** (aggressive):
- All of the above, plus removing all non-alphabetic characters and punctuation
- Retains only lowercase word tokens for TF-IDF compatibility

---

## Inference API

All three models expose a predict function and were tested on five real-world news articles (Sports, Politics, Business, Entertainment, Technology) scraped from the web — all three correctly classified all five unseen articles.

```python
# BERT
bert_predict(text)

# Naive Bayes
mnb_predict(text)

# KNN — returns label, category, confidence, and per-class probabilities
knn_predict(text)
```

Example KNN output:
```
─────────────────────────────────────────────
  📰  NEWS CATEGORY PREDICTION
─────────────────────────────────────────────
  Prediction  : Sports  (label 1)
  Confidence  : 99.42%

  Class probabilities:
    Sports          0.9942  ██████████████████████████████
    Politics        0.0031  
    Business        0.0018  
    Entertainment   0.0005  
    Technology      0.0004  
─────────────────────────────────────────────
```

---

## Setup

### Requirements

```bash
pip install torch transformers scikit-learn matplotlib seaborn tqdm wordcloud ipywidgets umap-learn joblib
```

### Run

1. Place `df_file.csv` in the path referenced in the notebook (`../input/text-classification-documentation/df_file.csv`) or update the `data_path` variable.
2. Run all cells in `Bert_Text_Classification_Project_Modeling.ipynb`.
3. BERT training requires a GPU. Classical models (MNB, KNN) run on CPU.

### Reproducibility

All random seeds are fixed globally:
```python
SEED = 43
set_all_seeds(SEED)
```

---

## Key Takeaway

On a dataset with high inter-class vocabulary contrast, well-tuned TF-IDF pipelines can match or outperform transformer models at a fraction of the compute cost. BERT's advantage is expected to emerge on harder tasks — ambiguous articles, short headline-only text, or cross-domain classification where surface vocabulary is insufficient to distinguish categories.

---

## Authors

Aakrit Sharma Lamsal · Aaryan Bhosale · Sahan Shrestha  
Department of Computer Science, Georgia State University — April 2026
