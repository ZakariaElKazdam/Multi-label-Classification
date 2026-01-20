## Codeforces Tag Classification Challenge

This repository contains ( so far ) a simple yet strong baseline for multi-label tag prediction on
Codeforces problems.

### Approach

- **Features**: Flexible feature selection - can use problem description, source code, or both.
  Each feature is preprocessed separately and vectorized independently.
  When multiple features are used, they are vectorized separately and concatenated.
- **Embeddings**: Three embedding methods available, can be chosen independently for description and code:
  - **TF-IDF** (default): Word + bigram TF-IDF representation - fast and efficient
  - **Word2Vec**: Word2Vec embeddings trained on the dataset (average of word vectors) - moderate speed
  - **CodeBERT**: Pre-trained transformer model optimized for code - high quality but slow on CPU
- **Model**: `OneVsRest` logistic regression with class balancing.
- **Evaluation**: Uses train/validation/test split (default: 70/15/15) to properly evaluate models during development and preserve test set for final evaluation.
- **Targets**: by default the classifier can learn all tags present in the dataset;
  you can optionally restrict supervision to the 8 focus tags:
  `['math', 'graphs', 'strings', 'number theory', 'trees', 'geometry', 'games', 'probabilities']`.

**Note on CodeBERT**: CodeBERT is optimized for code, not natural language. While it can be used for descriptions, TF-IDF or Word2Vec are recommended for text. CodeBERT is significantly slower on CPU (5-10 minutes for ~2000 examples) compared to TF-IDF (1-2 seconds) or Word2Vec (10-30 seconds). Consider using CodeBERT only for code embeddings when GPU is available.

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Optional dependencies**:
- For Word2Vec: `gensim` (included in requirements.txt)
- For CodeBERT: `transformers` and `torch` (included in requirements.txt)
  - CodeBERT will download ~499MB model on first use
  - GPU recommended for faster inference (CUDA-compatible PyTorch)

### CLI Usage

All commands are run from the repository root.

- **Train a model**:

**With TF-IDF (default, fastest):**
```bash
python -m code_classifier.cli train \
  --data-dir ./data \
  --model-path models/tfidf_model.joblib \
  --features description,code \
  --embedding-desc tfidf \
  --embedding-code tfidf \
  --focus-tags-only
```

**With Word2Vec:**
```bash
python -m code_classifier.cli train \
  --data-dir ./data \
  --model-path models/word2vec_model.joblib \
  --features description \
  --embedding-desc word2vec \
  --word2vec-vector-size 100 \
  --word2vec-min-count 2 \
  --word2vec-epochs 10 \
  --focus-tags-only
```

**With CodeBERT for code (TF-IDF for description):**
```bash
python -m code_classifier.cli train \
  --data-dir ./data \
  --model-path models/codebert_model.joblib \
  --features description,code \
  --embedding-desc tfidf \
  --embedding-code codebert \
  --focus-tags-only
```

**Arguments:**
- `--features`: Comma-separated list of features to use
  - `description` - use only problem description
  - `code` - use only source code
  - `description,code` - use both (vectorized separately then concatenated)
- `--embedding-desc`: Embedding method for description (default: `tfidf`)
  - `tfidf` - TF-IDF vectorization (word + bigram) - recommended
  - `word2vec` - Word2Vec embeddings trained on dataset
  - `codebert` - CodeBERT (not recommended for text, will show warning)
- `--embedding-code`: Embedding method for code (default: `tfidf`)
  - `tfidf` - TF-IDF vectorization (word + bigram)
  - `word2vec` - Word2Vec embeddings trained on dataset
  - `codebert` - Pre-trained CodeBERT model (slow on CPU, GPU recommended)
- `--word2vec-vector-size`: Dimension of Word2Vec embeddings (default: 100)
- `--word2vec-min-count`: Minimum word count for Word2Vec (default: 2)
- `--word2vec-epochs`: Number of training epochs for Word2Vec (default: 10)

This will:
- Split the data into train/validation/test sets (default: 70/15/15)
- Apply preprocessing (remove latex delimiters, normalize whitespace)
- Filter to focus tags if `--focus-tags-only` is set, and remove examples without tags
- Train the model on the training set
- Evaluate on the validation set
- Save the model and a detailed report in `reports/report_YYYY_MM_DD_HH_MM_SS.json`

The report contains configuration (features, embedding, classifier parameters), validation metrics, and the list of test set IDs (preserved for final evaluation after model comparison).

- **Predict tags**:

For a single file (works with both TF-IDF and Word2Vec models):
```bash
python -m code_classifier.cli predict \
  --model-path models/tfidf_model.joblib \
  --json-file data/sample_0.json \
  --threshold 0.5
```

For an external dataset (works with both TF-IDF and Word2Vec models):
```bash
python -m code_classifier.cli predict \
  --model-path models/word2vec_model.joblib \
  --data-dir path/to/external/dataset \
  --threshold 0.5
```

The output is a JSON object with predicted tags. The same `predict` command works for all model types (TF-IDF, Word2Vec, CodeBERT) - the model type is automatically detected from the saved model file.

**Performance Notes**:
- **TF-IDF**: Fastest option (~1-2 seconds for 2000 examples), recommended for quick iterations
- **Word2Vec**: Moderate speed (~10-30 seconds for 2000 examples), good balance of quality and speed
- **CodeBERT**: Slowest on CPU (~5-10 minutes for 2000 examples), but highest quality embeddings. GPU recommended for practical use. Note: The model is loaded from scratch each time (no caching), so first use will download ~499MB model.
