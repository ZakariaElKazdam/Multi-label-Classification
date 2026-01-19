## Codeforces Tag Classification Challenge

This repository contains ( so far ) a simple yet strong baseline for multi-label tag prediction on
Codeforces problems.

### Approach

- **Features**: Flexible feature selection - can use problem description, source code, or both.
  Each feature is preprocessed separately and vectorized independently.
  When multiple features are used, they are vectorized separately and concatenated.
- **Embeddings**: Two embedding methods available:
  - **TF-IDF** (default): Word + bigram TF-IDF representation
  - **Word2Vec**: Word2Vec embeddings trained on the dataset (average of word vectors)
- **Model**: `OneVsRest` logistic regression with class balancing.
- **Evaluation**: Uses train/validation/test split (default: 70/15/15) to properly evaluate models during development and preserve test set for final evaluation.
- **Targets**: by default the classifier can learn all tags present in the dataset;
  you can optionally restrict supervision to the 8 focus tags:
  `['math', 'graphs', 'strings', 'number theory', 'trees', 'geometry', 'games', 'probabilities']`.

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### CLI Usage

All commands are run from the repository root.

- **Train a model**:

**With TF-IDF:**
```bash
python -m code_classifier.cli train \
  --data-dir ./data \
  --model-path models/tfidf_model.joblib \
  --features description,code \
  --embedding tfidf \
  --focus-tags-only \
  --train-size 0.7 \
  --val-size 0.15 \
  --random-state 42
```

**With Word2Vec:**
```bash
python -m code_classifier.cli train \
  --data-dir ./data \
  --model-path models/word2vec_model.joblib \
  --features description \
  --embedding word2vec \
  --focus-tags-only \
  --word2vec-vector-size 100 \
  --word2vec-min-count 2 \
  --word2vec-epochs 10 \
  --train-size 0.7 \
  --val-size 0.15 \
  --random-state 42
```

**Arguments:**
- `--features`: Comma-separated list of features to use
  - `description` - use only problem description
  - `code` - use only source code
  - `description,code` - use both (vectorized separately then concatenated)
- `--embedding`: Embedding method to use (default: `tfidf`)
  - `tfidf` - TF-IDF vectorization (word + bigram)
  - `word2vec` - Word2Vec embeddings trained on dataset
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

The output is a JSON object with predicted tags. The same `predict` command works for both TF-IDF and Word2Vec models - the model type is automatically detected from the saved model file.
