## Codeforces Tag Classification Challenge

This repository contains a simple yet strong baseline for multi-label tag prediction on
Codeforces-style algorithmic problems.

### Approach

- **Features**: uses only `prob_desc_description` (problem description text), then applies a
  word + bigram TF-IDF representation.
- **Model**: `OneVsRest` logistic regression with class balancing, trained on all
  available examples.
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

```bash
python -m code_classifier.cli train \
  --data-dir ./data \
  --model-path models/focus_tags_model.joblib \
  --focus-tags-only \
  --random-state 42
```

- **Evaluate a trained model on a dataset**:

```bash
python -m code_classifier.cli evaluate \
  --data-dir ./data \
  --model-path models/focus_tags_model.joblib \
  --focus-tags-only
```

This prints a JSON version of `sklearn.metrics.classification_report`, including
per-tag precision, recall and F1, together with micro/macro averages.

- **Predict tags for a single example**:

```bash
python -m code_classifier.cli predict \
  --model-path models/focus_tags_model.joblib \
  --json-file data/sample_0.json \
  --threshold 0.5
```

The output is a JSON object listing the predicted tags.
