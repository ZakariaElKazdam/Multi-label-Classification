## Codeforces Tag Classification Challenge

This repository contains ( so far ) a simple yet strong baseline for multi-label tag prediction on
Codeforces problems.

### Approach

- **Features**: Flexible feature selection - can use problem description, source code, or both.
  Each feature is preprocessed separately and vectorized with TF-IDF .
  When multiple features are used, they are vectorized separately and concatenated.
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

```bash
python -m code_classifier.cli train \
  --data-dir ./data \
  --model-path models/focus_tags_model.joblib \
  --features description,code \
  --focus-tags-only \
  --train-size 0.7 \
  --val-size 0.15 \
  --random-state 42
```

The `--features` argument accepts:
- `description` - use only problem description
- `code` - use only source code
- `description,code` - use both (vectorized separately then concatenated)

This will:
- Split the data into train/validation/test sets (default: 70/15/15)
- Apply preprocessing (remove latex delimiters, normalize whitespace)
- Filter to focus tags if `--focus-tags-only` is set, and remove examples without tags
- Train the model on the training set
- Evaluate on the validation set
- Save the model and a detailed report in `reports/report_YYYY_MM_DD_HH_MM_SS.json`

The report contains configuration (features, embedding, classifier parameters), validation metrics, and the list of test set IDs (preserved for final evaluation after model comparison).

- **Predict tags**:

For a single file:
```bash
python -m code_classifier.cli predict \
  --model-path models/focus_tags_model.joblib \
  --json-file data/sample_0.json \
  --threshold 0.5
```

For an external dataset:
```bash
python -m code_classifier.cli predict \
  --model-path models/focus_tags_model.joblib \
  --data-dir path/to/external/dataset \
  --threshold 0.5
```

The output is a JSON object with predicted tags.
