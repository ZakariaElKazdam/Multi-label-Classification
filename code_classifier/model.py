#from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer


@dataclass
class TrainedModel:
    pipeline: Pipeline #modèle scikit-learn (TF-IDF + classifieur)
    mlb: MultiLabelBinarizer #convertit les tags en format binaire 
    classes_: List[str] #liste des tags possibles

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "mlb": self.mlb,
                "classes": self.classes_,
            },
            path,
        ) #sauvegarde les 3 composantes dans un dict (format binaire)

    @classmethod
    def load(cls, path: str) -> "TrainedModel":
        obj = joblib.load(path)
        return cls(
            pipeline=obj["pipeline"],
            mlb=obj["mlb"],
            classes_=list(obj["classes"]),
        )
        #mieux que créer une instance vide et la charger ensuite

    def predict(self, texts: Sequence[str], threshold: float = 0.5) -> List[List[str]]:
        """Predict labels for a batch of texts using a probability threshold."""
        # We configured a probabilistic classifier, so ``predict_proba`` is available.
        probas = self.pipeline.predict_proba(texts)
        # probas shape: (n_samples, n_classes)
        labels_bin = (probas >= threshold).astype(int)
        return self.mlb.inverse_transform(labels_bin)


def build_pipeline() -> Pipeline:
    """Create a simple TF-IDF + linear classifier pipeline."""
    base_clf = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=200,
        random_state=42,
    )
    clf = OneVsRestClassifier(base_clf)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), #Capture concepts composés
        min_df=2, #ignorer les mots très rares et fautes de frappes
        max_df=0.9, #ignorer les mots très communs ( the , a ,on ...)
        sublinear_tf=True, #améliore la stabilité numérique du modèle 1+log(tf)
    )
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def train_model(
    texts: Sequence[str],
    tag_lists: Sequence[Sequence[str]],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[TrainedModel, dict]:
    """Train the model and return it along with evaluation metrics on a hold-out split."""
    # Set random seeds for reproducibility
    np.random.seed(random_state)
    import random
    random.seed(random_state)

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(tag_lists)
    X_train, X_val, y_train, y_val = train_test_split(
        list(texts),
        Y,
        test_size=test_size,
        random_state=random_state,
        stratify=(Y.sum(axis=1) > 0).astype(int),
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    report = classification_report(
        y_val,
        y_pred,
        target_names=mlb.classes_,
        output_dict=True,
        zero_division=0,
    )

    model = TrainedModel(pipeline=pipeline, mlb=mlb, classes_=list(mlb.classes_))
    return model, report


def evaluate_model(
    model: TrainedModel,
    texts: Sequence[str],
    tag_lists: Sequence[Sequence[str]],
) -> dict:
    """Evaluate an existing model on a new dataset."""
    Y_true = model.mlb.transform(tag_lists)
    Y_pred = model.pipeline.predict(list(texts))
    report = classification_report(
        Y_true,
        Y_pred,
        target_names=model.classes_,
        output_dict=True,
        zero_division=0,
    )
    return report
