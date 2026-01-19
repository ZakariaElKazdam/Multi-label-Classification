#from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import hstack, csr_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from gensim.models import Word2Vec as W2V


@dataclass
class TrainedModel:
    vectorizers: dict  # Embedding vectorizers for each feature
    clf: OneVsRestClassifier  # classifier
    features: List[str]  # list of features used
    embedding_type: str  # type of embedding: "tfidf" or "word2vec"
    mlb: MultiLabelBinarizer  # convertit les tags en format binaire 
    classes_: List[str]  # liste des tags 

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "vectorizers": self.vectorizers,
                "clf": self.clf,
                "features": self.features,
                "embedding_type": self.embedding_type,
                "mlb": self.mlb,
                "classes": self.classes_,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "TrainedModel":
        obj = joblib.load(path)
        return cls(
            vectorizers=obj["vectorizers"],
            clf=obj["clf"],
            features=obj.get("features", ["description"]),
            embedding_type=obj.get("embedding_type", "tfidf"),
            mlb=obj["mlb"],
            classes_=obj["classes"],
        )

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> List[List[str]]:
        """Predict labels for a batch of examples.
        
        Args:
            X: DataFrame with columns text_description and/or text_code
            threshold: Probability threshold for predictions
        """
        # Vectorize each feature
        X_list = []
        if "description" in self.features:
            X_desc = self.vectorizers["description"].transform(
                X["text_description"].tolist()
            )
            X_list.append(X_desc)
        if "code" in self.features:
            X_code = self.vectorizers["code"].transform(
                X["text_code"].tolist()
            )
            X_list.append(X_code)

        # Combine embeddings
        if len(X_list) == 1:
            # Single feature: nothing to concatenate
            X_combined = X_list[0]
        else:
            # All embeddings ont été produits avec le même type d'embeddding
            first = X_list[0]
            if isinstance(first, np.ndarray):
                # Word2Vec: toutes les matrices sont denses
                X_combined = np.hstack(X_list)
            else:
                # TF-IDF: toutes les matrices sont sparse
                X_combined = hstack(X_list)
        
        probas = self.clf.predict_proba(X_combined)
        labels_bin = (probas >= threshold).astype(int)
        return self.mlb.inverse_transform(labels_bin)


class Word2VecVectorizer:
    """Simple Word2Vec vectorizer - trains on data."""
    
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 2, epochs: int = 10):
        try:
            self.Word2Vec = W2V
        except ImportError:
            raise ImportError("gensim package is required. Install with: pip install gensim")
        
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model = None
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Train Word2Vec on texts and transform them."""
        # Tokenize texts (simple split, can be improved !!!!!)
        sentences = [text.split() for text in texts]
        
        # Train Word2Vec model sur sentences
        self.model = self.Word2Vec(
            sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=1,
            sg=1,  # skip-gram
            epochs=self.epochs,
        )
        
        return self.transform(texts)
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to Word2Vec vectors (average of word vectors)."""
        if self.model is None:
            raise ValueError("Word2Vec model must be fitted first")
        
        vectors = []
        for text in texts:
            words = text.split()
            if words:
                word_vecs = []
                for word in words:
                    if word in self.model.wv: 
                        #check if the word exist in the vocabulary already, we skip it if it doesn't ( too rare to be useful )
                        word_vecs.append(self.model.wv[word])
                if word_vecs:
                    vectors.append(np.mean(word_vecs, axis=0))
                else:
                    vectors.append(np.zeros(self.vector_size))
            else:
                vectors.append(np.zeros(self.vector_size))
        return np.array(vectors)


def build_pipeline(features: List[str] = None, embedding_type: str = "tfidf", word2vec_vector_size: int = 100, word2vec_min_count: int = 2, word2vec_epochs: int = 10):
    """
    Create embedding vectorizers and classifier.
    
    Args:
        features: List of features to use. Options: ["description"], ["code"], or ["description", "code"]
                  Default: ["description"]
        embedding_type: Type of embedding ("tfidf" or "word2vec")
        word2vec_vector_size: Dimension of Word2Vec embeddings (default: 100)
        word2vec_min_count: Minimum word count for Word2Vec (default: 2)
        word2vec_epochs: Number of training epochs for Word2Vec (default: 10)
    
    Returns:
        Tuple of (vectorizers_dict, classifier)
    """
    
    base_clf = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=250,
        random_state=42,
    )
    clf = OneVsRestClassifier(base_clf)
    
    vectorizers = {}
    
    if embedding_type == "tfidf":
        # TF-IDF parameters
        tfidf_params = {
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.9,
            "sublinear_tf": True,
        }
        
        if "description" in features:
            vectorizers["description"] = TfidfVectorizer(**tfidf_params)
        if "code" in features:
            vectorizers["code"] = TfidfVectorizer(**tfidf_params)
    
    elif embedding_type == "word2vec":
        # Create separate Word2Vec vectorizers for each feature
        if "description" in features:
            vectorizers["description"] = Word2VecVectorizer(vector_size=word2vec_vector_size, min_count=word2vec_min_count, epochs=word2vec_epochs)
        if "code" in features:
            vectorizers["code"] = Word2VecVectorizer(vector_size=word2vec_vector_size, min_count=word2vec_min_count, epochs=word2vec_epochs)
    
    else:
        raise ValueError(f"Unknown embedding_type: {embedding_type}. Supported: 'tfidf', 'word2vec'")
    
    return vectorizers, clf


def train_model(
    df: pd.DataFrame,
    features: List[str],
    embedding_type: str = "tfidf",
    word2vec_vector_size: int = 100,
    word2vec_min_count: int = 2,
    word2vec_epochs: int = 10,
    train_size: float = 0.7,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[TrainedModel, dict, List[str]]:
    """
    Train the model with train/val/test split and return validation metrics and test set IDs.
    
    Args:
        df: DataFrame with columns: text_description, text_code, tags, problem_id
        features: List of features to use (e.g., ["description"], ["code"], or ["description", "code"])
        train_size: Fraction of data for training (default: 0.7)
        val_size: Fraction of data for validation (default: 0.15)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (model, validation_report, test_ids)
        test_ids contains the problem IDs of the test set (preserved for later evaluation)
    """
    # Validate split sizes
    test_size = 1.0 - train_size - val_size
    if test_size <= 0:
        raise ValueError(f"train_size + val_size must be < 1.0, got {train_size} + {val_size} = {train_size + val_size}")

    # Extract tags and IDs
    tag_lists = df["tags"].tolist()

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(tag_lists)
    
    # Create indices array to track splits
    indices = np.arange(len(df))
    
    # First split: train vs (val+test)
    temp_size = val_size + test_size
    df_train, df_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        df,
        Y,
        indices,
        test_size=temp_size,
        random_state=random_state,
        stratify=(Y.sum(axis=1) > 0).astype(int),
    )
    
    # Second split: val vs test
    val_ratio = val_size / temp_size
    df_val, df_test, y_val, y_test, idx_val, idx_test = train_test_split(
        df_temp,
        y_temp,
        idx_temp,
        test_size=1.0 - val_ratio,
        random_state=random_state,
        stratify=(y_temp.sum(axis=1) > 0).astype(int),
    )
    
    # Extract test IDs using indices
    ids_test = []

    ids_list = df["problem_id"].tolist()
    for idx in idx_test:
        ids_test.append(ids_list[idx])

    # building the pipeline
    vectorizers, clf = build_pipeline(features=features, embedding_type=embedding_type, word2vec_vector_size=word2vec_vector_size, word2vec_min_count=word2vec_min_count, word2vec_epochs=word2vec_epochs)
    
    # Vectorize each feature separately
    X_train_list = []
    X_val_list = []
    
    if "description" in features:
        X_train_desc = vectorizers["description"].fit_transform(df_train["text_description"].tolist())
        X_val_desc = vectorizers["description"].transform(df_val["text_description"].tolist())
        X_train_list.append(X_train_desc)
        X_val_list.append(X_val_desc)
    
    if "code" in features:
        X_train_code = vectorizers["code"].fit_transform(df_train["text_code"].tolist())
        X_val_code = vectorizers["code"].transform(df_val["text_code"].tolist())
        X_train_list.append(X_train_code)
        X_val_list.append(X_val_code)
    
    # Concatenate features (handle both sparse and dense)
    if len(X_train_list) == 1:
        X_train = X_train_list[0]
        X_val = X_val_list[0]
    else:
        # Toutes les features utilisent le même type d'embedding
        first = X_train_list[0]
        if isinstance(first, np.ndarray):
            # Word2Vec : tout est dense
            X_train = np.hstack(X_train_list)
            X_val = np.hstack(X_val_list)
        else:
            # TF-IDF : tout est sparse
            X_train = hstack(X_train_list)
            X_val = hstack(X_val_list)
    
    # Train classifier
    clf.fit(X_train, y_train)
    
    # Evaluate only on validation set (test set is preserved for later)
    y_val_pred = clf.predict(X_val)
    val_report = classification_report(
        y_val,
        y_val_pred,
        target_names=mlb.classes_,
        output_dict=True,
        zero_division=0,
    )

    model = TrainedModel(
        vectorizers=vectorizers,
        clf=clf,
        features=features,
        embedding_type=embedding_type,
        mlb=mlb,
        classes_=list(mlb.classes_)
    )
    return model, val_report, ids_test


def evaluate_model(
    model: TrainedModel,
    texts: Sequence[str],
    ground_truth: Sequence[Sequence[str]],
) -> dict:
    """Evaluate an existing model on a new dataset."""
    Y_true = model.mlb.transform(ground_truth)
    Y_pred = model.pipeline.predict(list(texts))
    report = classification_report(
        Y_true,
        Y_pred,
        target_names=model.classes_,
        output_dict=True,
        zero_division=0,
    )
    return report
