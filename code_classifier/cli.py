import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict

import pandas as pd

from .data import load_dataset_as_dataframe, load_json_file
from .preprocessing import (
    DEFAULT_FOCUS_TAGS,
    filter_to_focus_tags,
    remove_empty_tags,
    preprocess_description,
    preprocess_code,
)

from .model import TrainedModel, train_model


def cmd_train(args: argparse.Namespace) -> None:
    # Parse features
    features = args.features.split(",") if args.features else ["description"]
    features = [f.strip() for f in features]
    
    # Validate features
    valid_features = {"description", "code"}
    for feat in features:
        if feat not in valid_features:
            raise ValueError(f"Invalid feature: {feat}. Must be one of {valid_features}")
    
    # Load data with requested features
    df = load_dataset_as_dataframe(args.data_dir, features=features)
    
    # Apply preprocessing
    if "description" in features:
        df["text_description"] = df["text_description"].apply(preprocess_description)
    if "code" in features:
        df["text_code"] = df["text_code"].apply(preprocess_code)
    
    # Filter to focus tags if requested
    if args.focus_tags_only:
        df = filter_to_focus_tags(df, DEFAULT_FOCUS_TAGS)
        # Remove examples without tags (after filtering)
        df = remove_empty_tags(df)

    #all the arguments below have a default value
    random_state = args.random_state 
    train_size = args.train_size
    val_size = args.val_size

    embedding_type = getattr(args, "embedding", "tfidf")
    word2vec_vector_size = getattr(args, "word2vec_vector_size", 100)
    word2vec_min_count = getattr(args, "word2vec_min_count", 2)
    word2vec_epochs = getattr(args, "word2vec_epochs", 10)
    
    model, val_report, test_ids = train_model(
        df=df,
        features=features,
        embedding_type=embedding_type,
        word2vec_vector_size=word2vec_vector_size,
        word2vec_min_count=word2vec_min_count,
        word2vec_epochs=word2vec_epochs,
        train_size=train_size,
        val_size=val_size,
        random_state=random_state,
    )
    
    # Save model
    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    model.save(args.model_path)

    # Create report dictionary
    test_size = 1.0 - train_size - val_size
    if embedding_type == "tfidf":
        embedding_params = {
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.9,
            "sublinear_tf": True,
        }
    else:
        embedding_params = {
            "vector_size": word2vec_vector_size,
            "window": 5,
            "min_count": word2vec_min_count,
            "epochs": word2vec_epochs,
        }
    
    report_data = {
        "features": features,
        "embedding": embedding_type,
        "embedding_params": embedding_params,
        "classifier": "LogisticRegression",
        "classifier_params": {
            "solver": "liblinear",
            "class_weight": "balanced",
            "max_iter": 250,
            "strategy": "OneVsRest",
        },
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "random_state": random_state,
        "focus_tags_only": args.focus_tags_only,
        "validation_metrics": val_report,
        "test_set_ids": test_ids,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save report to reports/ directory
    os.makedirs("reports", exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    report_filename = f"reports/report_{timestamp_str}.json"
    
    with open(report_filename, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    
    print(f"Model saved to: {args.model_path}")
    print(f"Report saved to: {report_filename}")
    print(f"Test set contains {len(test_ids)} examples")
    print("\n=== Validation Metrics ===")
    print(json.dumps(val_report, indent=2))




def cmd_predict(args: argparse.Namespace) -> None:
    model = TrainedModel.load(args.model_path)
    
    if args.json_file and args.data_dir:
        raise ValueError("Cannot specify both --json-file and --data-dir. Choose one.")
    if not args.json_file and not args.data_dir:
        raise ValueError("Must specify either --json-file or --data-dir.")
    
    if args.json_file:
        # Single file mode
        record = load_json_file(args.json_file)
        desc = record.get("prob_desc_description") or ""
        code = record.get("source_code") or ""
        
        # Preprocess
        desc_processed = preprocess_description(desc)
        code_processed = preprocess_code(code)
        
        # Build DataFrame for prediction
        data = {}
        if "description" in model.features:
            data["text_description"] = [desc_processed]
        if "code" in model.features:
            data["text_code"] = [code_processed]
        X = pd.DataFrame(data)
        
        tags = model.predict(X, threshold=args.threshold)[0]
        
        out = { "predicted_tags": tags }
        print(json.dumps(out, indent=2))
    
    else:  # args.data_dir
        # Dataset mode - predict on all files in directory
        df = load_dataset_as_dataframe(args.data_dir, features=model.features)
        
        # Apply preprocessing only to loaded features
        if "description" in model.features:
            df["text_description"] = df["text_description"].apply(preprocess_description)
        if "code" in model.features:
            df["text_code"] = df["text_code"].apply(preprocess_code)
        
        # Predict directly on DataFrame
        all_tags = model.predict(df, threshold=args.threshold)
        
        predictions = []
        for i, tags in enumerate(all_tags):
            predictions.append({
                "problem_id": df.iloc[i]["problem_id"],
                "predicted_tags": tags,
            })
        
        out: Dict[str, Any] = {
            "predictions": predictions,
        }
        print(json.dumps(out, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Codeforces problem tag classifier")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    p_train = subparsers.add_parser("train", help="Train a new model")
    p_train.add_argument("--data-dir", required=True, help="Directory with json files")
    p_train.add_argument(
        "--model-path",
        required=True,
        help="Path to save the trained model (e.g. models/model.joblib)",
    )
    p_train.add_argument(
        "--focus-tags-only",   # this one is not required as it a boolean flag
        action="store_true",
        help="Keep only the 8 focus tags in targets",
    )
    p_train.add_argument(
        "--random-state",
        type=int,
        default=42,  
        help="Random seed for reproducibility (default: 42)",
    )
    p_train.add_argument(
        "--train-size",
        type=float,
        default=0.7,
        help="Fraction of data to use for training (default: 0.7)",
    )
    p_train.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Fraction of data to use for validation (default: 0.15)",
    )
    p_train.add_argument(
        "--features",
        type=str,
        default="description",
        help="Comma-separated list of features to use: 'description', 'code', or 'description,code' (default: 'description')",
    )
    p_train.add_argument(
        "--embedding",
        type=str,
        default="tfidf",
        choices=["tfidf", "word2vec"],
        help="Embedding method to use (default: 'tfidf')",
    )
    p_train.add_argument(
        "--word2vec-vector-size",
        type=int,
        default=100,
        help="Dimension of Word2Vec embeddings (default: 100)",
    )
    p_train.add_argument(
        "--word2vec-min-count",
        type=int,
        default=2,
        help="Minimum word count for Word2Vec (default: 2)",
    )
    p_train.add_argument(
        "--word2vec-epochs",
        type=int,
        default=10,
        help="Number of training epochs for Word2Vec (default: 10)",
    )
    p_train.set_defaults(func=cmd_train)

    # Predict
    p_pred = subparsers.add_parser("predict", help="Predict tags for a JSON file or dataset")
    p_pred.add_argument("--model-path", required=True, help="Path to a trained model file")
    p_pred.add_argument("--json-file", help="Path to a single sample_*.json file")
    p_pred.add_argument("--data-dir", help="Path to directory with json files (external dataset)")
    p_pred.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for assigning tags",
    )
    p_pred.set_defaults(func=cmd_predict)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
