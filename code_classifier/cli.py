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
    # Parse features from comma-separated string
    if args.features:
        features = [f.strip() for f in args.features.split(",")]
    else:
        features = ["description"]  # default to description only
    
    # Validate features - we only support description and code for now
    valid_features = {"description", "code"}
    for feat in features:
        if feat not in valid_features:
            raise ValueError(f"Unknown feature: {feat}. Available: {valid_features}")
    
    # Warn if using both features - might not always improve performance
    if len(features) > 1:
        print(f"Using features: {features}")
    
    # Load the dataset
    df = load_dataset_as_dataframe(args.data_dir, features=features)
    print(f"Loaded {len(df)} examples")
    
    # Clean the text data
    if "description" in features:
        df["text_description"] = df["text_description"].apply(preprocess_description)
    if "code" in features:
        df["text_code"] = df["text_code"].apply(preprocess_code)
    
    # Optionally filter to only the 8 focus tags
    # This reduces the dataset but makes the problem more focused
    if args.focus_tags_only:
        df = filter_to_focus_tags(df, DEFAULT_FOCUS_TAGS)
        df = remove_empty_tags(df)
        print(f"After filtering to focus tags: {len(df)} examples")

    random_state = args.random_state 
    train_size = args.train_size
    val_size = args.val_size

    # Get embedding type
    embedding_type = getattr(args, "embedding", "tfidf")
    word2vec_vector_size = getattr(args, "word2vec_vector_size", 100)
    word2vec_min_count = getattr(args, "word2vec_min_count", 2)
    word2vec_epochs = getattr(args, "word2vec_epochs", 10)
    
    # Note: TF-IDF generally performs better on this dataset than Word2Vec
    if embedding_type == "word2vec":
        print(f"Using Word2Vec (vector_size={word2vec_vector_size})")
    
    # Train the model with train/val/test split
    # We keep test set separate for final evaluation after comparing models
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
    
    # Save the trained model
    model_dir = os.path.dirname(args.model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    model.save(args.model_path)

    # Build the report with all the config and results
    test_size = 1.0 - train_size - val_size
    
    # Embedding parameters depend on the method used
    if embedding_type == "tfidf":
        embedding_params = {
            "ngram_range": [1, 2],  # unigrams and bigrams
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
            "solver": "liblinear",  # works well for this dataset size
            "class_weight": "balanced",  # helps with imbalanced classes
            "max_iter": 250,
            "strategy": "OneVsRest",
        },
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "random_state": random_state,
        "focus_tags_only": args.focus_tags_only,
        "validation_metrics": val_report,
        "test_set_ids": test_ids,  # save test IDs to avoid using them during development
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save report in reports/ folder
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
    # Load the trained model
    model = TrainedModel.load(args.model_path)
    
    # Check that user provided either single file or directory, not both
    if args.json_file and args.data_dir:
        raise ValueError("Please specify either --json-file OR --data-dir, not both")
    if not args.json_file and not args.data_dir:
        raise ValueError("Need to specify either --json-file or --data-dir")
    
    if args.json_file:
        # Single file prediction
        record = load_json_file(args.json_file)
        desc = record.get("prob_desc_description") or ""
        code = record.get("source_code") or ""
        
        # Apply same preprocessing as during training
        desc_processed = preprocess_description(desc)
        code_processed = preprocess_code(code)
        
        # Build DataFrame with the features the model expects
        data = {}
        if "description" in model.features:
            data["text_description"] = [desc_processed]
        if "code" in model.features:
            data["text_code"] = [code_processed]
        X = pd.DataFrame(data)
        
        tags = model.predict(X, threshold=args.threshold)[0]
        
        out = { "predicted_tags": tags }
        print(json.dumps(out, indent=2))
    
    else:
        # Batch prediction on a directory
        df = load_dataset_as_dataframe(args.data_dir, features=model.features)
        
        # Preprocess the data
        if "description" in model.features:
            df["text_description"] = df["text_description"].apply(preprocess_description)
        if "code" in model.features:
            df["text_code"] = df["text_code"].apply(preprocess_code)
        
        # Get predictions for all examples
        all_tags = model.predict(df, threshold=args.threshold)
        
        # Build output with problem IDs
        predictions = []
        for i, tags in enumerate(all_tags):
            predictions.append({
                "problem_id": df.iloc[i]["problem_id"],
                "predicted_tags": tags,
            })
        
        out = {
            "predictions": predictions,
        }
        print(json.dumps(out, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Codeforces problem tag classifier")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    p_train = subparsers.add_parser("train", help="Train a new model")
    p_train.add_argument("--data-dir", required=True, help="Directory with json files")
    p_train.add_argument("--model-path", required=True, help="Path to save the trained model")
    p_train.add_argument(
        "--focus-tags-only",
        action="store_true",
        help="Filter to only the 8 focus tags (reduces dataset size)",
    )
    p_train.add_argument("--random-state", type=int, default=42, help="Random seed")
    p_train.add_argument("--train-size", type=float, default=0.7, help="Training set fraction")
    p_train.add_argument("--val-size", type=float, default=0.15, help="Validation set fraction")
    p_train.add_argument(
        "--features",
        type=str,
        default="description",
        help="Features to use: 'description', 'code', or 'description,code'",
    )
    p_train.add_argument(
        "--embedding",
        type=str,
        default="tfidf",
        choices=["tfidf", "word2vec"],
        help="Embedding method (tfidf works better on this dataset)",
    )
    p_train.add_argument("--word2vec-vector-size", type=int, default=100, help="Word2Vec dimension")
    p_train.add_argument("--word2vec-min-count", type=int, default=2, help="Word2Vec min word count")
    p_train.add_argument("--word2vec-epochs", type=int, default=10, help="Word2Vec training epochs")
    p_train.set_defaults(func=cmd_train)

    # Predict command
    p_pred = subparsers.add_parser("predict", help="Predict tags")
    p_pred.add_argument("--model-path", required=True, help="Path to trained model")
    p_pred.add_argument("--json-file", help="Single file to predict")
    p_pred.add_argument("--data-dir", help="Directory with files to predict")
    p_pred.add_argument("--threshold", type=float, default=0.5, help="Probability threshold")
    p_pred.set_defaults(func=cmd_predict)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
