"""
Logistic Regression Module for Placement Status Classification (Placed vs Not Placed)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from utils import load_data

MODEL_DIR = "saved_models"

def train_and_evaluate_logistic_regression(custom_df=None):
    """
    Trains a Logistic Regression pipeline for predicting PlacementStatus (1 = Placed, 0 = Not Placed).
    Saves the trained pipeline and returns classification metrics.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = custom_df if custom_df is not None else load_data()
    
    if "PlacementStatus" not in df.columns:
        raise ValueError("PlacementStatus column missing from dataset.")

    # Target column
    y = df["PlacementStatus"]
    
    # Feature columns (exclude IDs, target, anomalies, and salary package)
    drop_cols = ["StudentID", "PlacementStatus", "Salary Package", "IsAnomaly"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(include=["object", "string", "category"]).columns.tolist()
    
    # Preprocessing pipelines
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols)
        ]
    )
    
    X = df[feature_cols]
    
    # Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training Placement Classifier (Logistic Regression)...")
    logistic_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    logistic_pipeline.fit(X_train, y_train)
    
    y_pred = logistic_pipeline.predict(X_test)
    y_prob = logistic_pipeline.predict_proba(X_test)[:, 1]
    
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    
    cm = confusion_matrix(y_test, y_pred).tolist()  # [[TN, FP], [FN, TP]]
    
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    # Downsample ROC curve data points for json metadata efficiency
    step = max(1, len(fpr) // 100)
    roc_data = {
        "fpr": np.round(fpr[::step], 4).tolist(),
        "tpr": np.round(tpr[::step], 4).tolist()
    }
    
    # Feature Odds Ratios / Coefficients
    try:
        ohe_cols = logistic_pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_cols).tolist()
        all_transformed_features = num_cols + ohe_cols
        coefs = logistic_pipeline.named_steps["classifier"].coef_[0]
        odds_ratios = np.exp(coefs)
        
        odds_df = pd.DataFrame({
            "Feature": all_transformed_features,
            "Log_Odds": coefs,
            "Odds_Ratio": odds_ratios,
            "Abs_Log_Odds": np.abs(coefs)
        }).sort_values(by="Abs_Log_Odds", ascending=False).head(20)
        
        odds_records = odds_df[["Feature", "Log_Odds", "Odds_Ratio"]].to_dict(orient="records")
        for rec_item in odds_records:
            rec_item["Log_Odds"] = round(float(rec_item["Log_Odds"]), 4)
            rec_item["Odds_Ratio"] = round(float(rec_item["Odds_Ratio"]), 4)
    except Exception as e:
        print(f"Warning extracting Logistic Regression coefficients: {e}")
        odds_records = []
        
    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm,
        "roc_curve": roc_data,
        "odds_ratios": odds_records
    }
    
    joblib.dump(logistic_pipeline, os.path.join(MODEL_DIR, "placement_logistic_pipeline.joblib"))
    print("Logistic Regression training completed successfully!")
    return logistic_pipeline, metrics

if __name__ == "__main__":
    pipeline, metrics = train_and_evaluate_logistic_regression()
    print("\n--- Logistic Regression Metrics ---")
    print(f"Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall   : {metrics['recall'] * 100:.2f}%")
    print(f"F1 Score : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("Confusion Matrix (TN, FP / FN, TP):", metrics['confusion_matrix'])