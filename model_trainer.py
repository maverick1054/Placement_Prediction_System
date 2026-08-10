import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    r2_score, mean_absolute_error, mean_squared_error
)
from utils import load_data

MODEL_DIR = "saved_models"

def train_and_evaluate_models():
    """
    Trains classification model for PlacementStatus and regression model for Salary Package.
    Saves trained pipelines and returns metrics dictionary.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_data()
    
    # Feature columns vs Target columns
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
    y_placement = df["PlacementStatus"]
    y_salary = df["Salary Package"]
    
    # Train-test split
    X_train, X_test, y_p_train, y_p_test, y_s_train, y_s_test = train_test_split(
        X, y_placement, y_salary, test_size=0.2, random_state=42, stratify=y_placement
    )
    
    # 1. Placement Status Classifier
    print("Training Placement Classifier...")
    clf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1))
    ])
    clf_pipeline.fit(X_train, y_p_train)
    
    y_p_pred = clf_pipeline.predict(X_test)
    y_p_proba = clf_pipeline.predict_proba(X_test)[:, 1]
    
    clf_metrics = {
        "accuracy": round(float(accuracy_score(y_p_test, y_p_pred)), 4),
        "precision": round(float(precision_score(y_p_test, y_p_pred)), 4),
        "recall": round(float(recall_score(y_p_test, y_p_pred)), 4),
        "f1_score": round(float(f1_score(y_p_test, y_p_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_p_test, y_p_proba)), 4),
        "confusion_matrix": confusion_matrix(y_p_test, y_p_pred).tolist()
    }
    
    # 2. Salary Regressor (Trained on placed students or full dataset)
    print("Training Salary Regressor...")
    reg_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1))
    ])
    reg_pipeline.fit(X_train, y_s_train)
    
    y_s_pred = reg_pipeline.predict(X_test)
    mse = mean_squared_error(y_s_test, y_s_pred)
    
    reg_metrics = {
        "r2_score": round(float(r2_score(y_s_test, y_s_pred)), 4),
        "mae": round(float(mean_absolute_error(y_s_test, y_s_pred)), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(np.sqrt(mse)), 4)
    }
    
    # Feature Importances extraction
    # Get feature names after one-hot encoding
    ohe_cols = clf_pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_cols).tolist()
    all_transformed_features = num_cols + ohe_cols
    
    clf_importances = clf_pipeline.named_steps["classifier"].feature_importances_
    importance_df = pd.DataFrame({
        "Feature": all_transformed_features,
        "Importance": clf_importances
    }).sort_values(by="Importance", ascending=False).head(15)
    
    # Save Artifacts
    joblib.dump(clf_pipeline, os.path.join(MODEL_DIR, "placement_pipeline.joblib"))
    joblib.dump(reg_pipeline, os.path.join(MODEL_DIR, "salary_pipeline.joblib"))
    
    metadata = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "clf_metrics": clf_metrics,
        "reg_metrics": reg_metrics,
        "feature_importances": importance_df.to_dict(orient="records"),
        "cat_options": {c: sorted([str(val) for val in df[c].dropna().unique().tolist()]) for c in cat_cols}
    }
    
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    print("Model training completed successfully!")
    return metadata

def load_trained_pipelines():
    """
    Loads saved classification and regression pipelines along with metadata.
    """
    clf_path = os.path.join(MODEL_DIR, "placement_pipeline.joblib")
    reg_path = os.path.join(MODEL_DIR, "salary_pipeline.joblib")
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    
    if not (os.path.exists(clf_path) and os.path.exists(reg_path) and os.path.exists(meta_path)):
        return train_and_evaluate_models()
        
    clf_pipeline = joblib.load(clf_path)
    reg_pipeline = joblib.load(reg_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)
        
    return clf_pipeline, reg_pipeline, metadata

if __name__ == "__main__":
    train_and_evaluate_models()
