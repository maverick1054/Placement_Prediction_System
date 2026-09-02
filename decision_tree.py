"""
Decision Tree Module for Placement Status Classification and Salary Package Regression
Supports ID3 (Information Gain/Entropy), C4.5 (Gain Ratio & Pruned Tree), and CART / Karth (Gini Impurity & MSE).
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
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    r2_score, mean_absolute_error, mean_squared_error
)
from utils import load_data

MODEL_DIR = "saved_models"

def _evaluate_classifier(pipeline, X_test, y_test, num_cols, cat_cols, model_name="Classifier"):
    clf = pipeline.named_steps["classifier"]
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred
    
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.5
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    step = max(1, len(fpr) // 100)
    roc_data = {
        "fpr": np.round(fpr[::step], 4).tolist(),
        "tpr": np.round(tpr[::step], 4).tolist()
    }
    
    # Extract Tree Impurity Metrics
    tree = clf.tree_
    criterion_name = str(getattr(clf, "criterion", "gini")).title()
    root_impurity = round(float(tree.impurity[0]), 4)
    mean_impurity = round(float(np.mean(tree.impurity)), 4)
    
    is_leaf = tree.children_left == -1
    leaf_impurities = tree.impurity[is_leaf]
    leaf_samples = tree.n_node_samples[is_leaf]
    total_samples = tree.n_node_samples[0]
    weighted_leaf_impurity = round(float(np.sum(leaf_impurities * leaf_samples) / total_samples), 4)
    impurity_reduction = round(float(root_impurity - weighted_leaf_impurity), 4)
    node_count = int(tree.node_count)
    leaf_count = int(tree.n_leaves)
    max_depth = int(clf.get_depth())
    
    clf_importances = []
    try:
        ohe_cols = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_cols).tolist()
        all_features = num_cols + ohe_cols
        importances = clf.feature_importances_
        
        imp_df = pd.DataFrame({
            "Feature": all_features,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(20)
        
        clf_importances = [
            {"Feature": row["Feature"], "Importance": round(float(row["Importance"]), 4)}
            for _, row in imp_df.iterrows()
            if row["Importance"] > 0
        ]
    except Exception as e:
        print(f"Warning extracting {model_name} importances: {e}")
        
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm,
        "roc_curve": roc_data,
        "feature_importances": clf_importances,
        "impurity_metrics": {
            "criterion": criterion_name,
            "root_impurity": root_impurity,
            "mean_impurity": mean_impurity,
            "weighted_leaf_impurity": weighted_leaf_impurity,
            "impurity_reduction": impurity_reduction,
            "node_count": node_count,
            "leaf_count": leaf_count,
            "max_depth": max_depth
        }
    }

def _evaluate_regressor(pipeline, X_test, y_test, num_cols, cat_cols, model_name="Regressor"):
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    reg_importances = []
    try:
        ohe_cols = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_cols).tolist()
        all_features = num_cols + ohe_cols
        importances_r = pipeline.named_steps["regressor"].feature_importances_
        
        imp_reg_df = pd.DataFrame({
            "Feature": all_features,
            "Importance": importances_r
        }).sort_values(by="Importance", ascending=False).head(20)
        
        reg_importances = [
            {"Feature": row["Feature"], "Importance": round(float(row["Importance"]), 4)}
            for _, row in imp_reg_df.iterrows()
            if row["Importance"] > 0
        ]
    except Exception as e:
        print(f"Warning extracting {model_name} importances: {e}")
        
    return {
        "r2_score": round(float(r2_score(y_test, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(np.sqrt(mse)), 4),
        "feature_importances": reg_importances
    }

def train_and_evaluate_decision_tree(custom_df=None):
    """
    Trains ID3, C4.5, and CART (Karth) Classifiers & Regressors.
    Saves trained pipelines and returns dictionary of pipelines & metric dictionaries.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = custom_df if custom_df is not None else load_data()
    
    drop_cols = ["StudentID", "PlacementStatus", "Salary Package", "IsAnomaly"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(include=["object", "string", "category"]).columns.tolist()
    
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
    
    # ------------------ DATA SPLITS ------------------
    y_cls = df["PlacementStatus"]
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    
    y_reg = df["Salary Package"]
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    # ------------------ 1. ID3 DECISION TREE (Information Gain / Entropy) ------------------
    print("Training ID3 Classifier (Entropy / Information Gain)...")
    id3_c_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42))
    ])
    id3_c_pipeline.fit(X_train_c, y_train_c)
    id3_c_metrics = _evaluate_classifier(id3_c_pipeline, X_test_c, y_test_c, num_cols, cat_cols, "ID3 Classifier")

    print("Training ID3 Regressor (MAE / Entropy equivalent)...")
    id3_r_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", DecisionTreeRegressor(criterion="absolute_error", max_depth=5, random_state=42))
    ])
    id3_r_pipeline.fit(X_train_r, y_train_r)
    id3_r_metrics = _evaluate_regressor(id3_r_pipeline, X_test_r, y_test_r, num_cols, cat_cols, "ID3 Regressor")

    # ------------------ 2. C4.5 DECISION TREE (Gain Ratio & Post-Pruning) ------------------
    print("Training C4.5 Classifier (Gain Ratio / Pruned Tree)...")
    c45_c_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(criterion="entropy", min_samples_split=5, min_samples_leaf=2, ccp_alpha=0.005, max_depth=5, random_state=42))
    ])
    c45_c_pipeline.fit(X_train_c, y_train_c)
    c45_c_metrics = _evaluate_classifier(c45_c_pipeline, X_test_c, y_test_c, num_cols, cat_cols, "C4.5 Classifier")

    print("Training C4.5 Regressor (Pruned Tree with min samples split/leaf)...")
    c45_r_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", DecisionTreeRegressor(criterion="squared_error", min_samples_split=5, min_samples_leaf=2, max_depth=5, random_state=42))
    ])
    c45_r_pipeline.fit(X_train_r, y_train_r)
    c45_r_metrics = _evaluate_regressor(c45_r_pipeline, X_test_r, y_test_r, num_cols, cat_cols, "C4.5 Regressor")

    # ------------------ 3. CART / KARTH DECISION TREE (Gini Impurity & MSE) ------------------
    print("Training CART (Karth) Classifier (Gini Impurity)...")
    cart_c_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42))
    ])
    cart_c_pipeline.fit(X_train_c, y_train_c)
    cart_c_metrics = _evaluate_classifier(cart_c_pipeline, X_test_c, y_test_c, num_cols, cat_cols, "CART Classifier")

    print("Training CART (Karth) Regressor (Mean Squared Error)...")
    cart_r_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", DecisionTreeRegressor(criterion="squared_error", max_depth=5, random_state=42))
    ])
    cart_r_pipeline.fit(X_train_r, y_train_r)
    cart_r_metrics = _evaluate_regressor(cart_r_pipeline, X_test_r, y_test_r, num_cols, cat_cols, "CART Regressor")

    # ------------------ SAVE MODELS ------------------
    joblib.dump(id3_c_pipeline, os.path.join(MODEL_DIR, "placement_id3_pipeline.joblib"))
    joblib.dump(id3_r_pipeline, os.path.join(MODEL_DIR, "salary_id3_pipeline.joblib"))

    joblib.dump(c45_c_pipeline, os.path.join(MODEL_DIR, "placement_c45_pipeline.joblib"))
    joblib.dump(c45_r_pipeline, os.path.join(MODEL_DIR, "salary_c45_pipeline.joblib"))

    joblib.dump(cart_c_pipeline, os.path.join(MODEL_DIR, "placement_cart_pipeline.joblib"))
    joblib.dump(cart_r_pipeline, os.path.join(MODEL_DIR, "salary_cart_pipeline.joblib"))

    # Also save aliases for backward compatibility
    joblib.dump(cart_c_pipeline, os.path.join(MODEL_DIR, "placement_dt_pipeline.joblib"))
    joblib.dump(cart_r_pipeline, os.path.join(MODEL_DIR, "salary_dt_pipeline.joblib"))

    print("Decision Tree models (ID3, C4.5, CART/Karth) training completed successfully!")

    dt_pipelines = {
        "id3_c": id3_c_pipeline,
        "id3_r": id3_r_pipeline,
        "c45_c": c45_c_pipeline,
        "c45_r": c45_r_pipeline,
        "cart_c": cart_c_pipeline,
        "cart_r": cart_r_pipeline,
        "dt_c": cart_c_pipeline,
        "dt_r": cart_r_pipeline
    }

    dt_metrics = {
        "id3_classifier_metrics": id3_c_metrics,
        "id3_regressor_metrics": id3_r_metrics,
        "c45_classifier_metrics": c45_c_metrics,
        "c45_regressor_metrics": c45_r_metrics,
        "cart_classifier_metrics": cart_c_metrics,
        "cart_regressor_metrics": cart_r_metrics,
        "dt_classifier_metrics": cart_c_metrics,
        "dt_regressor_metrics": cart_r_metrics
    }

    return dt_pipelines, dt_metrics

if __name__ == "__main__":
    pipes, metrics = train_and_evaluate_decision_tree()
    print("\n--- ID3 Classifier ---")
    print(f"Accuracy: {metrics['id3_classifier_metrics']['accuracy']*100:.2f}% | ROC-AUC: {metrics['id3_classifier_metrics']['roc_auc']:.4f}")
    print("--- C4.5 Classifier ---")
    print(f"Accuracy: {metrics['c45_classifier_metrics']['accuracy']*100:.2f}% | ROC-AUC: {metrics['c45_classifier_metrics']['roc_auc']:.4f}")
    print("--- CART (Karth) Classifier ---")
    print(f"Accuracy: {metrics['cart_classifier_metrics']['accuracy']*100:.2f}% | ROC-AUC: {metrics['cart_classifier_metrics']['roc_auc']:.4f}")

