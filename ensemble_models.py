"""
Ensemble Learning Module for Placement Classification and Salary Package Regression.
Implements separate models for:
1. GBM (Gradient Boosting Machine)
2. LightGBM (Light Gradient Boosting Machine)
3. XGBoost (eXtreme Gradient Boosting)
4. AdaBoost (Adaptive Boosting)
5. Random Forest & Bagging
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Standard Sklearn Ensembles
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    BaggingClassifier, BaggingRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)

# LightGBM
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    r2_score, mean_absolute_error, mean_squared_error
)
from utils import load_data

MODEL_DIR = "saved_models"


def _build_preprocessor(df, feature_cols):
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
    return preprocessor, num_cols, cat_cols


def _eval_classifier(pipeline, X_test, y_test, num_cols, cat_cols):
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred
    
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    step = max(1, len(fpr) // 100)
    roc_data = {
        "fpr": np.round(fpr[::step], 4).tolist(),
        "tpr": np.round(tpr[::step], 4).tolist()
    }
    
    # Feature Importances (if available)
    feature_importances = []
    try:
        classifier_step = pipeline.named_steps["classifier"]
        if hasattr(classifier_step, "feature_importances_"):
            ohe_cols = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_cols).tolist()
            all_features = num_cols + ohe_cols
            importances = classifier_step.feature_importances_
            
            imp_df = pd.DataFrame({
                "Feature": all_features,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(20)
            
            feature_importances = [
                {"Feature": row["Feature"], "Importance": round(float(row["Importance"]), 4)}
                for _, row in imp_df.iterrows()
                if row["Importance"] > 0
            ]
    except Exception as e:
        print(f"Warning extracting feature importances: {e}")
        
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm,
        "roc_curve": roc_data,
        "feature_importances": feature_importances
    }


def _eval_regressor(pipeline, X_test, y_test, num_cols, cat_cols):
    y_pred = pipeline.predict(X_test)
    mse = float(mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    
    feature_importances = []
    try:
        regressor_step = pipeline.named_steps["regressor"]
        if hasattr(regressor_step, "feature_importances_"):
            ohe_cols = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_cols).tolist()
            all_features = num_cols + ohe_cols
            importances = regressor_step.feature_importances_
            
            imp_df = pd.DataFrame({
                "Feature": all_features,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(20)
            
            feature_importances = [
                {"Feature": row["Feature"], "Importance": round(float(row["Importance"]), 4)}
                for _, row in imp_df.iterrows()
                if row["Importance"] > 0
            ]
    except Exception as e:
        print(f"Warning extracting regressor feature importances: {e}")
        
    return {
        "r2_score": round(r2, 4),
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "feature_importances": feature_importances
    }


def train_and_evaluate_ensemble_models(custom_df=None):
    """
    Trains separate models for:
    - Random Forest & Bagging
    - Standard GBM (Gradient Boosting Machine)
    - LightGBM (Light Gradient Boosting Machine)
    - XGBoost (eXtreme Gradient Boosting Machine)
    - AdaBoost (Adaptive Boosting)
    
    Saves joblib pipelines and returns evaluation metrics dictionary.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = custom_df if custom_df is not None else load_data()
    
    # Subsample if dataset is very large for rapid, responsive training
    if len(df) > 10000:
        df_sample = df.sample(n=10000, random_state=42)
    else:
        df_sample = df
    
    drop_cols = ["StudentID", "PlacementStatus", "Salary Package", "IsAnomaly"]
    feature_cols = [c for c in df_sample.columns if c not in drop_cols]
    
    X = df_sample[feature_cols]
    y_cls = df_sample["PlacementStatus"]
    y_reg = df_sample["Salary Package"]
    
    preprocessor, num_cols, cat_cols = _build_preprocessor(df_sample, feature_cols)
    
    # Stratified split for classification
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    
    # Split for regression
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    
    pipelines = {}
    metrics = {}
    
    # ------------------ 1. BAGGING & RANDOM FOREST ------------------
    print("Training Random Forest Classifier...")
    rf_c_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=20, max_depth=8, random_state=42))
    ])
    rf_c_pipe.fit(Xc_train, yc_train)
    pipelines["placement_rf_pipeline.joblib"] = rf_c_pipe
    metrics["rf_classifier_metrics"] = _eval_classifier(rf_c_pipe, Xc_test, yc_test, num_cols, cat_cols)
    
    print("Training Random Forest Regressor...")
    rf_r_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=20, max_depth=8, random_state=42))
    ])
    rf_r_pipe.fit(Xr_train, yr_train)
    pipelines["salary_rf_pipeline.joblib"] = rf_r_pipe
    metrics["rf_regressor_metrics"] = _eval_regressor(rf_r_pipe, Xr_test, yr_test, num_cols, cat_cols)
    
    print("Training Bagging Classifier...")
    bag_c_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=8), n_estimators=15, random_state=42))
    ])
    bag_c_pipe.fit(Xc_train, yc_train)
    pipelines["placement_bagging_pipeline.joblib"] = bag_c_pipe
    metrics["bagging_classifier_metrics"] = _eval_classifier(bag_c_pipe, Xc_test, yc_test, num_cols, cat_cols)

    print("Training Bagging Regressor...")
    bag_r_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=8), n_estimators=15, random_state=42))
    ])
    bag_r_pipe.fit(Xr_train, yr_train)
    pipelines["salary_bagging_pipeline.joblib"] = bag_r_pipe
    metrics["bagging_regressor_metrics"] = _eval_regressor(bag_r_pipe, Xr_test, yr_test, num_cols, cat_cols)
    
    # ------------------ 2. STANDARD GBM (Gradient Boosting Machine) ------------------
    print("Training Standard GBM (Gradient Boosting Machine) Classifier...")
    gbm_c_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(n_estimators=30, learning_rate=0.1, max_depth=4, random_state=42))
    ])
    gbm_c_pipe.fit(Xc_train, yc_train)
    pipelines["placement_gbm_pipeline.joblib"] = gbm_c_pipe
    metrics["gbm_classifier_metrics"] = _eval_classifier(gbm_c_pipe, Xc_test, yc_test, num_cols, cat_cols)

    print("Training Standard GBM (Gradient Boosting Machine) Regressor...")
    gbm_r_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(n_estimators=30, learning_rate=0.1, max_depth=4, random_state=42))
    ])
    gbm_r_pipe.fit(Xr_train, yr_train)
    pipelines["salary_gbm_pipeline.joblib"] = gbm_r_pipe
    metrics["gbm_regressor_metrics"] = _eval_regressor(gbm_r_pipe, Xr_test, yr_test, num_cols, cat_cols)

    # ------------------ 3. LIGHTGBM ------------------
    print("Training LightGBM Classifier...")
    if HAS_LIGHTGBM:
        lgb_c = LGBMClassifier(n_estimators=30, learning_rate=0.1, max_depth=5, random_state=42, verbose=-1)
    else:
        lgb_c = HistGradientBoostingClassifier(max_iter=30, learning_rate=0.1, max_depth=5, random_state=42)
    
    lgb_c_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", lgb_c)
    ])
    lgb_c_pipe.fit(Xc_train, yc_train)
    pipelines["placement_lightgbm_pipeline.joblib"] = lgb_c_pipe
    metrics["lightgbm_classifier_metrics"] = _eval_classifier(lgb_c_pipe, Xc_test, yc_test, num_cols, cat_cols)

    print("Training LightGBM Regressor...")
    if HAS_LIGHTGBM:
        lgb_r = LGBMRegressor(n_estimators=30, learning_rate=0.1, max_depth=5, random_state=42, verbose=-1)
    else:
        lgb_r = HistGradientBoostingRegressor(max_iter=30, learning_rate=0.1, max_depth=5, random_state=42)
        
    lgb_r_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", lgb_r)
    ])
    lgb_r_pipe.fit(Xr_train, yr_train)
    pipelines["salary_lightgbm_pipeline.joblib"] = lgb_r_pipe
    metrics["lightgbm_regressor_metrics"] = _eval_regressor(lgb_r_pipe, Xr_test, yr_test, num_cols, cat_cols)

    # ------------------ 4. XGBOOST ------------------
    print("Training XGBoost Classifier...")
    if HAS_XGBOOST:
        xgb_c = XGBClassifier(n_estimators=30, learning_rate=0.1, max_depth=4, random_state=42, eval_metric="logloss")
    else:
        xgb_c = GradientBoostingClassifier(n_estimators=30, learning_rate=0.1, max_depth=4, random_state=42)
        
    xgb_c_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", xgb_c)
    ])
    xgb_c_pipe.fit(Xc_train, yc_train)
    pipelines["placement_xgb_pipeline.joblib"] = xgb_c_pipe
    metrics["xgb_classifier_metrics"] = _eval_classifier(xgb_c_pipe, Xc_test, yc_test, num_cols, cat_cols)

    print("Training XGBoost Regressor...")
    if HAS_XGBOOST:
        xgb_r = XGBRegressor(n_estimators=30, learning_rate=0.1, max_depth=4, random_state=42)
    else:
        xgb_r = GradientBoostingRegressor(n_estimators=30, learning_rate=0.1, max_depth=4, random_state=42)
        
    xgb_r_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", xgb_r)
    ])
    xgb_r_pipe.fit(Xr_train, yr_train)
    pipelines["salary_xgb_pipeline.joblib"] = xgb_r_pipe
    metrics["xgb_regressor_metrics"] = _eval_regressor(xgb_r_pipe, Xr_test, yr_test, num_cols, cat_cols)

    # ------------------ 5. ADABOOST ------------------
    print("Training AdaBoost Classifier...")
    ada_c_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", AdaBoostClassifier(n_estimators=20, learning_rate=0.1, random_state=42))
    ])
    ada_c_pipe.fit(Xc_train, yc_train)
    pipelines["placement_adaboost_pipeline.joblib"] = ada_c_pipe
    metrics["adaboost_classifier_metrics"] = _eval_classifier(ada_c_pipe, Xc_test, yc_test, num_cols, cat_cols)

    print("Training AdaBoost Regressor...")
    ada_r_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", AdaBoostRegressor(n_estimators=20, learning_rate=0.1, random_state=42))
    ])
    ada_r_pipe.fit(Xr_train, yr_train)
    pipelines["salary_adaboost_pipeline.joblib"] = ada_r_pipe
    metrics["adaboost_regressor_metrics"] = _eval_regressor(ada_r_pipe, Xr_test, yr_test, num_cols, cat_cols)
    
    # Save Joblib Pipelines
    for filename, pipe in pipelines.items():
        joblib.dump(pipe, os.path.join(MODEL_DIR, filename))
        
    print("All ML Ensemble Models (GBM, LightGBM, XGBoost, AdaBoost, Random Forest, Bagging) trained successfully!")
    return pipelines, metrics


if __name__ == "__main__":
    train_and_evaluate_ensemble_models()
