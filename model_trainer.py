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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error
)
from utils import load_data
from logistic_regression import train_and_evaluate_logistic_regression
from decision_tree import train_and_evaluate_decision_tree
from ensemble_models import train_and_evaluate_ensemble_models
from kmeans_clustering import train_and_evaluate_kmeans
from pca_analysis import train_and_evaluate_pca

MODEL_DIR = "saved_models"

def train_and_evaluate_models(custom_df=None):
    """
    Trains ML models:
    1. Multi-feature Linear Regression (Salary Package)
    2. Simple CGPA Linear Regression (Salary Package)
    3. Logistic Regression (PlacementStatus: Placed vs Not Placed)
    4. Decision Tree Classifier (PlacementStatus)
    5. Decision Tree Regressor (Salary Package)
    6. Bagging Ensemble Models (Random Forest & Bagging Classifiers/Regressors)
    7. Boosting Ensemble Models (Gradient Boosting & AdaBoost Classifiers/Regressors)
    
    Saves trained pipelines and returns unified metrics metadata dictionary.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = custom_df if custom_df is not None else load_data()
    
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
    y_salary = df["Salary Package"]
    
    # Train-test split for salary regression
    X_train, X_test, y_s_train, y_s_test = train_test_split(
        X, y_salary, test_size=0.2, random_state=42
    )
    
    # 1. Multi-feature Linear Regression
    print("Training Salary Regressor (Linear Regression - All Features)...")
    lr_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_s_train)
    
    y_lr_pred = lr_pipeline.predict(X_test)
    lr_mse = mean_squared_error(y_s_test, y_lr_pred)
    
    lr_metrics = {
        "r2_score": round(float(r2_score(y_s_test, y_lr_pred)), 4),
        "mae": round(float(mean_absolute_error(y_s_test, y_lr_pred)), 4),
        "mse": round(float(lr_mse), 4),
        "rmse": round(float(np.sqrt(lr_mse)), 4)
    }
    
    # 2. Simple Linear Regression (CGPA -> Salary Package)
    print("Training Simple Linear Regression (CGPA vs Salary Package)...")
    cgpa_X_train = X_train[["CGPA"]]
    cgpa_X_test = X_test[["CGPA"]]
    
    simple_lr = LinearRegression()
    simple_lr.fit(cgpa_X_train, y_s_train)
    
    y_simple_lr_pred = simple_lr.predict(cgpa_X_test)
    simple_lr_mse = mean_squared_error(y_s_test, y_simple_lr_pred)
    
    slope = float(simple_lr.coef_[0])
    intercept = float(simple_lr.intercept_)
    corr_coef = float(df["CGPA"].corr(df["Salary Package"])) if "CGPA" in df.columns and "Salary Package" in df.columns else 0.0
    
    simple_lr_metrics = {
        "r2_score": round(float(r2_score(y_s_test, y_simple_lr_pred)), 4),
        "mae": round(float(mean_absolute_error(y_s_test, y_simple_lr_pred)), 4),
        "mse": round(float(simple_lr_mse), 4),
        "rmse": round(float(np.sqrt(simple_lr_mse)), 4),
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "correlation": round(corr_coef, 4)
    }
    
    # Extract Linear Regression Coefficients for Interpretability
    try:
        ohe_cols = lr_pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_cols).tolist()
        all_transformed_features = num_cols + ohe_cols
        coefficients = lr_pipeline.named_steps["regressor"].coef_
        
        coef_df = pd.DataFrame({
            "Feature": all_transformed_features,
            "Coefficient": coefficients,
            "Abs_Coefficient": np.abs(coefficients)
        }).sort_values(by="Abs_Coefficient", ascending=False).head(20)
        coef_records = coef_df[["Feature", "Coefficient"]].to_dict(orient="records")
    except Exception as e:
        print(f"Warning extracting coefficients: {e}")
        coef_records = []
        
    # 3. Logistic Regression Classifier (PlacementStatus)
    logistic_pipeline, logistic_metrics = train_and_evaluate_logistic_regression(df)

    # 4. Decision Tree Models (ID3, C4.5, CART / Karth)
    dt_pipes, dt_metrics = train_and_evaluate_decision_tree(df)
    
    # 5 & 6. Ensemble Models (Bagging & Boosting)
    ensemble_pipes, ensemble_metrics = train_and_evaluate_ensemble_models(df)
    
    # 7. Unsupervised Learning (KMeans Clustering & Silhouette Score Analysis)
    print("Training KMeans Unsupervised Clustering & Silhouette Analysis...")
    kmeans_pipeline, kmeans_metrics = train_and_evaluate_kmeans(df)
    
    # 8. Principal Component Analysis (PCA & Dimensionality Reduction Models)
    print("Training PCA Dimensionality Reduction & PCA Downstream Models...")
    pca_metrics, pca_pipelines, pca_df = train_and_evaluate_pca(df)
    
    # Save Artifacts
    joblib.dump(lr_pipeline, os.path.join(MODEL_DIR, "salary_lr_pipeline.joblib"))
    joblib.dump(simple_lr, os.path.join(MODEL_DIR, "simple_cgpa_lr_model.joblib"))
    joblib.dump(logistic_pipeline, os.path.join(MODEL_DIR, "placement_logistic_pipeline.joblib"))
    
    metadata = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "lr_metrics": lr_metrics,
        "simple_lr_metrics": simple_lr_metrics,
        "logistic_metrics": logistic_metrics,
        "id3_classifier_metrics": dt_metrics.get("id3_classifier_metrics", {}),
        "id3_regressor_metrics": dt_metrics.get("id3_regressor_metrics", {}),
        "c45_classifier_metrics": dt_metrics.get("c45_classifier_metrics", {}),
        "c45_regressor_metrics": dt_metrics.get("c45_regressor_metrics", {}),
        "cart_classifier_metrics": dt_metrics.get("cart_classifier_metrics", {}),
        "cart_regressor_metrics": dt_metrics.get("cart_regressor_metrics", {}),
        "dt_classifier_metrics": dt_metrics.get("dt_classifier_metrics", {}),
        "dt_regressor_metrics": dt_metrics.get("dt_regressor_metrics", {}),
        "rf_classifier_metrics": ensemble_metrics.get("rf_classifier_metrics", {}),
        "rf_regressor_metrics": ensemble_metrics.get("rf_regressor_metrics", {}),
        "bagging_classifier_metrics": ensemble_metrics.get("bagging_classifier_metrics", {}),
        "bagging_regressor_metrics": ensemble_metrics.get("bagging_regressor_metrics", {}),
        "gbm_classifier_metrics": ensemble_metrics.get("gbm_classifier_metrics", {}),
        "gbm_regressor_metrics": ensemble_metrics.get("gbm_regressor_metrics", {}),
        "lightgbm_classifier_metrics": ensemble_metrics.get("lightgbm_classifier_metrics", {}),
        "lightgbm_regressor_metrics": ensemble_metrics.get("lightgbm_regressor_metrics", {}),
        "xgb_classifier_metrics": ensemble_metrics.get("xgb_classifier_metrics", {}),
        "xgb_regressor_metrics": ensemble_metrics.get("xgb_regressor_metrics", {}),
        "adaboost_classifier_metrics": ensemble_metrics.get("adaboost_classifier_metrics", {}),
        "adaboost_regressor_metrics": ensemble_metrics.get("adaboost_regressor_metrics", {}),
        "kmeans_metrics": kmeans_metrics,
        "pca_metrics": pca_metrics,
        "feature_coefficients": coef_records,
        "cat_options": {c: sorted([str(val) for val in df[c].dropna().unique().tolist()]) for c in cat_cols if c in df.columns}
    }
    
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    print("All ML models (Base, ID3, C4.5, CART/Karth, GBM, LightGBM, XGBoost, AdaBoost, Random Forest, Bagging, KMeans, PCA) trained and saved successfully!")
    return metadata

def load_trained_pipelines():
    """
    Loads saved ML pipelines (Linear Regression, Logistic Regression, ID3, C4.5, CART/Karth,
    Random Forest, Bagging, GBM, LightGBM, XGBoost, AdaBoost, PCA) along with metadata.
    """
    lr_path = os.path.join(MODEL_DIR, "salary_lr_pipeline.joblib")
    simple_lr_path = os.path.join(MODEL_DIR, "simple_cgpa_lr_model.joblib")
    logistic_path = os.path.join(MODEL_DIR, "placement_logistic_pipeline.joblib")
    
    id3_c_path = os.path.join(MODEL_DIR, "placement_id3_pipeline.joblib")
    id3_r_path = os.path.join(MODEL_DIR, "salary_id3_pipeline.joblib")
    c45_c_path = os.path.join(MODEL_DIR, "placement_c45_pipeline.joblib")
    c45_r_path = os.path.join(MODEL_DIR, "salary_c45_pipeline.joblib")
    cart_c_path = os.path.join(MODEL_DIR, "placement_cart_pipeline.joblib")
    cart_r_path = os.path.join(MODEL_DIR, "salary_cart_pipeline.joblib")
    
    dt_c_path = os.path.join(MODEL_DIR, "placement_dt_pipeline.joblib")
    dt_r_path = os.path.join(MODEL_DIR, "salary_dt_pipeline.joblib")
    
    rf_c_path = os.path.join(MODEL_DIR, "placement_rf_pipeline.joblib")
    rf_r_path = os.path.join(MODEL_DIR, "salary_rf_pipeline.joblib")
    bag_c_path = os.path.join(MODEL_DIR, "placement_bagging_pipeline.joblib")
    bag_r_path = os.path.join(MODEL_DIR, "salary_bagging_pipeline.joblib")
    gbm_c_path = os.path.join(MODEL_DIR, "placement_gbm_pipeline.joblib")
    gbm_r_path = os.path.join(MODEL_DIR, "salary_gbm_pipeline.joblib")
    lgb_c_path = os.path.join(MODEL_DIR, "placement_lightgbm_pipeline.joblib")
    lgb_r_path = os.path.join(MODEL_DIR, "salary_lightgbm_pipeline.joblib")
    xgb_c_path = os.path.join(MODEL_DIR, "placement_xgb_pipeline.joblib")
    xgb_r_path = os.path.join(MODEL_DIR, "salary_xgb_pipeline.joblib")
    ada_c_path = os.path.join(MODEL_DIR, "placement_adaboost_pipeline.joblib")
    ada_r_path = os.path.join(MODEL_DIR, "salary_adaboost_pipeline.joblib")
    kmeans_path = os.path.join(MODEL_DIR, "kmeans_pipeline.joblib")
    pca_c_path = os.path.join(MODEL_DIR, "placement_pca_logistic_pipeline.joblib")
    pca_r_path = os.path.join(MODEL_DIR, "salary_pca_lr_pipeline.joblib")
    
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    
    required_paths = [
        lr_path, simple_lr_path, logistic_path, id3_c_path, id3_r_path,
        c45_c_path, c45_r_path, cart_c_path, cart_r_path,
        rf_c_path, rf_r_path, bag_c_path, bag_r_path,
        gbm_c_path, gbm_r_path, lgb_c_path, lgb_r_path,
        xgb_c_path, xgb_r_path, ada_c_path, ada_r_path, kmeans_path,
        pca_c_path, pca_r_path, meta_path
    ]
    
    if not all(os.path.exists(p) for p in required_paths):
        train_and_evaluate_models()
        
    lr_pipeline = joblib.load(lr_path)
    simple_lr = joblib.load(simple_lr_path)
    logistic_pipeline = joblib.load(logistic_path)
    
    id3_c_pipeline = joblib.load(id3_c_path)
    id3_r_pipeline = joblib.load(id3_r_path)
    c45_c_pipeline = joblib.load(c45_c_path)
    c45_r_pipeline = joblib.load(c45_r_path)
    cart_c_pipeline = joblib.load(cart_c_path)
    cart_r_pipeline = joblib.load(cart_r_path)
    
    dt_c_pipeline = joblib.load(dt_c_path) if os.path.exists(dt_c_path) else cart_c_pipeline
    dt_r_pipeline = joblib.load(dt_r_path) if os.path.exists(dt_r_path) else cart_r_pipeline
    
    rf_c_pipeline = joblib.load(rf_c_path)
    rf_r_pipeline = joblib.load(rf_r_path)
    bag_c_pipeline = joblib.load(bag_c_path)
    bag_r_pipeline = joblib.load(bag_r_path)
    gbm_c_pipeline = joblib.load(gbm_c_path)
    gbm_r_pipeline = joblib.load(gbm_r_path)
    lgb_c_pipeline = joblib.load(lgb_c_path)
    lgb_r_pipeline = joblib.load(lgb_r_path)
    xgb_c_pipeline = joblib.load(xgb_c_path)
    xgb_r_pipeline = joblib.load(xgb_r_path)
    ada_c_pipeline = joblib.load(ada_c_path)
    ada_r_pipeline = joblib.load(ada_r_path)
    kmeans_pipeline = joblib.load(kmeans_path) if os.path.exists(kmeans_path) else None
    pca_c_pipeline = joblib.load(pca_c_path) if os.path.exists(pca_c_path) else None
    pca_r_pipeline = joblib.load(pca_r_path) if os.path.exists(pca_r_path) else None
    
    with open(meta_path, "r") as f:
        metadata = json.load(f)
        
    ensemble_pipelines = {
        "id3_c": id3_c_pipeline,
        "id3_r": id3_r_pipeline,
        "c45_c": c45_c_pipeline,
        "c45_r": c45_r_pipeline,
        "cart_c": cart_c_pipeline,
        "cart_r": cart_r_pipeline,
        "rf_c": rf_c_pipeline,
        "rf_r": rf_r_pipeline,
        "bag_c": bag_c_pipeline,
        "bag_r": bag_r_pipeline,
        "gbm_c": gbm_c_pipeline,
        "gbm_r": gbm_r_pipeline,
        "lgb_c": lgb_c_pipeline,
        "lgb_r": lgb_r_pipeline,
        "xgb_c": xgb_c_pipeline,
        "xgb_r": xgb_r_pipeline,
        "ada_c": ada_c_pipeline,
        "ada_r": ada_r_pipeline,
        "kmeans": kmeans_pipeline,
        "pca_c": pca_c_pipeline,
        "pca_r": pca_r_pipeline
    }
        
    return lr_pipeline, simple_lr, logistic_pipeline, dt_c_pipeline, dt_r_pipeline, ensemble_pipelines, metadata


if __name__ == "__main__":
    train_and_evaluate_models()
