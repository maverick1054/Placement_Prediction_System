"""
Placement Analytics & Machine Learning Platform Entrypoint Script
Run this script to train/update Logistic & Linear Regression models and launch the Streamlit Dashboard.
"""

import sys
import subprocess
from model_trainer import train_and_evaluate_models

def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 65)
    print("🎓 PLACEMENT ANALYTICS — LOGISTIC & LINEAR REGRESSION PLATFORM")
    print("=" * 65)
    
    print("\n[1/2] Training & Evaluating Machine Learning Models...")
    metadata = train_and_evaluate_models()
    
    if 'logistic_metrics' in metadata:
        log_m = metadata['logistic_metrics']
        print(f"\n--- [1] Logistic Regression Classifier (PlacementStatus) ---")
        print(f"  * Accuracy : {log_m['accuracy'] * 100:.2f}%")
        print(f"  * Precision: {log_m['precision'] * 100:.2f}%")
        print(f"  * Recall   : {log_m['recall'] * 100:.2f}%")
        print(f"  * F1 Score : {log_m['f1_score']:.4f}")
        print(f"  * ROC-AUC  : {log_m['roc_auc']:.4f}")

    if 'id3_classifier_metrics' in metadata:
        id3_c = metadata['id3_classifier_metrics']
        print(f"\n--- [2a] ID3 Decision Tree Classifier (Entropy / Info Gain) ---")
        print(f"  * Accuracy : {id3_c['accuracy'] * 100:.2f}% | F1: {id3_c['f1_score']:.4f} | ROC-AUC: {id3_c['roc_auc']:.4f}")

    if 'c45_classifier_metrics' in metadata:
        c45_c = metadata['c45_classifier_metrics']
        print(f"\n--- [2b] C4.5 Decision Tree Classifier (Gain Ratio & Pruned) ---")
        print(f"  * Accuracy : {c45_c['accuracy'] * 100:.2f}% | F1: {c45_c['f1_score']:.4f} | ROC-AUC: {c45_c['roc_auc']:.4f}")

    if 'cart_classifier_metrics' in metadata:
        cart_c = metadata['cart_classifier_metrics']
        print(f"\n--- [2c] CART (Karth) Decision Tree Classifier (Gini Impurity) ---")
        print(f"  * Accuracy : {cart_c['accuracy'] * 100:.2f}% | F1: {cart_c['f1_score']:.4f} | ROC-AUC: {cart_c['roc_auc']:.4f}")

    if 'id3_regressor_metrics' in metadata:
        id3_r = metadata['id3_regressor_metrics']
        print(f"\n--- [8a] ID3 Decision Tree Regressor (MAE) ---")
        print(f"  * R2 Score : {id3_r['r2_score']:.4f} | MAE: Rs. {id3_r['mae']:.2f} LPA")

    if 'c45_regressor_metrics' in metadata:
        c45_r = metadata['c45_regressor_metrics']
        print(f"\n--- [8b] C4.5 Decision Tree Regressor (Friedman MSE) ---")
        print(f"  * R2 Score : {c45_r['r2_score']:.4f} | MAE: Rs. {c45_r['mae']:.2f} LPA")

    if 'cart_regressor_metrics' in metadata:
        cart_r = metadata['cart_regressor_metrics']
        print(f"\n--- [8c] CART (Karth) Decision Tree Regressor (MSE) ---")
        print(f"  * R2 Score : {cart_r['r2_score']:.4f} | MAE: Rs. {cart_r['mae']:.2f} LPA")

    if 'gbm_regressor_metrics' in metadata:
        gbm_r = metadata['gbm_regressor_metrics']
        print(f"\n--- [9] Standard GBM Regressor ---")
        print(f"  * R2 Score : {gbm_r['r2_score']:.4f} | MAE: Rs. {gbm_r['mae']:.2f} LPA")

    if 'lightgbm_regressor_metrics' in metadata:
        lgb_r = metadata['lightgbm_regressor_metrics']
        print(f"\n--- [10] LightGBM Regressor ---")
        print(f"  * R2 Score : {lgb_r['r2_score']:.4f} | MAE: Rs. {lgb_r['mae']:.2f} LPA")

    if 'xgb_regressor_metrics' in metadata:
        xgb_r = metadata['xgb_regressor_metrics']
        print(f"\n--- [11] XGBoost Regressor ---")
        print(f"  * R2 Score : {xgb_r['r2_score']:.4f} | MAE: Rs. {xgb_r['mae']:.2f} LPA")

    if 'simple_lr_metrics' in metadata:
        m = metadata['simple_lr_metrics']
        print(f"\n--- [12] Simple Linear Regression (CGPA -> Salary Package) ---")
        print(f"  * Equation : Salary = {m['slope']:.2f} * CGPA + ({m['intercept']:.2f})")
        print(f"  * R2 Score : {m['r2_score']:.4f} (Correlation r = {m['correlation']:.4f})")

    if 'kmeans_metrics' in metadata:
        km_m = metadata['kmeans_metrics']
        best_k = km_m.get('best_k', 3)
        best_sil = km_m.get('best_silhouette', 0.0)
        print(f"\n--- [13] KMeans Unsupervised Clustering & Silhouette Analysis ---")
        print(f"  * Optimal k Clusters : {best_k}")
        print(f"  * Peak Silhouette Score: {best_sil:.4f}")
        print(f"  * 2D PCA Explained Var : {km_m.get('pca_variance_explained_2d', 0):.2f}%")

    if 'pca_metrics' in metadata:
        pca_m = metadata['pca_metrics']
        pca_c_m = pca_m.get("pca_logistic_metrics", {})
        pca_r_m = pca_m.get("pca_lr_metrics", {})
        print(f"\n--- [14] Principal Component Analysis (PCA) & Dimensionality Reduction ---")
        print(f"  * Total Original Features   : {pca_m.get('n_total_features', 0)}")
        print(f"  * Components for 95% Var    : {pca_m.get('n_comp_95', 0)}")
        print(f"  * PC1 & PC2 Variance        : {pca_m.get('pc1_var_pct', 0):.2f}% & {pca_m.get('pc2_var_pct', 0):.2f}%")
        print(f"  * PCA + Logistic Classifier : Acc {pca_c_m.get('accuracy', 0)*100:.2f}% | F1 {pca_c_m.get('f1_score', 0):.4f} | ROC-AUC {pca_c_m.get('roc_auc', 0):.4f}")
        print(f"  * PCA + Linear Regressor    : R2 {pca_r_m.get('r2_score', 0):.4f} | MAE Rs. {pca_r_m.get('mae', 0):.2f} LPA")
    
    print("\n" + "=" * 65)
    print("[2/2] Launching Streamlit Interactive Dashboard...")
    print("Opening browser at http://localhost:8501 ...")
    print("=" * 65)
    
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()