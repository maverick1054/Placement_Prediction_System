"""
KMeans & Multi-Model Unsupervised Clustering with Mixed Feature Categorical Encoding Analytics Module
Placement & Salary Analytics System
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score
)
from utils import load_data, COLOR_PRIMARY, COLOR_PLACED, COLOR_NOT_PLACED, COLOR_NEUTRAL, PLOT_BG, GRID_COLOR

MODEL_DIR = "saved_models"


def build_clustering_preprocessor(num_cols, cat_cols, encoding_strategy="onehot"):
    """
    Constructs a scikit-learn ColumnTransformer for mixed numerical and categorical features.
    Supports One-Hot Encoding ('onehot'), Ordinal Encoding ('ordinal'), or Combined Encoding ('combined').
    """
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    if encoding_strategy == "ordinal":
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ("scaler", StandardScaler())
        ])
    else:  # 'onehot' (Default / Recommended for unbiased distance metrics)
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])
    return preprocessor


def compute_cluster_categorical_profiles(df, labels, cat_cols):
    """
    Calculates the top category mode and distribution percentage share (%) per cluster
    for all categorical attributes (e.g. Gender, CollegeTier, Stream, Specialisation).
    """
    if len(df) != len(labels):
        df_temp = df.iloc[:len(labels)].copy()
    else:
        df_temp = df.copy()
    df_temp["Cluster"] = labels
    
    unique_clusters = sorted(df_temp["Cluster"].unique())
    cat_profiles = []
    
    for c in unique_clusters:
        c_df = df_temp[df_temp["Cluster"] == c]
        total_c = len(c_df)
        c_record = {"Cluster ID": f"Cluster {c}", "Student Count": total_c}
        
        for col in cat_cols:
            if col in c_df.columns:
                top_mode = c_df[col].mode()[0] if not c_df[col].dropna().empty else "N/A"
                mode_cnt = (c_df[col] == top_mode).sum()
                mode_pct = round((mode_cnt / total_c) * 100, 1) if total_c > 0 else 0.0
                c_record[f"{col} (Top Mode)"] = f"{top_mode} ({mode_pct}%)"
                
        cat_profiles.append(c_record)
        
    return pd.DataFrame(cat_profiles)


def compute_silhouette_intervals(X_scaled, labels, sample_size=3000, random_state=42):
    """
    Calculates detailed silhouette score statistics, confidence intervals (95%), 
    and quantile intervals [Min, Q25, Median, Q75, Max] overall and per cluster.
    """
    if len(X_scaled) > sample_size:
        np.random.seed(random_state)
        idx = np.random.choice(len(X_scaled), size=sample_size, replace=False)
        X_sub = X_scaled[idx]
        labels_sub = labels[idx]
    else:
        X_sub = X_scaled
        labels_sub = labels

    sil_samples = silhouette_samples(X_sub, labels_sub)
    overall_mean = float(np.mean(sil_samples))
    overall_std = float(np.std(sil_samples))
    overall_sem = overall_std / np.sqrt(len(sil_samples))
    
    unique_clusters = sorted(np.unique(labels_sub))
    per_cluster_stats = []
    
    for c in unique_clusters:
        c_vals = sil_samples[labels_sub == c]
        if len(c_vals) == 0:
            continue
        c_mean = float(np.mean(c_vals))
        c_std = float(np.std(c_vals)) if len(c_vals) > 1 else 0.0
        c_sem = c_std / np.sqrt(len(c_vals)) if len(c_vals) > 1 else 0.0
        
        ci_low = max(-1.0, c_mean - 1.96 * c_sem)
        ci_high = min(1.0, c_mean + 1.96 * c_sem)
        
        per_cluster_stats.append({
            "Cluster": f"Cluster {c}",
            "Cluster_ID": int(c),
            "Count": int(len(c_vals)),
            "Mean_Silhouette": round(c_mean, 4),
            "Std_Dev": round(c_std, 4),
            "CI_95_Lower": round(ci_low, 4),
            "CI_95_Upper": round(ci_high, 4),
            "Min": round(float(np.min(c_vals)), 4),
            "Q25": round(float(np.percentile(c_vals, 25)), 4),
            "Median": round(float(np.median(c_vals)), 4),
            "Q75": round(float(np.percentile(c_vals, 75)), 4),
            "Max": round(float(np.max(c_vals)), 4)
        })
        
    return {
        "overall_mean": round(overall_mean, 4),
        "overall_std": round(overall_std, 4),
        "overall_ci_95": [
            round(max(-1.0, overall_mean - 1.96 * overall_sem), 4),
            round(min(1.0, overall_mean + 1.96 * overall_sem), 4)
        ],
        "per_cluster_stats": per_cluster_stats
    }


def compute_cluster_feature_intervals(df_num, labels, confidence=0.95):
    """
    Computes 95% Confidence Intervals and Interquartile Ranges (Q25, Median, Q75)
    for every numerical feature across all clusters.
    """
    if len(df_num) != len(labels):
        df_temp = df_num.iloc[:len(labels)].copy()
    else:
        df_temp = df_num.copy()
    df_temp["Cluster"] = labels
    
    unique_clusters = sorted(df_temp["Cluster"].unique())
    records = []
    
    for feat in df_num.columns:
        for c in unique_clusters:
            vals = df_temp[df_temp["Cluster"] == c][feat].dropna().values
            if len(vals) == 0:
                continue
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals)) if len(vals) > 1 else 0.0
            sem_val = std_val / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            
            h = sem_val * stats.t.ppf((1 + confidence) / 2., max(1, len(vals)-1)) if len(vals) > 1 else 0.0
            ci_lower = mean_val - h
            ci_upper = mean_val + h
            
            q25 = float(np.percentile(vals, 25))
            median_val = float(np.median(vals))
            q75 = float(np.percentile(vals, 75))
            
            records.append({
                "Feature": feat,
                "Cluster": f"Cluster {c}",
                "Cluster_ID": int(c),
                "Count": int(len(vals)),
                "Mean": round(mean_val, 2),
                "Std": round(std_val, 2),
                "CI_Lower_95": round(ci_lower, 2),
                "CI_Upper_95": round(ci_upper, 2),
                "Q25": round(q25, 2),
                "Median": round(median_val, 2),
                "Q75": round(q75, 2),
                "IQR": round(q75 - q25, 2)
            })
            
    return pd.DataFrame(records)


def process_unlabelled_data(custom_df=None, pipeline=None):
    """
    Processes unlabelled dataset or strips label columns (PlacementStatus, Salary Package)
    to perform pure unsupervised discovery, pseudo-labeling, and cluster breakdown.
    """
    df = custom_df.copy() if custom_df is not None else load_data()
    
    label_cols = [c for c in ["PlacementStatus", "Salary Package"] if c in df.columns]
    unlabelled_mask = df[label_cols].isna().any(axis=1) if label_cols else pd.Series(True, index=df.index)
    
    drop_cols = ["StudentID", "IsAnomaly"] + label_cols
    cat_cols = [c for c in ["Gender", "City", "CollegeTier", "Stream", "Specialisation", "Hostel", "HistoryOfBacklogs", "CGPA_Tier"] if c in df.columns]
    num_cols = [c for c in df.columns if c not in drop_cols and c not in cat_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    if pipeline is None:
        preprocessor = build_clustering_preprocessor(num_cols, cat_cols, encoding_strategy="onehot")
        X_processed = preprocessor.fit_transform(df[num_cols + cat_cols])
        km = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42)
        pseudo_labels = km.fit_predict(X_processed)
    else:
        pseudo_labels = pipeline.predict(df)
        
    df_unlabelled = df.copy()
    df_unlabelled["Predicted_Cluster"] = pseudo_labels
    
    unlabelled_summary = {
        "total_records": int(len(df)),
        "unlabelled_records_count": int(unlabelled_mask.sum()) if label_cols else int(len(df)),
        "feature_count": int(len(num_cols) + len(cat_cols)),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "pseudo_cluster_counts": {f"Cluster {c}": int(cnt) for c, cnt in zip(*np.unique(pseudo_labels, return_counts=True))}
    }
    
    return df_unlabelled, unlabelled_summary


def train_gmm_clustering(X_scaled, n_clusters=3, random_state=42):
    """
    Fits Gaussian Mixture Model (GMM) soft clustering to produce cluster assignment probabilities P(Cluster_k | x).
    Calculates BIC, AIC, and log-likelihood.
    """
    gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=random_state)
    gmm.fit(X_scaled)
    
    probs = gmm.predict_proba(X_scaled)
    hard_labels = probs.argmax(axis=1)
    
    bic = float(gmm.bic(X_scaled))
    aic = float(gmm.aic(X_scaled))
    log_likelihood = float(gmm.score(X_scaled) * len(X_scaled))
    
    sil = float(silhouette_score(X_scaled, hard_labels, sample_size=min(3000, len(X_scaled)), random_state=random_state))
    db = float(davies_bouldin_score(X_scaled, hard_labels))
    ch = float(calinski_harabasz_score(X_scaled, hard_labels))
    
    return {
        "model": gmm,
        "labels": hard_labels,
        "probs": probs,
        "bic": round(bic, 2),
        "aic": round(aic, 2),
        "log_likelihood": round(log_likelihood, 2),
        "silhouette_score": round(sil, 4),
        "davies_bouldin": round(db, 4),
        "calinski_harabasz": round(ch, 2)
    }


def compute_cluster_overlap_analysis(probs, margin_threshold=0.20, entropy_threshold=0.60):
    """
    Quantifies cluster overlap and boundary ambiguity using Membership Entropy
    H = -sum(p_k * ln(p_k)) / ln(K) and Probability Margin Index (top1_prob - top2_prob).
    """
    n_samples, n_clusters = probs.shape
    
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    top1_p = sorted_probs[:, 0]
    top2_p = sorted_probs[:, 1]
    margin = top1_p - top2_p
    
    eps = 1e-12
    entropy = -np.sum(probs * np.log(probs + eps), axis=1) / np.log(n_clusters)
    
    is_overlapping = (margin < margin_threshold) | (entropy > entropy_threshold)
    overlap_count = int(is_overlapping.sum())
    overlap_rate = round(float((overlap_count / n_samples) * 100), 2)
    
    return {
        "n_samples": n_samples,
        "n_clusters": n_clusters,
        "overlap_count": overlap_count,
        "overlap_rate_percent": overlap_rate,
        "avg_entropy": round(float(np.mean(entropy)), 4),
        "avg_margin": round(float(np.mean(margin)), 4),
        "entropy_series": entropy,
        "margin_series": margin,
        "is_overlapping": is_overlapping
    }


def train_hierarchical_clustering(X_scaled, n_clusters=3, linkage="ward", sample_size=3000, random_state=42):
    """
    Fits Agglomerative Hierarchical Clustering with Ward linkage.
    Sub-samples dataset if N > sample_size to prevent MemoryError on large pairwise distance matrices.
    """
    if len(X_scaled) > sample_size:
        np.random.seed(random_state)
        idx = np.random.choice(len(X_scaled), size=sample_size, replace=False)
        X_sub = X_scaled[idx]
    else:
        X_sub = X_scaled

    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = agg.fit_predict(X_sub)
    
    sil = float(silhouette_score(X_sub, labels, sample_size=min(3000, len(X_sub)), random_state=random_state))
    db = float(davies_bouldin_score(X_sub, labels))
    ch = float(calinski_harabasz_score(X_sub, labels))
    
    return {
        "labels": labels,
        "X_sub": X_sub,
        "silhouette_score": round(sil, 4),
        "davies_bouldin": round(db, 4),
        "calinski_harabasz": round(ch, 2)
    }


def train_dbscan_clustering(X_scaled, eps=1.2, min_samples=10, sample_size=3000, random_state=42):
    """
    Fits DBSCAN density-based clustering to isolate noise/outliers (-1) and discover non-spherical shapes.
    Sub-samples dataset if N > sample_size to prevent MemoryError.
    """
    if len(X_scaled) > sample_size:
        np.random.seed(random_state)
        idx = np.random.choice(len(X_scaled), size=sample_size, replace=False)
        X_sub = X_scaled[idx]
    else:
        X_sub = X_scaled

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_sub)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = int((labels == -1).sum())
    noise_ratio = round(float((noise_count / len(X_sub)) * 100), 2)
    
    if n_clusters > 1:
        mask = labels != -1
        sil = float(silhouette_score(X_sub[mask], labels[mask], sample_size=min(3000, mask.sum()), random_state=random_state)) if mask.sum() > n_clusters else 0.0
        db = float(davies_bouldin_score(X_sub[mask], labels[mask])) if mask.sum() > n_clusters else 0.0
        ch = float(calinski_harabasz_score(X_sub[mask], labels[mask])) if mask.sum() > n_clusters else 0.0
    else:
        sil, db, ch = 0.0, 0.0, 0.0
        
    return {
        "labels": labels,
        "X_sub": X_sub,
        "n_clusters": n_clusters,
        "noise_count": noise_count,
        "noise_ratio_percent": noise_ratio,
        "silhouette_score": round(sil, 4),
        "davies_bouldin": round(db, 4),
        "calinski_harabasz": round(ch, 2)
    }


def benchmark_all_clustering_models(X_scaled, n_clusters=3, random_state=42):
    """
    Benchmarks KMeans, GMM, Agglomerative Hierarchical, and DBSCAN side-by-side on encoded feature space.
    """
    km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=random_state)
    km_labels = km.fit_predict(X_scaled)
    km_sil = float(silhouette_score(X_scaled, km_labels, sample_size=min(3000, len(X_scaled)), random_state=random_state))
    km_db = float(davies_bouldin_score(X_scaled, km_labels))
    km_ch = float(calinski_harabasz_score(X_scaled, km_labels))
    
    gmm_res = train_gmm_clustering(X_scaled, n_clusters=n_clusters, random_state=random_state)
    hier_res = train_hierarchical_clustering(X_scaled, n_clusters=n_clusters, linkage="ward", random_state=random_state)
    dbs_res = train_dbscan_clustering(X_scaled, eps=1.5, min_samples=10, random_state=random_state)
    
    benchmark_records = [
        {
            "Algorithm": "K-Means Clustering",
            "Clustering Type": "Partition-based (Hard)",
            "Clusters (k)": n_clusters,
            "Silhouette Score": round(km_sil, 4),
            "Davies-Bouldin Index": round(km_db, 4),
            "Calinski-Harabasz Index": round(km_ch, 2),
            "Noise / Overlap Info": "Non-overlapping centroids"
        },
        {
            "Algorithm": "Gaussian Mixture Model (GMM)",
            "Clustering Type": "Probabilistic (Soft/Overlapping)",
            "Clusters (k)": n_clusters,
            "Silhouette Score": round(gmm_res["silhouette_score"], 4),
            "Davies-Bouldin Index": round(gmm_res["davies_bouldin"], 4),
            "Calinski-Harabasz Index": round(gmm_res["calinski_harabasz"], 2),
            "Noise / Overlap Info": f"BIC: {gmm_res['bic']:,} | AIC: {gmm_res['aic']:,}"
        },
        {
            "Algorithm": "Agglomerative Hierarchical",
            "Clustering Type": "Tree-structured Hierarchy",
            "Clusters (k)": n_clusters,
            "Silhouette Score": round(hier_res["silhouette_score"], 4),
            "Davies-Bouldin Index": round(hier_res["davies_bouldin"], 4),
            "Calinski-Harabasz Index": round(hier_res["calinski_harabasz"], 2),
            "Noise / Overlap Info": "Ward Linkage Distance"
        },
        {
            "Algorithm": "DBSCAN Density-Based",
            "Clustering Type": "Density-based Spatial",
            "Clusters (k)": dbs_res["n_clusters"],
            "Silhouette Score": round(dbs_res["silhouette_score"], 4),
            "Davies-Bouldin Index": round(dbs_res["davies_bouldin"], 4),
            "Calinski-Harabasz Index": round(dbs_res["calinski_harabasz"], 2),
            "Noise / Overlap Info": f"Noise: {dbs_res['noise_count']} ({dbs_res['noise_ratio_percent']}%)"
        }
    ]
    
    return benchmark_records


def train_and_evaluate_kmeans(custom_df=None, n_clusters_range=range(2, 11), encoding_strategy="onehot", random_state=42):
    """
    Trains mixed-feature KMeans (Numerical + Categorical Encoding), evaluates metrics across k=2..10,
    computes Silhouette intervals, feature confidence intervals, GMM soft clustering, DBSCAN, Hierarchical,
    and returns categorical & numerical cluster persona profiles.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = custom_df if custom_df is not None else load_data()
    
    drop_cols = ["StudentID", "IsAnomaly", "PlacementStatus", "Salary Package"]
    cat_cols = [c for c in ["Gender", "City", "CollegeTier", "Stream", "Specialisation", "Hostel", "HistoryOfBacklogs", "CGPA_Tier"] if c in df.columns]
    num_cols = [c for c in df.columns if c not in drop_cols and c not in cat_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    if not num_cols and not cat_cols:
        raise ValueError("No numerical or categorical features found for clustering.")

    preprocessor = build_clustering_preprocessor(num_cols, cat_cols, encoding_strategy=encoding_strategy)
    X_processed = preprocessor.fit_transform(df[num_cols + cat_cols])
    
    metrics_per_k = {}
    best_k = 3
    best_silhouette = -1.0
    
    for k in n_clusters_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=random_state)
        cluster_labels = km.fit_predict(X_processed)
        
        inertia = float(km.inertia_)
        sample_sz = min(3000, len(X_processed))
        sil_score = float(silhouette_score(X_processed, cluster_labels, sample_size=sample_sz, random_state=random_state))
        ch_score = float(calinski_harabasz_score(X_processed, cluster_labels))
        db_score = float(davies_bouldin_score(X_processed, cluster_labels))
        
        if sil_score > best_silhouette:
            best_silhouette = sil_score
            best_k = k
            
        metrics_per_k[str(k)] = {
            "k": k,
            "inertia": round(inertia, 2),
            "silhouette_score": round(sil_score, 4),
            "calinski_harabasz": round(ch_score, 2),
            "davies_bouldin": round(db_score, 4)
        }
    
    # Best KMeans model
    opt_km = KMeans(n_clusters=best_k, init="k-means++", n_init=10, max_iter=300, random_state=random_state)
    opt_labels = opt_km.fit_predict(X_processed)
    
    # PCA on full encoded space
    pca2 = PCA(n_components=2, random_state=random_state)
    pca2.fit(X_processed)
    var_exp_2d = float(np.sum(pca2.explained_variance_ratio_))
    
    pca3 = PCA(n_components=3, random_state=random_state)
    pca3.fit(X_processed)
    var_exp_3d = float(np.sum(pca3.explained_variance_ratio_))
    
    # Complete End-to-End Pipeline (Preprocessor + KMeans)
    kmeans_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("kmeans", opt_km)
    ])
    joblib.dump(kmeans_pipeline, os.path.join(MODEL_DIR, "kmeans_pipeline.joblib"))
    
    df_clustered = df.copy()
    df_clustered["Cluster"] = opt_labels
    
    cluster_profiles = []
    for c_id in sorted(df_clustered["Cluster"].unique()):
        c_df = df_clustered[df_clustered["Cluster"] == c_id]
        
        placement_rate = round(float((c_df["PlacementStatus"] == 1).mean() * 100), 2) if "PlacementStatus" in c_df.columns else 0.0
        avg_cgpa = round(float(c_df["CGPA"].mean()), 2) if "CGPA" in c_df.columns else 0.0
        avg_salary = round(float(c_df["Salary Package"].mean()), 2) if "Salary Package" in c_df.columns else 0.0
        avg_aptitude = round(float(c_df["AptitudeTestScore"].mean()), 2) if "AptitudeTestScore" in c_df.columns else 0.0
        avg_soft = round(float(c_df["SoftSkillsScore"].mean()), 2) if "SoftSkillsScore" in c_df.columns else 0.0
        
        if placement_rate >= 75 or avg_cgpa >= 8.0:
            persona = "🌟 High Achievers (Top Tier)"
        elif placement_rate >= 45 or avg_cgpa >= 6.5:
            persona = "📊 Moderate Performers (Mid Tier)"
        else:
            persona = "⚠️ Need Improvement (Support Required)"
            
        cluster_profiles.append({
            "Cluster ID": f"Cluster {c_id}",
            "Persona": persona,
            "Student Count": int(len(c_df)),
            "Placement Rate (%)": placement_rate,
            "Avg CGPA": avg_cgpa,
            "Avg Salary (LPA)": avg_salary,
            "Avg Aptitude Score": avg_aptitude,
            "Avg Soft Skills Score": avg_soft
        })

    best_elbow_k = find_elbow_k(metrics_per_k)
    
    sil_intervals = compute_silhouette_intervals(X_processed, opt_labels)
    feat_intervals_df = compute_cluster_feature_intervals(df[num_cols], opt_labels)
    
    gmm_res = train_gmm_clustering(X_processed, n_clusters=best_k, random_state=random_state)
    overlap_res = compute_cluster_overlap_analysis(gmm_res["probs"])
    
    benchmark_records = benchmark_all_clustering_models(X_processed, n_clusters=best_k, random_state=random_state)
    cat_profiles_df = compute_cluster_categorical_profiles(df, opt_labels, cat_cols)

    try:
        transformed_feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        transformed_feature_names = num_cols + cat_cols

    kmeans_metrics = {
        "all_feature_names": transformed_feature_names,
        "num_feature_names": num_cols,
        "cat_feature_names": cat_cols,
        "encoding_strategy": encoding_strategy,
        "best_k": best_k,
        "best_elbow_k": best_elbow_k,
        "best_silhouette": round(best_silhouette, 4),
        "pca_variance_explained_2d": round(var_exp_2d * 100, 2),
        "pca_variance_explained_3d": round(var_exp_3d * 100, 2),
        "metrics_per_k": metrics_per_k,
        "cluster_profiles": cluster_profiles,
        "categorical_cluster_profiles": cat_profiles_df.to_dict(orient="records"),
        "silhouette_intervals": sil_intervals,
        "feature_confidence_intervals": feat_intervals_df.to_dict(orient="records"),
        "gmm_metrics": {
            "bic": gmm_res["bic"],
            "aic": gmm_res["aic"],
            "log_likelihood": gmm_res["log_likelihood"],
            "silhouette_score": gmm_res["silhouette_score"],
            "overlap_count": overlap_res["overlap_count"],
            "overlap_rate_percent": overlap_res["overlap_rate_percent"],
            "avg_entropy": overlap_res["avg_entropy"],
            "avg_margin": overlap_res["avg_margin"]
        },
        "benchmark_comparison": benchmark_records
    }
    
    return kmeans_pipeline, kmeans_metrics


def find_elbow_k(metrics_per_k):
    """
    Calculates the optimal 'elbow' point (k) from the Elbow Curve (Inertia vs k)
    using the maximum distance to the secant line algorithm (Kneedle method).
    """
    ks = np.array([int(k) for k in metrics_per_k.keys()])
    inertias = np.array([metrics_per_k[str(k)]["inertia"] for k in ks])
    
    if len(ks) < 3:
        return int(ks[0])
        
    x_norm = (ks - ks.min()) / (ks.max() - ks.min())
    y_norm = (inertias - inertias.min()) / (inertias.max() - inertias.min())
    
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    v = p2 - p1
    v_norm = np.linalg.norm(v)
    
    distances = []
    for i in range(len(ks)):
        p = np.array([x_norm[i], y_norm[i]])
        w = p - p1
        cross_mag = np.abs(v[0] * w[1] - v[1] * w[0])
        dist = cross_mag / v_norm if v_norm > 0 else 0.0
        distances.append(dist)
        
    elbow_idx = np.argmax(distances)
    return int(ks[elbow_idx])


# ==================== PLOTLY VISUALIZATION HELPERS ====================

def plot_cluster_categorical_distribution(df, labels, cat_col):
    """
    Renders stacked bar chart showing distribution of levels for selected categorical attribute across clusters.
    """
    if len(df) != len(labels):
        df_temp = df.iloc[:len(labels)].copy()
    else:
        df_temp = df.copy()
    df_temp["Cluster"] = [f"Cluster {c}" for c in labels]
    
    ct = pd.crosstab(df_temp["Cluster"], df_temp[cat_col], normalize="index") * 100
    
    fig = go.Figure()
    colors = px.colors.qualitative.Bold
    
    for idx, category_level in enumerate(ct.columns):
        fig.add_trace(go.Bar(
            x=ct.index,
            y=ct[category_level],
            name=str(category_level),
            marker=dict(color=colors[idx % len(colors)]),
            text=[f"{v:.1f}%" for v in ct[category_level]],
            textposition="auto",
            hovertemplate=f"<b>%{{x}}</b><br>{cat_col}: {category_level}<br>Share: %{{y:.1f}}%<extra></extra>"
        ))
        
    fig.update_layout(
        title=dict(text=f"🏷️ <b>Categorical Attribute Distribution Share (%): {cat_col}</b>", font=dict(size=16)),
        barmode="stack",
        xaxis=dict(title="Cluster Segment", gridcolor=GRID_COLOR),
        yaxis=dict(title="Percentage Share (%)", range=[0, 105], gridcolor=GRID_COLOR),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_elbow_curve(metrics_per_k, selected_k=3):
    """
    Renders an interactive Plotly Elbow Curve (Inertia vs k) with Kneedle elbow marker.
    """
    ks = [int(k) for k in metrics_per_k.keys()]
    inertias = [metrics_per_k[str(k)]["inertia"] for k in ks]
    
    elbow_k = find_elbow_k(metrics_per_k)
    elbow_inertia = metrics_per_k[str(elbow_k)]["inertia"]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=ks,
        y=inertias,
        mode="lines+markers",
        marker=dict(size=9, color=COLOR_PRIMARY),
        line=dict(width=3, color=COLOR_PRIMARY),
        name="Inertia (WCSS)",
        hovertemplate="<b>k-value = %{x}</b><br>Inertia: %{y:,.2f}<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=[ks[0], ks[-1]],
        y=[inertias[0], inertias[-1]],
        mode="lines",
        line=dict(width=1.5, color="#94a3b8", dash="dash"),
        name="Chord Line (Secant)",
        hoverinfo="skip"
    ))
    
    fig.add_trace(go.Scatter(
        x=[elbow_k],
        y=[elbow_inertia],
        mode="markers+text",
        marker=dict(size=16, color="#d97706", symbol="star"),
        text=[f"  🎯 Best k (Elbow = {elbow_k})"],
        textposition="top right",
        name=f"Optimal Best k (Elbow = {elbow_k})",
        hovertemplate=f"<b>Optimal Best k={elbow_k}</b><br>Inertia: {elbow_inertia:,.2f}<extra></extra>"
    ))
    
    fig.add_vline(
        x=elbow_k,
        line_dash="dot",
        line_color="#d97706",
        line_width=2,
        annotation_text=f"Best k = {elbow_k}",
        annotation_position="bottom right"
    )

    if str(selected_k) in metrics_per_k and selected_k != elbow_k:
        sel_inertia = metrics_per_k[str(selected_k)]["inertia"]
        fig.add_trace(go.Scatter(
            x=[selected_k],
            y=[sel_inertia],
            mode="markers",
            marker=dict(size=14, color=COLOR_NOT_PLACED, symbol="diamond"),
            name=f"Selected k = {selected_k}",
            hovertemplate=f"<b>Selected k-value = {selected_k}</b><br>Inertia: {sel_inertia:,.2f}<extra></extra>"
        ))
        
    fig.update_layout(
        title=dict(text="📉 <b>KMeans Elbow Curve (X-axis: k-value, Y-axis: WCSS Inertia)</b>", font=dict(size=16)),
        xaxis=dict(title="Number of Clusters (k-value)", tickmode="linear", dtick=1, gridcolor=GRID_COLOR),
        yaxis=dict(title="Inertia (Within-Cluster Sum of Squares)", gridcolor=GRID_COLOR),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_silhouette_curve(metrics_per_k, selected_k=3):
    """
    Renders an interactive Plotly Silhouette Score Curve (Silhouette Score vs k).
    """
    ks = [int(k) for k in metrics_per_k.keys()]
    sil_scores = [metrics_per_k[str(k)]["silhouette_score"] for k in ks]
    
    best_k_idx = np.argmax(sil_scores)
    best_k = ks[best_k_idx]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=ks,
        y=sil_scores,
        mode="lines+markers",
        marker=dict(size=9, color="#059669"),
        line=dict(width=3, color="#059669"),
        name="Silhouette Score",
        hovertemplate="<b>k = %{x}</b><br>Silhouette Score: %{y:.4f}<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=[best_k],
        y=[sil_scores[best_k_idx]],
        mode="markers",
        marker=dict(size=14, color="#d97706", symbol="diamond"),
        name=f"Optimal Peak k={best_k}",
        hovertemplate=f"<b>Optimal Peak k={best_k}</b><br>Silhouette Score: {sil_scores[best_k_idx]:.4f}<extra></extra>"
    ))

    if str(selected_k) in metrics_per_k:
        sel_score = metrics_per_k[str(selected_k)]["silhouette_score"]
        fig.add_trace(go.Scatter(
            x=[selected_k],
            y=[sel_score],
            mode="markers",
            marker=dict(size=14, color=COLOR_NOT_PLACED, symbol="star"),
            name=f"Selected k={selected_k}",
            hovertemplate=f"<b>Selected k={selected_k}</b><br>Silhouette Score: {sel_score:.4f}<extra></extra>"
        ))
        
    fig.update_layout(
        title=dict(text="📊 <b>Silhouette Score Trend Across Cluster Sizes (k)</b>", font=dict(size=16)),
        xaxis=dict(title="Number of Clusters (k)", tickmode="linear", dtick=1, gridcolor=GRID_COLOR),
        yaxis=dict(title="Silhouette Coefficient S", range=[-0.1, 1.0], gridcolor=GRID_COLOR),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_silhouette_profile(X_scaled, labels, n_clusters, sample_size=2000, random_state=42):
    """
    Renders detailed Silhouette Profile Plot (cluster-by-cluster silhouette coefficients).
    """
    if len(X_scaled) > sample_size:
        np.random.seed(random_state)
        indices = np.random.choice(len(X_scaled), size=sample_size, replace=False)
        X_sample = X_scaled[indices]
        labels_sample = labels[indices]
    else:
        X_sample = X_scaled
        labels_sample = labels
        
    sil_values = silhouette_samples(X_sample, labels_sample)
    avg_sil = silhouette_score(X_sample, labels_sample)
    
    fig = go.Figure()
    colors = px.colors.qualitative.Bold
    
    y_lower = 10
    for i in range(n_clusters):
        ith_sil_values = sil_values[labels_sample == i]
        ith_sil_values.sort()
        
        size_i = ith_sil_values.shape[0]
        y_upper = y_lower + size_i
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=ith_sil_values,
            y=np.arange(y_lower, y_upper),
            mode="lines",
            fill="tozerox",
            line=dict(width=0.5, color=color),
            name=f"Cluster {i}",
            hovertemplate=f"<b>Cluster {i}</b><br>Silhouette Val: %{{x:.4f}}<extra></extra>"
        ))
        
        y_lower = y_upper + 10
        
    fig.add_vline(
        x=avg_sil,
        line_dash="dash",
        line_color="#dc2626",
        line_width=2,
        annotation_text=f"Mean S = {avg_sil:.4f}",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title=dict(text=f"📐 <b>Silhouette Coefficient Profile per Cluster (k = {n_clusters})</b>", font=dict(size=16)),
        xaxis=dict(title="Silhouette Coefficient Value", range=[-0.2, 1.0], gridcolor=GRID_COLOR),
        yaxis=dict(title="Sample Index Grouped by Cluster", showticklabels=False, gridcolor=GRID_COLOR),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_silhouette_intervals_chart(sil_interval_data):
    """
    Renders an error-bar and box interval chart showing Silhouette ranges [Q25, Median, Q75]
    and 95% Confidence Intervals per cluster.
    """
    df_stats = pd.DataFrame(sil_interval_data["per_cluster_stats"])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_stats["Cluster"],
        y=df_stats["Mean_Silhouette"],
        error_y=dict(
            type="data",
            symmetric=False,
            array=df_stats["CI_95_Upper"] - df_stats["Mean_Silhouette"],
            arrayminus=df_stats["Mean_Silhouette"] - df_stats["CI_95_Lower"],
            color="#2563eb",
            thickness=2,
            width=6
        ),
        mode="markers+lines",
        marker=dict(size=12, color="#2563eb", symbol="circle"),
        name="Mean Silhouette (95% CI Error Bars)",
        hovertemplate="<b>%{x}</b><br>Mean S: %{y:.4f}<br>95% CI: [%{customdata[0]:.4f}, %{customdata[1]:.4f}]<extra></extra>",
        customdata=df_stats[["CI_95_Lower", "CI_95_Upper"]].values
    ))
    
    fig.add_trace(go.Scatter(
        x=df_stats["Cluster"],
        y=df_stats["Median"],
        mode="markers",
        marker=dict(size=10, color="#d97706", symbol="diamond"),
        name="Median Silhouette Score",
        hovertemplate="<b>%{x}</b><br>Median: %{y:.4f}<br>Q25: %{customdata[0]:.4f} | Q75: %{customdata[1]:.4f}<extra></extra>",
        customdata=df_stats[["Q25", "Q75"]].values
    ))
    
    fig.add_hline(
        y=sil_interval_data["overall_mean"],
        line_dash="dash",
        line_color="#dc2626",
        annotation_text=f"Global Mean S = {sil_interval_data['overall_mean']:.4f}",
        annotation_position="bottom right"
    )
    
    fig.update_layout(
        title=dict(text="📏 <b>Cluster Silhouette Score Intervals & 95% Confidence Ranges</b>", font=dict(size=16)),
        xaxis=dict(title="Cluster Segment", gridcolor=GRID_COLOR),
        yaxis=dict(title="Silhouette Score Value", range=[-0.2, 1.0], gridcolor=GRID_COLOR),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_feature_confidence_intervals(feature_interval_records, feature_name):
    """
    Renders 95% Confidence Interval error bar chart for a selected feature across cluster centroids.
    """
    df_feat = pd.DataFrame(feature_interval_records)
    df_sub = df_feat[df_feat["Feature"] == feature_name]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_sub["Cluster"],
        y=df_sub["Mean"],
        error_y=dict(
            type="data",
            symmetric=False,
            array=df_sub["CI_Upper_95"] - df_sub["Mean"],
            arrayminus=df_sub["Mean"] - df_sub["CI_Lower_95"],
            color="#059669",
            thickness=2,
            width=6
        ),
        mode="markers+lines+text",
        marker=dict(size=12, color="#059669"),
        text=[f"  {m:.2f}" for m in df_sub["Mean"]],
        textposition="top right",
        name=f"{feature_name} Mean",
        hovertemplate=f"<b>%{{x}}</b><br>{feature_name} Mean: %{{y:.2f}}<br>95% CI: [%{{customdata[0]:.2f}}, %{{customdata[1]:.2f}}]<br>IQR [Q25, Q75]: [%{{customdata[2]:.2f}}, %{{customdata[3]:.2f}}]<extra></extra>",
        customdata=df_sub[["CI_Lower_95", "CI_Upper_95", "Q25", "Q75"]].values
    ))
    
    fig.update_layout(
        title=dict(text=f"🎯 <b>Centroid Feature Interval (95% CI & Mean): {feature_name}</b>", font=dict(size=16)),
        xaxis=dict(title="Cluster Segment", gridcolor=GRID_COLOR),
        yaxis=dict(title=f"Mean {feature_name} Score", gridcolor=GRID_COLOR),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_overlapping_clusters_scatter(df_processed, probs, labels, is_3d=False, sample_size=2500, random_state=42):
    """
    Renders 2D or 3D PCA scatter highlighting cluster overlap boundary zones across encoded feature space.
    """
    overlap_info = compute_cluster_overlap_analysis(probs)
    entropy = overlap_info["entropy_series"]
    margin = overlap_info["margin_series"]
    is_overlapping = overlap_info["is_overlapping"]
    
    if len(df_processed) > sample_size:
        np.random.seed(random_state)
        idx = np.random.choice(len(df_processed), size=sample_size, replace=False)
        X_sample = df_processed[idx] if isinstance(df_processed, np.ndarray) else df_processed.iloc[idx].values
        labels_sample = labels[idx]
        entropy_sample = entropy[idx]
        margin_sample = margin[idx]
        overlap_sample = is_overlapping[idx]
    else:
        X_sample = df_processed if isinstance(df_processed, np.ndarray) else df_processed.values
        labels_sample = labels
        entropy_sample = entropy
        margin_sample = margin
        overlap_sample = is_overlapping

    n_comp = 3 if is_3d else 2
    pca = PCA(n_components=n_comp, random_state=random_state)
    X_pca = pca.fit_transform(X_sample)
    
    colors = px.colors.qualitative.Bold
    fig = go.Figure()
    
    if is_3d:
        fig.add_trace(go.Scatter3d(
            x=X_pca[~overlap_sample, 0],
            y=X_pca[~overlap_sample, 1],
            z=X_pca[~overlap_sample, 2],
            mode="markers",
            marker=dict(size=4, color=labels_sample[~overlap_sample], colorscale="Jet", opacity=0.6),
            name="Core Cluster Points",
            hovertemplate="<b>Core Cluster %{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>",
            text=[f"{c}" for c in labels_sample[~overlap_sample]]
        ))
        
        if overlap_sample.sum() > 0:
            fig.add_trace(go.Scatter3d(
                x=X_pca[overlap_sample, 0],
                y=X_pca[overlap_sample, 1],
                z=X_pca[overlap_sample, 2],
                mode="markers",
                marker=dict(size=7, color="#dc2626", symbol="circle", opacity=0.9),
                name=f"⚠️ Overlapping / Boundary Zone ({overlap_sample.sum()} points)",
                hovertemplate="<b>⚠️ Overlapping Student Profile</b><br>Entropy: %{customdata[0]:.4f}<br>Margin: %{customdata[1]:.4f}<extra></extra>",
                customdata=np.column_stack((entropy_sample[overlap_sample], margin_sample[overlap_sample]))
            ))
            
        var_exp = np.sum(pca.explained_variance_ratio_) * 100
        fig.update_layout(
            title=dict(text=f"🌀 <b>3D Soft Overlapping Cluster Scatter (Explained Var: {var_exp:.1f}%)</b>", font=dict(size=16)),
            scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
            paper_bgcolor=PLOT_BG,
            margin=dict(l=20, r=20, t=50, b=20)
        )
    else:
        for c in np.unique(labels_sample):
            mask = (labels_sample == c) & (~overlap_sample)
            color = colors[c % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=X_pca[mask, 0],
                y=X_pca[mask, 1],
                mode="markers",
                marker=dict(size=6, color=color, opacity=0.6),
                name=f"Cluster {c} Core",
                hovertemplate=f"<b>Cluster {c} Core</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>"
            ))
            
        if overlap_sample.sum() > 0:
            fig.add_trace(go.Scatter(
                x=X_pca[overlap_sample, 0],
                y=X_pca[overlap_sample, 1],
                mode="markers",
                marker=dict(size=9, color="#dc2626", symbol="diamond-open", line=dict(width=2)),
                name=f"⚠️ Overlapping Boundary Zone ({overlap_sample.sum()} points)",
                hovertemplate="<b>⚠️ Overlapping Zone Student</b><br>Membership Entropy: %{customdata[0]:.4f}<br>Prob Margin: %{customdata[1]:.4f}<extra></extra>",
                customdata=np.column_stack((entropy_sample[overlap_sample], margin_sample[overlap_sample]))
            ))
            
        var_exp = np.sum(pca.explained_variance_ratio_) * 100
        fig.update_layout(
            title=dict(text=f"🌀 <b>2D Soft Overlapping Cluster Scatter Plot (Explained Var: {var_exp:.1f}%)</b>", font=dict(size=16)),
            xaxis=dict(title="PC1", gridcolor=GRID_COLOR),
            yaxis=dict(title="PC2", gridcolor=GRID_COLOR),
            paper_bgcolor=PLOT_BG,
            plot_bgcolor=PLOT_BG,
            margin=dict(l=40, r=40, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
    return fig


def plot_clustering_comparison_chart(benchmark_records):
    """
    Renders grouped bar comparison chart of Silhouette Score & Davies-Bouldin Index across clustering algorithms.
    """
    df_bm = pd.DataFrame(benchmark_records)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_bm["Algorithm"],
        y=df_bm["Silhouette Score"],
        name="Silhouette Score (Higher Better)",
        marker=dict(color="#059669"),
        text=[f"{v:.4f}" for v in df_bm["Silhouette Score"]],
        textposition="auto",
        hovertemplate="<b>%{x}</b><br>Silhouette Score: %{y:.4f}<extra></extra>"
    ))
    
    fig.add_trace(go.Bar(
        x=df_bm["Algorithm"],
        y=df_bm["Davies-Bouldin Index"],
        name="Davies-Bouldin Index (Lower Better)",
        marker=dict(color="#2563eb"),
        text=[f"{v:.4f}" for v in df_bm["Davies-Bouldin Index"]],
        textposition="auto",
        hovertemplate="<b>%{x}</b><br>Davies-Bouldin: %{y:.4f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text="🌳 <b>Multi-Clustering Model Metric Benchmark (KMeans vs GMM vs Hierarchical vs DBSCAN)</b>", font=dict(size=16)),
        barmode="group",
        xaxis=dict(title="Clustering Algorithm Paradigm", gridcolor=GRID_COLOR),
        yaxis=dict(title="Metric Score Value", gridcolor=GRID_COLOR),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_cluster_pca_scatter(df_processed, labels, is_3d=False, sample_size=3000, random_state=42):
    """
    Renders 2D or 3D PCA scatter plot of student clusters with centroids on processed feature matrix.
    """
    labels = np.asarray(labels)
    
    # Align lengths if feature matrix and labels differ in size (e.g. sub-sampled clustering)
    if len(df_processed) != len(labels):
        min_len = min(len(df_processed), len(labels))
        if isinstance(df_processed, np.ndarray):
            df_processed = df_processed[:min_len]
        elif hasattr(df_processed, "iloc"):
            df_processed = df_processed.iloc[:min_len]
        labels = labels[:min_len]

    if len(df_processed) > sample_size:
        np.random.seed(random_state)
        indices = np.random.choice(len(df_processed), size=sample_size, replace=False)
        X_sample = df_processed[indices] if isinstance(df_processed, np.ndarray) else df_processed.iloc[indices].values
        labels_sample = labels[indices]
    else:
        X_sample = df_processed if isinstance(df_processed, np.ndarray) else (df_processed.values if hasattr(df_processed, "values") else df_processed)
        labels_sample = labels

    n_comp = 3 if is_3d else 2
    pca = PCA(n_components=n_comp, random_state=random_state)
    X_pca = pca.fit_transform(X_sample)
    
    unique_clusters = np.unique(labels_sample)
    centroids = np.array([X_pca[labels_sample == c].mean(axis=0) for c in unique_clusters])
    
    colors = px.colors.qualitative.Bold
    cluster_str = [f"Cluster {c}" for c in labels_sample]
    
    if is_3d:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter3d(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            z=X_pca[:, 2],
            mode="markers",
            marker=dict(size=4, color=labels_sample, colorscale="Jet", opacity=0.7),
            text=cluster_str,
            name="Students",
            hovertemplate="<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter3d(
            x=centroids[:, 0],
            y=centroids[:, 1],
            z=centroids[:, 2],
            mode="markers+text",
            marker=dict(size=10, color="black", symbol="diamond"),
            text=[f"Centroid {c}" for c in unique_clusters],
            textposition="top center",
            name="Centroids"
        ))
        
        var_exp = np.sum(pca.explained_variance_ratio_) * 100
        fig.update_layout(
            title=dict(text=f"🌐 <b>3D PCA Cluster Scatter Plot (Explained Var: {var_exp:.1f}%)</b>", font=dict(size=16)),
            scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
            paper_bgcolor=PLOT_BG,
            margin=dict(l=20, r=20, t=50, b=20)
        )
    else:
        fig = go.Figure()
        
        for idx, c in enumerate(unique_clusters):
            mask = labels_sample == c
            color = colors[idx % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=X_pca[mask, 0],
                y=X_pca[mask, 1],
                mode="markers",
                marker=dict(size=6, color=color, opacity=0.6),
                name=f"Cluster {c}",
                hovertemplate=f"<b>Cluster {c}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>"
            ))
            
        fig.add_trace(go.Scatter(
            x=centroids[:, 0],
            y=centroids[:, 1],
            mode="markers+text",
            marker=dict(size=14, color="black", symbol="x"),
            text=[f"C{c}" for c in unique_clusters],
            textposition="top center",
            name="Centroids",
            hovertemplate="<b>Centroid %{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>"
        ))
        
        var_exp = np.sum(pca.explained_variance_ratio_) * 100
        fig.update_layout(
            title=dict(text=f"🗺️ <b>2D PCA Cluster Scatter Plot (Explained Var: {var_exp:.1f}%)</b>", font=dict(size=16)),
            xaxis=dict(title="Principal Component 1 (PC1)", gridcolor=GRID_COLOR),
            yaxis=dict(title="Principal Component 2 (PC2)", gridcolor=GRID_COLOR),
            paper_bgcolor=PLOT_BG,
            plot_bgcolor=PLOT_BG,
            margin=dict(l=40, r=40, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
    return fig


def plot_cluster_distribution(labels):
    """
    Renders interactive Plotly Bar & Donut chart showing Student Count & Percentage per Cluster.
    """
    unique_clusters, counts = np.unique(labels, return_counts=True)
    total_students = len(labels)
    
    cluster_names = [f"Cluster {c}" for c in sorted(unique_clusters)]
    colors = px.colors.qualitative.Bold
    
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "xy"}, {"type": "domain"}]],
        subplot_titles=("📊 Student Count per Cluster", "🍩 Cluster Percentage Distribution Share (%)")
    )
    
    fig.add_trace(
        go.Bar(
            x=cluster_names,
            y=counts,
            text=[f"{cnt:,} ({cnt/total_students*100:.1f}%)" for cnt in counts],
            textposition="auto",
            marker=dict(color=colors[:len(unique_clusters)]),
            name="Student Count",
            hovertemplate="<b>%{x}</b><br>Count: %{y:,} students<extra></extra>"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Pie(
            labels=cluster_names,
            values=counts,
            hole=0.45,
            marker=dict(colors=colors[:len(unique_clusters)]),
            name="Share %",
            hovertemplate="<b>%{label}</b><br>Share: %{percent}<br>Count: %{value:,}<extra></extra>"
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=dict(text="📊 <b>Cluster Size & Student Distribution Breakdown</b>", font=dict(size=16)),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False
    )
    return fig


def plot_cluster_centroids_heatmap(df_num, labels):
    """
    Renders an interactive Plotly Heatmap of Cluster Centroids (Mean Feature Values) and returns Centroid DF.
    """
    if len(df_num) != len(labels):
        df_temp = df_num.iloc[:len(labels)].copy()
    else:
        df_temp = df_num.copy()
    df_temp["Cluster"] = [f"Cluster {c}" for c in labels]
    
    centroids_df = df_temp.groupby("Cluster").mean()
    
    min_vals = centroids_df.min()
    max_vals = centroids_df.max()
    denom = max_vals - min_vals
    denom[denom == 0] = 1.0
    centroids_norm = (centroids_df - min_vals) / denom
    
    fig = go.Figure(data=go.Heatmap(
        z=centroids_norm.values,
        x=centroids_norm.columns,
        y=centroids_norm.index,
        colorscale="Blues",
        colorbar=dict(title="Relative Scale"),
        hovertemplate="<b>%{y}</b><br>Feature: %{x}<br>Relative Scale: %{z:.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text="🌡️ <b>Cluster Centroids Feature Heatmap (Relative Strength)</b>", font=dict(size=16)),
        xaxis=dict(title="Features", tickangle=-45, gridcolor=GRID_COLOR),
        yaxis=dict(title="Cluster Centroid", gridcolor=GRID_COLOR),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=40, r=40, t=50, b=120)
    )
    return fig, centroids_df
