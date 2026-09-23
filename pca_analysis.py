"""
Principal Component Analysis (PCA) & Dimensionality Reduction Analytics Module
Placement & Salary Analytics System
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    r2_score, mean_absolute_error, mean_squared_error
)
from utils import (
    load_data, apply_professional_layout,
    COLOR_PRIMARY, COLOR_PLACED, COLOR_NOT_PLACED, COLOR_NEUTRAL, PLOT_BG, GRID_COLOR
)

MODEL_DIR = "saved_models"


def train_and_evaluate_pca(custom_df=None, random_state=42):
    """
    Fits PCA on dataset features, performs feature loadings analysis, scree variance profiling,
    and builds downstream PCA + Logistic Regression (Classification) and PCA + Linear Regression (Regression) pipelines.

    Saves pipeline artifacts and returns PCA metrics dictionary.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = custom_df if custom_df is not None else load_data()

    # Target columns
    has_placement = "PlacementStatus" in df.columns
    has_salary = "Salary Package" in df.columns

    # Feature columns
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

    # Preprocess entire dataset for PCA extraction
    X_processed = preprocessor.fit_transform(X)

    # Feature names after OneHotEncoding
    ohe_cat_features = []
    if cat_cols:
        try:
            ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            ohe_cat_features = ohe.get_feature_names_out(cat_cols).tolist()
        except Exception:
            ohe_cat_features = [f"cat_{i}" for i in range(X_processed.shape[1] - len(num_cols))]

    all_feature_names = num_cols + ohe_cat_features

    # Fit Full PCA
    max_components = min(X_processed.shape[0], X_processed.shape[1])
    pca = PCA(n_components=max_components, random_state=random_state)
    pca.fit(X_processed)

    explained_variance_ratio = pca.explained_variance_ratio_.tolist()
    cumulative_variance = np.cumsum(explained_variance_ratio).tolist()
    eigenvalues = pca.explained_variance_.tolist()

    # Find component counts needed for target variance thresholds
    thresholds = [0.80, 0.90, 0.95, 0.99]
    components_needed = {}
    for thresh in thresholds:
        cnt = int(np.argmax(np.array(cumulative_variance) >= thresh) + 1)
        components_needed[f"{int(thresh * 100)}%"] = cnt

    n_comp_95 = components_needed.get("95%", min(10, max_components))

    # Feature Loadings calculation for top 10 PCs
    n_loadings_pcs = min(10, max_components)
    loadings_matrix = pca.components_[:n_loadings_pcs, :]  # Shape: (n_pcs, n_features)

    loadings_data = []
    for pc_idx in range(n_loadings_pcs):
        pc_name = f"PC{pc_idx + 1}"
        for feat_idx, feat_name in enumerate(all_feature_names):
            val = float(loadings_matrix[pc_idx, feat_idx])
            loadings_data.append({
                "Component": pc_name,
                "Feature": feat_name,
                "Loading": round(val, 4),
                "AbsLoading": round(abs(val), 4)
            })

    loadings_df = pd.DataFrame(loadings_data)

    # Top features per PC summary
    top_features_per_pc = {}
    for pc_idx in range(n_loadings_pcs):
        pc_name = f"PC{pc_idx + 1}"
        pc_df = loadings_df[loadings_df["Component"] == pc_name].sort_values(by="AbsLoading", ascending=False).head(5)
        top_features_per_pc[pc_name] = pc_df[["Feature", "Loading"]].to_dict(orient="records")

    # PCA 2D/3D Transformed sample data for scatter plot visualization
    X_pca_full = pca.transform(X_processed)
    pca_df = pd.DataFrame({
        "PC1": X_pca_full[:, 0],
        "PC2": X_pca_full[:, 1],
        "PC3": X_pca_full[:, 2] if X_pca_full.shape[1] >= 3 else X_pca_full[:, 1]
    })

    if has_placement:
        pca_df["PlacementStatus"] = df["PlacementStatus"].values
    if has_salary:
        pca_df["Salary Package"] = df["Salary Package"].values

    # -------------------------------------------------------------
    # Downstream Model 1: PCA + Logistic Regression (Classification)
    # -------------------------------------------------------------
    pca_logistic_metrics = {}
    pca_logistic_pipeline = None

    if has_placement:
        y_class = df["PlacementStatus"]
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X, y_class, test_size=0.2, random_state=random_state, stratify=y_class
        )

        pca_logistic_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("pca", PCA(n_components=n_comp_95, random_state=random_state)),
            ("classifier", LogisticRegression(random_state=random_state, max_iter=1000))
        ])

        pca_logistic_pipeline.fit(X_train_c, y_train_c)
        y_pred_c = pca_logistic_pipeline.predict(X_test_c)
        y_prob_c = pca_logistic_pipeline.predict_proba(X_test_c)[:, 1]

        acc = float(accuracy_score(y_test_c, y_pred_c))
        prec = float(precision_score(y_test_c, y_pred_c, zero_division=0))
        rec = float(recall_score(y_test_c, y_pred_c, zero_division=0))
        f1 = float(f1_score(y_test_c, y_pred_c, zero_division=0))
        roc_auc = float(roc_auc_score(y_test_c, y_prob_c))

        cm = confusion_matrix(y_test_c, y_pred_c).tolist()
        fpr, tpr, _ = roc_curve(y_test_c, y_prob_c)

        pca_logistic_metrics = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "confusion_matrix": cm,
            "roc_curve": {"fpr": [round(float(x), 4) for x in fpr], "tpr": [round(float(x), 4) for x in tpr]},
            "n_components_used": n_comp_95
        }

        joblib.dump(pca_logistic_pipeline, os.path.join(MODEL_DIR, "placement_pca_logistic_pipeline.joblib"))

    # -------------------------------------------------------------
    # Downstream Model 2: PCA + Linear Regression (Regression)
    # -------------------------------------------------------------
    pca_lr_metrics = {}
    pca_lr_pipeline = None

    if has_salary:
        y_sal = df["Salary Package"]
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X, y_sal, test_size=0.2, random_state=random_state
        )

        pca_lr_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("pca", PCA(n_components=n_comp_95, random_state=random_state)),
            ("regressor", LinearRegression())
        ])

        pca_lr_pipeline.fit(X_train_r, y_train_r)
        y_pred_r = pca_lr_pipeline.predict(X_test_r)

        r2 = float(r2_score(y_test_r, y_pred_r))
        mae = float(mean_absolute_error(y_test_r, y_pred_r))
        mse = float(mean_squared_error(y_test_r, y_pred_r))
        rmse = float(np.sqrt(mse))

        pca_lr_metrics = {
            "r2_score": round(r2, 4),
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "n_components_used": n_comp_95
        }

        joblib.dump(pca_lr_pipeline, os.path.join(MODEL_DIR, "salary_pca_lr_pipeline.joblib"))

    # Save standalone PCA transformer pipeline
    standalone_pca_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("pca", PCA(n_components=n_comp_95, random_state=random_state))
    ])
    standalone_pca_pipe.fit(X)
    joblib.dump(standalone_pca_pipe, os.path.join(MODEL_DIR, "pca_transformer.joblib"))

    pca_metrics = {
        "n_total_features": X_processed.shape[1],
        "n_samples": X_processed.shape[0],
        "explained_variance_ratio": [round(float(v), 4) for v in explained_variance_ratio],
        "cumulative_explained_variance": [round(float(v), 4) for v in cumulative_variance],
        "eigenvalues": [round(float(v), 4) for v in eigenvalues],
        "components_needed": components_needed,
        "n_comp_95": n_comp_95,
        "pc1_var_pct": round(float(explained_variance_ratio[0]) * 100, 2),
        "pc2_var_pct": round(float(explained_variance_ratio[1]) * 100, 2) if len(explained_variance_ratio) > 1 else 0.0,
        "pc3_var_pct": round(float(explained_variance_ratio[2]) * 100, 2) if len(explained_variance_ratio) > 2 else 0.0,
        "top_features_per_pc": top_features_per_pc,
        "loadings_summary": loadings_df.head(100).to_dict(orient="records"),
        "pca_logistic_metrics": pca_logistic_metrics,
        "pca_lr_metrics": pca_lr_metrics
    }

    pca_pipelines = {
        "pca_logistic": pca_logistic_pipeline,
        "pca_lr": pca_lr_pipeline,
        "pca_transformer": standalone_pca_pipe
    }

    return pca_metrics, pca_pipelines, pca_df


def plot_pca_scree_plot(pca_metrics):
    """
    Renders Scree Plot showing individual explained variance bars and cumulative variance line.
    """
    exp_var = [v * 100 for v in pca_metrics.get("explained_variance_ratio", [])[:15]]
    cum_var = [v * 100 for v in pca_metrics.get("cumulative_explained_variance", [])[:15]]
    pcs = [f"PC{i+1}" for i in range(len(exp_var))]

    fig = go.Figure()

    # Individual Variance Bar Chart
    fig.add_trace(go.Bar(
        x=pcs,
        y=exp_var,
        name="Individual Variance Explained (%)",
        marker_color=COLOR_PRIMARY,
        text=[f"{v:.1f}%" for v in exp_var],
        textposition="outside",
        opacity=0.85
    ))

    # Cumulative Variance Line Chart
    fig.add_trace(go.Scatter(
        x=pcs,
        y=cum_var,
        name="Cumulative Variance (%)",
        mode="lines+markers",
        line=dict(color=COLOR_PLACED, width=3),
        marker=dict(size=8, symbol="diamond")
    ))

    # 95% Variance Threshold Reference Line
    fig.add_hline(
        y=95, line_dash="dash", line_color=COLOR_NOT_PLACED,
        annotation_text="95% Variance Threshold", annotation_position="bottom right"
    )

    fig = apply_professional_layout(
        fig,
        title="📊 PCA Scree Plot — Individual & Cumulative Explained Variance",
        x_title="Principal Components",
        y_title="Variance Explained (%)",
        height=450
    )
    fig.update_layout(yaxis=dict(range=[0, 105]))
    return fig


def plot_pca_loadings_heatmap(pca_metrics, top_n_features=15, n_components=5):
    """
    Renders feature loadings heatmap across top Principal Components.
    """
    loadings = pca_metrics.get("loadings_summary", [])
    if not loadings:
        fig = go.Figure()
        fig.add_annotation(text="No loading data available", showarrow=False)
        return fig

    df_loadings = pd.DataFrame(loadings)

    # Filter top components
    pcs = [f"PC{i+1}" for i in range(n_components)]
    df_filtered = df_loadings[df_loadings["Component"].isin(pcs)]

    # Aggregate max absolute loading to pick top N features
    top_feats = df_filtered.groupby("Feature")["AbsLoading"].max().sort_values(ascending=False).head(top_n_features).index.tolist()
    pivot_df = df_filtered[df_filtered["Feature"].isin(top_feats)].pivot(index="Feature", columns="Component", values="Loading")

    # Reorder columns
    valid_pcs = [p for p in pcs if p in pivot_df.columns]
    pivot_df = pivot_df[valid_pcs]

    fig = px.imshow(
        pivot_df,
        labels=dict(x="Principal Components", y="Original Features", color="Loading Weight"),
        x=valid_pcs,
        y=pivot_df.index,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        text_auto=".2f"
    )

    fig = apply_professional_layout(
        fig,
        title=f"🔥 Top {top_n_features} Feature Loadings Matrix (PC Components Breakdown)",
        height=500
    )
    return fig


def plot_pca_2d_scatter(pca_df, color_col="PlacementStatus"):
    """
    Renders 2D PCA Scatter plot of PC1 vs PC2.
    """
    if "PC1" not in pca_df.columns or "PC2" not in pca_df.columns:
        fig = go.Figure()
        return fig

    df_sample = pca_df.sample(n=min(3000, len(pca_df)), random_state=42)

    color_kwargs = {}
    if color_col in df_sample.columns:
        if color_col == "PlacementStatus":
            df_sample["Status_Label"] = df_sample["PlacementStatus"].map({1: "Placed", 0: "Not Placed"})
            color_kwargs["color"] = "Status_Label"
            color_kwargs["color_discrete_map"] = {"Placed": COLOR_PLACED, "Not Placed": COLOR_NOT_PLACED}
        else:
            color_kwargs["color"] = color_col
            color_kwargs["color_continuous_scale"] = "Viridis"

    fig = px.scatter(
        df_sample,
        x="PC1",
        y="PC2",
        opacity=0.75,
        title="🗺️ 2D PCA Projection (PC1 vs PC2)",
        **color_kwargs
    )

    fig = apply_professional_layout(
        fig,
        title="🗺️ 2D PCA Dimensionality Reduction Scatter Plot (PC1 vs PC2)",
        x_title="Principal Component 1 (PC1)",
        y_title="Principal Component 2 (PC2)",
        height=500
    )
    return fig


def plot_pca_3d_scatter(pca_df, color_col="PlacementStatus"):
    """
    Renders 3D PCA Scatter plot of PC1 vs PC2 vs PC3.
    """
    if "PC1" not in pca_df.columns or "PC2" not in pca_df.columns or "PC3" not in pca_df.columns:
        fig = go.Figure()
        return fig

    df_sample = pca_df.sample(n=min(3000, len(pca_df)), random_state=42)

    color_kwargs = {}
    if color_col in df_sample.columns:
        if color_col == "PlacementStatus":
            df_sample["Status_Label"] = df_sample["PlacementStatus"].map({1: "Placed", 0: "Not Placed"})
            color_kwargs["color"] = "Status_Label"
            color_kwargs["color_discrete_map"] = {"Placed": COLOR_PLACED, "Not Placed": COLOR_NOT_PLACED}
        else:
            color_kwargs["color"] = color_col
            color_kwargs["color_continuous_scale"] = "Viridis"

    fig = px.scatter_3d(
        df_sample,
        x="PC1",
        y="PC2",
        z="PC3",
        opacity=0.7,
        title="🌐 3D PCA Interactive Projection (PC1 vs PC2 vs PC3)",
        **color_kwargs
    )

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor=PLOT_BG,
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3"
        )
    )
    return fig


def plot_pca_vs_raw_comparison(raw_log_m, pca_log_m, raw_lr_m, pca_lr_m):
    """
    Renders comparative performance chart between full feature baseline vs PCA-reduced models.
    """
    categories = ["Classification Accuracy (%)", "ROC-AUC Score", "Salary Regression R²"]

    raw_acc = raw_log_m.get("accuracy", 0) * 100
    pca_acc = pca_log_m.get("accuracy", 0) * 100

    raw_auc = raw_log_m.get("roc_auc", 0)
    pca_auc = pca_log_m.get("roc_auc", 0)

    raw_r2 = raw_lr_m.get("r2_score", 0)
    pca_r2 = pca_lr_m.get("r2_score", 0)

    raw_vals = [raw_acc, raw_auc, raw_r2]
    pca_vals = [pca_acc, pca_auc, pca_r2]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=categories,
        y=raw_vals,
        name="Full Feature Baseline Model",
        marker_color=COLOR_PRIMARY,
        text=[f"{v:.2f}" for v in raw_vals],
        textposition="auto"
    ))

    fig.add_trace(go.Bar(
        x=categories,
        y=pca_vals,
        name="PCA-Reduced Model (95% Var)",
        marker_color=COLOR_PLACED,
        text=[f"{v:.2f}" for v in pca_vals],
        textposition="auto"
    ))

    fig = apply_professional_layout(
        fig,
        title="⚖️ Performance Comparison: Full Feature Models vs PCA-Transformed Models",
        y_title="Metric Score",
        height=450
    )
    fig.update_layout(barmode="group")
    return fig
