import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as px_go

EXCEL_PATH = "data/placement_predict_50k_Dataset.xlsx"
PARQUET_PATH = "data/placement_dataset_cached.parquet"

# High-Contrast Professional Palette
COLOR_PRIMARY = "#1d4ed8"      # Royal Blue
COLOR_PLACED = "#059669"       # Deep Emerald Green
COLOR_NOT_PLACED = "#dc2626"   # Deep Crimson Red
COLOR_NEUTRAL = "#475569"       # Slate 600
PLOT_BG = "#ffffff"            # Pure White
GRID_COLOR = "#e2e8f0"         # Light Gray Grid

def load_data(uploaded_file=None):
    """
    Loads dataset from uploaded file (CSV or Excel) or default cached parquet/excel dataset.
    """
    if uploaded_file is not None:
        try:
            filename = uploaded_file.name.lower()
            if filename.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            print(f"Error loading uploaded file: {e}")

    if os.path.exists(PARQUET_PATH):
        df = pd.read_parquet(PARQUET_PATH)
    elif os.path.exists(EXCEL_PATH):
        df = pd.read_excel(EXCEL_PATH)
        os.makedirs("data", exist_ok=True)
        df.to_parquet(PARQUET_PATH, index=False)
    else:
        raise FileNotFoundError(f"Dataset file not found at {EXCEL_PATH}")
    return df

def get_dataset_metadata(df):
    """
    Returns high-level statistics and metadata about the dataset.
    """
    num_rows, num_cols = df.shape
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    has_placement = "PlacementStatus" in df.columns
    if has_placement:
        placed_count = (df["PlacementStatus"] == 1).sum()
        placement_rate = (placed_count / num_rows) * 100
        placed_df = df[df["PlacementStatus"] == 1]
    else:
        placement_rate = 0.0
        placed_df = df

    has_salary = "Salary Package" in df.columns
    if has_salary:
        avg_salary = placed_df["Salary Package"].mean() if len(placed_df) > 0 else df["Salary Package"].mean()
        max_salary = df["Salary Package"].max()
    else:
        avg_salary = 0.0
        max_salary = 0.0
    
    col_summary = []
    for col in df.columns:
        null_cnt = df[col].isnull().sum()
        null_pct = (null_cnt / num_rows) * 100
        col_summary.append({
            "Column Name": col,
            "Data Type": str(df[col].dtype),
            "Non-Null Count": num_rows - null_cnt,
            "Null Count": null_cnt,
            "Null Percentage (%)": round(null_pct, 2),
            "Unique Values": df[col].nunique(),
            "Sample Value": str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "N/A"
        })
    
    col_summary_df = pd.DataFrame(col_summary)
    
    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "memory_mb": round(memory_mb, 2),
        "placement_rate": round(placement_rate, 2),
        "avg_salary": round(avg_salary, 2),
        "max_salary": round(max_salary, 2),
        "columns_summary": col_summary_df
    }

def get_missing_values_summary(df):
    """
    Calculates null values and missing percentages per column.
    """
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    
    if len(null_cols) == 0:
        return pd.DataFrame(columns=["Column", "Missing Count", "Missing Percentage (%)"])
    
    missing_df = pd.DataFrame({
        "Column": null_cols.index,
        "Missing Count": null_cols.values,
        "Missing Percentage (%)": np.round((null_cols.values / len(df)) * 100, 2)
    }).sort_values(by="Missing Count", ascending=False)
    
    return missing_df

def apply_professional_layout(fig, title=None, height=450):
    """
    Applies clean, high-contrast, professional light styling to any Plotly figure.
    """
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#0f172a", family="sans-serif")) if title else None,
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color="#0f172a", size=12),
        height=height,
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            linecolor="#cbd5e1",
            tickfont=dict(color="#1e293b", size=11),
            title=dict(font=dict(color="#0f172a", size=13))
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            linecolor="#cbd5e1",
            tickfont=dict(color="#1e293b", size=11),
            title=dict(font=dict(color="#0f172a", size=13))
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#cbd5e1",
            borderwidth=1,
            font=dict(color="#0f172a", size=11)
        )
    )
    return fig

def plot_missing_values_chart(missing_df):
    """
    Plotly bar chart for missing values with high contrast.
    """
    if missing_df.empty:
        fig = px_go.Figure()
        fig.add_annotation(text="No missing values in dataset!", showarrow=False, font=dict(size=18, color="#059669"))
        return apply_professional_layout(fig, height=300)
    
    fig = px.bar(
        missing_df,
        x="Column",
        y="Missing Count",
        text="Missing Percentage (%)",
        title="Missing Values Profile per Feature",
        labels={"Missing Count": "Number of Null Records", "Column": "Feature"},
        color="Missing Percentage (%)",
        color_continuous_scale=["#fca5a5", "#b91c1c"]
    )
    fig.update_traces(texttemplate='<b>%{text}%</b>', textposition='outside', textfont=dict(color="#0f172a", size=12))
    return apply_professional_layout(fig, "Missing Values Profile per Feature", height=450)

def plot_univariate_distribution(df, column):
    """
    Plots interactive distribution for a given feature.
    """
    if np.issubdtype(df[column].dtype, np.number):
        fig = px.histogram(
            df,
            x=column,
            marginal="box",
            nbins=30,
            title=f"Distribution of {column}",
            color_discrete_sequence=[COLOR_PRIMARY],
            opacity=0.85
        )
    else:
        value_counts = df[column].value_counts().reset_index()
        value_counts.columns = [column, "Count"]
        fig = px.bar(
            value_counts,
            x=column,
            y="Count",
            title=f"Category Counts of {column}",
            color=column,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
    return apply_professional_layout(fig, f"Distribution of {column}", height=450)

def plot_placement_by_category(df, cat_col):
    """
    Bar chart showing Placement status breakdown across categorical feature.
    """
    if "PlacementStatus" not in df.columns:
        fig = px_go.Figure()
        fig.add_annotation(text="PlacementStatus column not in dataset", showarrow=False, font=dict(size=16))
        return apply_professional_layout(fig, height=300)
        
    grouped = df.groupby([cat_col, "PlacementStatus"]).size().reset_index(name="Count")
    grouped["Placement Status"] = grouped["PlacementStatus"].map({0: "Not Placed", 1: "Placed"})
    
    fig = px.bar(
        grouped,
        x=cat_col,
        y="Count",
        color="Placement Status",
        barmode="group",
        title=f"Placement Outcomes by {cat_col}",
        color_discrete_map={"Not Placed": COLOR_NOT_PLACED, "Placed": COLOR_PLACED}
    )
    return apply_professional_layout(fig, f"Placement Outcomes by {cat_col}", height=450)

def plot_feature_vs_placement_box(df, num_col):
    """
    Box plot comparing numerical feature against Placement Status.
    """
    if "PlacementStatus" not in df.columns:
        fig = px_go.Figure()
        fig.add_annotation(text="PlacementStatus column not in dataset", showarrow=False, font=dict(size=16))
        return apply_professional_layout(fig, height=300)

    df_copy = df.copy()
    df_copy["Placement Status"] = df_copy["PlacementStatus"].map({0: "Not Placed", 1: "Placed"})
    
    fig = px.box(
        df_copy,
        x="Placement Status",
        y=num_col,
        color="Placement Status",
        title=f"{num_col} vs Placement Status",
        color_discrete_map={"Not Placed": COLOR_NOT_PLACED, "Placed": COLOR_PLACED},
        points="outliers"
    )
    return apply_professional_layout(fig, f"{num_col} vs Placement Status", height=450)

def plot_correlation_heatmap(df):
    """
    Plotly correlation heatmap for numerical columns with bold text contrast.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    num_cols = [c for c in num_cols if c != "StudentID"]
    
    if len(num_cols) == 0:
        fig = px_go.Figure()
        fig.add_annotation(text="No numerical features found for correlation", showarrow=False)
        return apply_professional_layout(fig, height=300)

    corr = df[num_cols].corr().round(2)
    
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title="Numerical Features Correlation Matrix"
    )
    fig.update_traces(textfont=dict(color="#0f172a", size=10))
    return apply_professional_layout(fig, "Numerical Features Correlation Matrix", height=650)

def plot_cgpa_vs_salary_regression(df):
    """
    Plots CGPA vs Salary Package scatter plot with a fitted OLS Linear Regression line.
    """
    if "CGPA" not in df.columns or "Salary Package" not in df.columns:
        fig = px_go.Figure()
        fig.add_annotation(text="Required columns CGPA or Salary Package missing", showarrow=False)
        return apply_professional_layout(fig, height=300)

    valid_df = df.dropna(subset=["CGPA", "Salary Package"])
    
    # Calculate OLS fit
    x = valid_df["CGPA"].values
    y = valid_df["Salary Package"].values
    if len(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        r_corr = np.corrcoef(x, y)[0, 1]
    else:
        slope, intercept, r_corr = 0.0, 0.0, 0.0
        
    x_line = np.linspace(x.min() if len(x)>0 else 0, x.max() if len(x)>0 else 10, 100)
    y_line = slope * x_line + intercept
    
    color_kwargs = {}
    if "PlacementStatus" in valid_df.columns:
        color_kwargs["color"] = valid_df["PlacementStatus"].map({0: "Not Placed", 1: "Placed"})
        color_kwargs["color_discrete_map"] = {"Not Placed": COLOR_NOT_PLACED, "Placed": COLOR_PLACED}

    fig = px.scatter(
        valid_df,
        x="CGPA",
        y="Salary Package",
        opacity=0.6,
        title=f"CGPA vs Salary Package — Linear Regression Line (r = {r_corr:.3f})",
        **color_kwargs
    )
    
    # Add OLS line
    fig.add_trace(px_go.Scatter(
        x=x_line,
        y=y_line,
        mode="lines",
        name=f"Linear Fit: Salary = {slope:.2f} × CGPA + ({intercept:.2f})",
        line=dict(color="#1d4ed8", width=3.5, dash="dash")
    ))
    
    return apply_professional_layout(fig, f"Linear Regression: CGPA vs Salary Package (Correlation r = {r_corr:.3f})", height=520)

def plot_residuals(y_true, y_pred):
    """
    Plots residual diagnostic scatter plot (Actual vs Predicted & Residuals Distribution).
    """
    residuals = y_true - y_pred
    res_df = pd.DataFrame({
        "Actual Salary": y_true,
        "Predicted Salary": y_pred,
        "Residual": residuals
    })
    
    fig = px.scatter(
        res_df,
        x="Predicted Salary",
        y="Residual",
        opacity=0.6,
        color_discrete_sequence=[COLOR_PRIMARY],
        title="Linear Regression Residual Analysis (Residuals vs Predicted Values)"
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#dc2626", line_width=2)
    return apply_professional_layout(fig, "Linear Regression Residual Analysis", height=420)

def plot_confusion_matrix(cm):
    """
    Plots interactive confusion matrix heatmap for Logistic Regression classification.
    """
    cm_arr = np.array(cm)
    labels = ["Not Placed (0)", "Placed (1)"]
    
    # Text matrix formatted with count and label
    text_matrix = [
        [f"True Neg (TN)<br><b>{cm_arr[0][0]:,}</b>", f"False Pos (FP)<br><b>{cm_arr[0][1]:,}</b>"],
        [f"False Neg (FN)<br><b>{cm_arr[1][0]:,}</b>", f"True Pos (TP)<br><b>{cm_arr[1][1]:,}</b>"]
    ]
    
    fig = px_go.Figure(data=px_go.Heatmap(
        z=cm_arr,
        x=labels,
        y=labels,
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=14, color="#0f172a"),
        colorscale="Blues",
        showscale=True
    ))
    fig.update_layout(
        title="Logistic Regression — Confusion Matrix",
        xaxis_title="Predicted Placement Status",
        yaxis_title="Actual Placement Status",
        yaxis=dict(autorange="reversed")
    )
    return apply_professional_layout(fig, "Logistic Regression — Confusion Matrix", height=420)

def plot_roc_curve(fpr, tpr, roc_auc):
    """
    Plots interactive ROC Curve for Logistic Regression model.
    """
    fig = px_go.Figure()
    
    # ROC Curve trace
    fig.add_trace(px_go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name=f"Logistic Regression (AUC = {roc_auc:.4f})",
        line=dict(color="#1d4ed8", width=3)
    ))
    
    # Baseline random guess trace
    fig.add_trace(px_go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Classifier (AUC = 0.50)",
        line=dict(color="#dc2626", width=2, dash="dash")
    ))
    
    fig.update_layout(
        title=f"Receiver Operating Characteristic (ROC) Curve — AUC = {roc_auc:.4f}",
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Sensitivity / Recall)",
        legend=dict(x=0.55, y=0.1)
    )
    return apply_professional_layout(fig, f"ROC Curve (AUC = {roc_auc:.4f})", height=420)

def plot_odds_ratios(odds_records):
    """
    Plots Odds Ratios / Feature Impact bar chart for Logistic Regression.
    """
    if not odds_records:
        fig = px_go.Figure()
        fig.add_annotation(text="Odds Ratio data unavailable", showarrow=False)
        return apply_professional_layout(fig, height=300)

    odds_df = pd.DataFrame(odds_records)
    
    fig = px.bar(
        odds_df,
        x="Log_Odds",
        y="Feature",
        orientation="h",
        text="Odds_Ratio",
        title="Logistic Regression — Log Odds / Feature Influence on Placement",
        color="Log_Odds",
        color_continuous_scale="RdYlGn"
    )
    fig.update_traces(texttemplate='<b>OR: %{text}</b>', textposition='outside')
    fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Log-Odds Coefficient (Positive = Boosts Placement)")
    return apply_professional_layout(fig, "Logistic Regression — Feature Impact on Placement Odds", height=450)

def plot_feature_importances(importance_records, title="Decision Tree Feature Importances"):
    """
    Plots Gini/MSE Impurity Feature Importance horizontal bar chart for Decision Trees.
    """
    if not importance_records:
        fig = px_go.Figure()
        fig.add_annotation(text="Feature Importance data unavailable", showarrow=False)
        return apply_professional_layout(fig, height=300)

    imp_df = pd.DataFrame(importance_records)
    
    fig = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
        title=title,
        color="Importance",
        color_continuous_scale="Viridis"
    )
    fig.update_traces(texttemplate='<b>%{text:.4f}</b>', textposition='outside')
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis_title="Gini / Impurity Reduction Importance"
    )
    return apply_professional_layout(fig, title, height=450)

def plot_impurity_comparison_curves():
    """
    Plots interactive comparison curves for Decision Tree Impurity Measures:
    Gini Impurity, Entropy (Information Gain), and Misclassification Error
    across binary class probability range p in [0, 1].
    """
    p = np.linspace(0.0001, 0.9999, 200)
    
    # 1. Gini Impurity: G(p) = 2p(1-p) for binary classification
    gini = 2 * p * (1 - p)
    
    # 2. Entropy: H(p) = -p log2(p) - (1-p) log2(1-p), scaled by 0.5 for direct comparison
    entropy = - (p * np.log2(p) + (1 - p) * np.log2(1 - p))
    entropy_scaled = entropy / 2.0
    
    # 3. Misclassification Error: E(p) = 1 - max(p, 1-p)
    misclass = 1 - np.maximum(p, 1 - p)
    
    fig = px_go.Figure()
    
    fig.add_trace(px_go.Scatter(
        x=p, y=entropy_scaled,
        mode="lines",
        name="Entropy / 2 (ID3 & C4.5)",
        line=dict(color="#8b5cf6", width=3)
    ))
    
    fig.add_trace(px_go.Scatter(
        x=p, y=gini,
        mode="lines",
        name="Gini Impurity (CART / Karth)",
        line=dict(color="#059669", width=3)
    ))
    
    fig.add_trace(px_go.Scatter(
        x=p, y=misclass,
        mode="lines",
        name="Misclassification Error",
        line=dict(color="#dc2626", width=2, dash="dash")
    ))
    
    fig.update_layout(
        title="Decision Tree Impurity Measures Comparison (Gini vs Entropy vs Misclassification Error)",
        xaxis_title="Class 1 Probability p",
        yaxis_title="Impurity Score",
        legend=dict(x=0.65, y=0.95)
    )
    return apply_professional_layout(fig, "Decision Tree Impurity Measures Comparison", height=420)





