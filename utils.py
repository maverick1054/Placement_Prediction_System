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

def load_data():
    """
    Loads dataset with parquet caching for sub-second dashboard startup.
    """
    if os.path.exists(PARQUET_PATH):
        df = pd.read_parquet(PARQUET_PATH)
    else:
        if os.path.exists(EXCEL_PATH):
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
    placed_count = (df["PlacementStatus"] == 1).sum()
    placement_rate = (placed_count / num_rows) * 100
    
    placed_df = df[df["PlacementStatus"] == 1]
    avg_salary = placed_df["Salary Package"].mean() if len(placed_df) > 0 else 0.0
    max_salary = df["Salary Package"].max()
    
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
