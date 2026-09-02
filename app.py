import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as px_go
from utils import (
    load_data, get_dataset_metadata, get_missing_values_summary,
    plot_missing_values_chart, plot_univariate_distribution,
    plot_placement_by_category, plot_feature_vs_placement_box,
    plot_correlation_heatmap, plot_cgpa_vs_salary_regression, plot_residuals,
    plot_confusion_matrix, plot_roc_curve, plot_odds_ratios, plot_feature_importances,
    plot_impurity_comparison_curves,
    apply_professional_layout, COLOR_PRIMARY, COLOR_PLACED, COLOR_NOT_PLACED
)
from model_trainer import load_trained_pipelines, train_and_evaluate_models

# Page Configuration
st.set_page_config(
    page_title="Placement Analytics & ML Predictor (ID3, C4.5, CART, Logistic & Linear Regression)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Dataset Uploader & Theme Switcher
st.sidebar.title("📁 Dataset & Theme Controls")

theme_choice = st.sidebar.selectbox(
    "🎨 UI Background & Color Theme:",
    [
        "Ultra-Clean Bright Light (Recommended)",
        "Ice Blue Mesh Gradient",
        "Emerald Mint Fresh Light",
        "Midnight Dark Glass"
    ]
)

if theme_choice.startswith("Ultra-Clean"):
    app_bg = "#f8fafc"
    card_bg = "#ffffff"
    card_border = "1px solid #cbd5e1"
    text_color = "#0f172a"
    sub_text = "#475569"
    tab_list_bg = "#e2e8f0"
    tab_unselected_bg = "#ffffff"
    tab_unselected_text = "#0f172a"
    tab_unselected_border = "#cbd5e1"
    tab_active_bg = "#2563eb"
    tab_active_text = "#ffffff"
    tab_active_border = "#1d4ed8"
elif theme_choice.startswith("Ice"):
    app_bg = "linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 50%, #e2e8f0 100%)"
    card_bg = "#ffffff"
    card_border = "1px solid #93c5fd"
    text_color = "#0f172a"
    sub_text = "#334155"
    tab_list_bg = "#dbeafe"
    tab_unselected_bg = "#f0f9ff"
    tab_unselected_text = "#1e3a8a"
    tab_unselected_border = "#bfdbfe"
    tab_active_bg = "#1d4ed8"
    tab_active_text = "#ffffff"
    tab_active_border = "#1e40af"
elif theme_choice.startswith("Emerald"):
    app_bg = "linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 50%, #f8fafc 100%)"
    card_bg = "#ffffff"
    card_border = "1px solid #a7f3d0"
    text_color = "#064e3b"
    sub_text = "#1e293b"
    tab_list_bg = "#d1fae5"
    tab_unselected_bg = "#f0fdf4"
    tab_unselected_text = "#064e3b"
    tab_unselected_border = "#6ee7b7"
    tab_active_bg = "#059669"
    tab_active_text = "#ffffff"
    tab_active_border = "#047857"
else:
    app_bg = "linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%)"
    card_bg = "#1e293b"
    card_border = "1px solid #475569"
    text_color = "#ffffff"
    sub_text = "#e2e8f0"
    tab_list_bg = "#0f172a"
    tab_unselected_bg = "#334155"
    tab_unselected_text = "#ffffff"
    tab_unselected_border = "#64748b"
    tab_active_bg = "#3b82f6"
    tab_active_text = "#ffffff"
    tab_active_border = "#60a5fa"

st.markdown(f"""
<style>
    /* Main App Background & Font */
    .stApp {{
        background: {app_bg} !important;
        color: {text_color} !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    
    /* Header Styling */
    .main-title {{
        font-size: 2.2rem;
        font-weight: 800;
        color: {text_color} !important;
        letter-spacing: -0.025em;
        margin-bottom: 0.2rem;
    }}
    
    .sub-title {{
        font-size: 1.05rem;
        color: {sub_text} !important;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }}
    
    /* Professional Glass & Card Containers */
    .metric-card {{
        background: {card_bg} !important;
        border: {card_border} !important;
        border-radius: 14px;
        padding: 18px 15px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 12px 20px rgba(0, 0, 0, 0.12);
    }}
    
    .metric-val {{
        font-size: 1.9rem;
        font-weight: 800;
        line-height: 1.2;
    }}
    
    .metric-lbl {{
        font-size: 0.85rem;
        font-weight: 700;
        color: {sub_text} !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }}
    
    /* High-Contrast Tab Bar Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        background-color: {tab_list_bg} !important;
        padding: 8px;
        border-radius: 12px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 44px;
        white-space: pre-wrap;
        border-radius: 8px;
        font-weight: 700 !important;
        font-size: 0.92rem;
        background-color: {tab_unselected_bg} !important;
        color: {tab_unselected_text} !important;
        border: 1px solid {tab_unselected_border} !important;
        padding: 0 16px !important;
    }}
    
    .stTabs [data-baseweb="tab"] p, .stTabs [data-baseweb="tab"] span, .stTabs [data-baseweb="tab"] div {{
        color: {tab_unselected_text} !important;
        font-weight: 700 !important;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {tab_active_bg} !important;
        color: {tab_active_text} !important;
        border: 2px solid {tab_active_border} !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important;
    }}

    .stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] span, .stTabs [aria-selected="true"] div {{
        color: {tab_active_text} !important;
        font-weight: 800 !important;
    }}
    
    /* Buttons */
    .stButton>button {{
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: #ffffff !important;
        font-weight: 700;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3);
    }}
    
    .stButton>button:hover {{
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        color: #ffffff !important;
    }}
</style>
""", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader(
    "Upload Custom Dataset (CSV or Excel):",
    type=["csv", "xlsx", "xls"],
    help="Upload your dataset file to perform custom EDA, preprocessing, and ML modeling."
)

if uploaded_file is not None:
    st.sidebar.success(f"✅ Loaded: `{uploaded_file.name}`")
    df = load_data(uploaded_file)
    lr_pipeline, simple_lr, logistic_pipeline, dt_c_pipeline, dt_r_pipeline, ensemble_pipelines, metadata = load_trained_pipelines()
else:
    df = load_data()
    lr_pipeline, simple_lr, logistic_pipeline, dt_c_pipeline, dt_r_pipeline, ensemble_pipelines, metadata = load_trained_pipelines()

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Dynamic Data Filters")

# Apply sidebar filters if columns exist
filtered_df = df.copy()

if "Stream" in filtered_df.columns:
    selected_stream = st.sidebar.multiselect(
        "Filter by Stream:",
        options=sorted(filtered_df["Stream"].dropna().unique().tolist()),
        default=filtered_df["Stream"].dropna().unique().tolist()
    )
    if selected_stream:
        filtered_df = filtered_df[filtered_df["Stream"].isin(selected_stream)]

if "CollegeTier" in filtered_df.columns:
    selected_tier = st.sidebar.multiselect(
        "Filter by College Tier:",
        options=sorted(filtered_df["CollegeTier"].dropna().unique().tolist()),
        default=filtered_df["CollegeTier"].dropna().unique().tolist()
    )
    if selected_tier:
        filtered_df = filtered_df[filtered_df["CollegeTier"].isin(selected_tier)]

# Dashboard Header
st.markdown('<div class="main-title">🎓 Student Placement & Salary Analytics — Machine Learning Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Classification & Regression using Base Models (Logistic, Linear, Decision Trees) & Ensemble Models (Bagging: Random Forest & Bagging; Boosting: Gradient Boosting & AdaBoost)</div>', unsafe_allow_html=True)

# Tabs Navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset Overview & Upload",
    "🧹 Data Preprocessing & Health",
    "📈 Exploratory Data Analysis (EDA)",
    "📐 Machine Learning Models & Metrics",
    "🔮 Live Placement & Salary Predictor"
])

# ==================== TAB 1: DATASET OVERVIEW & UPLOAD ====================
with tab1:
    st.subheader("📌 Key Dataset Metrics & Upload Summary")
    meta_info = get_dataset_metadata(filtered_df)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #1d4ed8;">{meta_info['num_rows']:,}</div>
            <div class="metric-lbl">Total Records</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #0f172a;">{meta_info['num_cols']}</div>
            <div class="metric-lbl">Total Columns</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #059669;">{meta_info['placement_rate']}%</div>
            <div class="metric-lbl">Placement Rate</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #d97706;">₹{meta_info['avg_salary']} LPA</div>
            <div class="metric-lbl">Avg Salary Package</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #475569;">{meta_info['memory_mb']} MB</div>
            <div class="metric-lbl">Memory Footprint</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    col_left, col_right = st.columns([1.3, 1])
    
    with col_left:
        st.markdown("### 📋 Dataset Interactive Explorer")
        st.dataframe(filtered_df.head(100), height=420, use_container_width=True)
        
    with col_right:
        st.markdown("### 🔍 Column Schema & Data Types")
        st.dataframe(meta_info["columns_summary"], height=420, use_container_width=True)
        
    st.markdown("### 📈 Numerical Features Statistical Summary")
    st.dataframe(filtered_df.describe().T.style.format("{:.2f}"), use_container_width=True)

# ==================== TAB 2: MISSING VALUES & DATA PREPROCESSING ====================
with tab2:
    st.subheader("🧹 Data Preprocessing & Missing Values Profiling")
    
    missing_summary = get_missing_values_summary(filtered_df)
    
    col_m1, col_m2 = st.columns([1, 1.2])
    
    with col_m1:
        st.markdown("#### Preprocessing & Imputation Strategy")
        st.info("""
        **Pipeline Preprocessing Steps**:
        - **Numerical Imputation**: SimpleImputer using **Median** strategy for missing numeric test scores and ratings.
        - **Numerical Scaling**: **StandardScaler** to standardize features to zero mean and unit variance (critical for Logistic Regression & Linear Regression).
        - **Categorical Encoding**: **OneHotEncoder** with handle_unknown='ignore' for robust categorical feature transformation.
        - **Column Transformer**: Encapsulates preprocessing for seamless end-to-end Machine Learning modeling.
        """)
        
        if not missing_summary.empty:
            st.markdown("#### Null Value Summary Table")
            st.dataframe(missing_summary, use_container_width=True)
        else:
            st.success("🎉 No missing values detected in the current dataset slice!")
            
    with col_m2:
        st.markdown("#### Null Records Profile")
        fig_missing = plot_missing_values_chart(missing_summary)
        st.plotly_chart(fig_missing, use_container_width=True, key="eda_missing_chart")

# ==================== TAB 3: EDA & VISUALIZATIONS ====================
with tab3:
    st.subheader("📈 Exploratory Data Analysis (EDA)")
    
    eda_option = st.selectbox(
        "Select EDA Analysis View:",
        [
            "1. Univariate Feature Distributions",
            "2. Categorical Features vs Placement Outcome",
            "3. Feature Performance & Box Plots",
            "4. Numerical Features Correlation Matrix",
            "5. CGPA vs Salary Package (Linear Regression Fit)"
        ]
    )
    
    st.markdown("---")
    
    if eda_option.startswith("1"):
        st.markdown("### Univariate Feature Distributions")
        num_cat_cols = [c for c in filtered_df.columns if c not in ["StudentID"]]
        default_idx = num_cat_cols.index("CGPA") if "CGPA" in num_cat_cols else 0
        selected_col = st.selectbox("Select Feature to Plot:", options=num_cat_cols, index=default_idx)
        fig_dist = plot_univariate_distribution(filtered_df, selected_col)
        st.plotly_chart(fig_dist, use_container_width=True, key=f"eda_univariate_{selected_col}")
        
    elif eda_option.startswith("2"):
        st.markdown("### Categorical Features vs Placement Status")
        cat_options = [c for c in ["Gender", "Stream", "Specialisation", "CollegeTier", "HistoryOfBacklogs", "Hostel"] if c in filtered_df.columns]
        if cat_options:
            selected_cat = st.selectbox("Select Categorical Feature:", options=cat_options, index=0)
            fig_cat = plot_placement_by_category(filtered_df, selected_cat)
            st.plotly_chart(fig_cat, use_container_width=True, key=f"eda_cat_{selected_cat}")
        else:
            st.info("No categorical features available for this breakdown.")
        
    elif eda_option.startswith("3"):
        st.markdown("### Feature Performance vs Placement Status")
        box_options = [c for c in ["CGPA", "CodingTestScore", "AptitudeTestScore", "SoftSkillsRating", "MockInterviewScore", "AttendancePercent", "Internships", "Salary Package"] if c in filtered_df.columns]
        if box_options:
            selected_box = st.selectbox("Select Numerical Feature for Box Plot:", options=box_options, index=0)
            fig_box = plot_feature_vs_placement_box(filtered_df, selected_box)
            st.plotly_chart(fig_box, use_container_width=True, key=f"eda_box_{selected_box}")
        
        if "AptitudeTestScore" in filtered_df.columns and "CodingTestScore" in filtered_df.columns:
            st.markdown("#### Scatter Plot: Aptitude Score vs Coding Test Score")
            color_kwargs = {}
            if "PlacementStatus" in filtered_df.columns:
                color_kwargs["color"] = filtered_df["PlacementStatus"].map({0: "Not Placed", 1: "Placed"})
                color_kwargs["color_discrete_map"] = {"Not Placed": COLOR_NOT_PLACED, "Placed": COLOR_PLACED}

            fig_scat = px.scatter(
                filtered_df,
                x="AptitudeTestScore",
                y="CodingTestScore",
                title="Coding Test Score vs Aptitude Test Score",
                opacity=0.8,
                **color_kwargs
            )
            fig_scat = apply_professional_layout(fig_scat, "Coding Test Score vs Aptitude Test Score", height=500)
            st.plotly_chart(fig_scat, use_container_width=True, key="eda_scatter_aptitude_coding")
        
    elif eda_option.startswith("4"):
        st.markdown("### Features Correlation Matrix")
        fig_corr = plot_correlation_heatmap(filtered_df)
        st.plotly_chart(fig_corr, use_container_width=True, key="eda_corr_heatmap")

    elif eda_option.startswith("5"):
        st.markdown("### 📐 CGPA vs Salary Package — Linear Regression Line Fit")
        st.info("""
        **Linear Regression Analysis**: CGPA and Salary Package demonstrate a clear linear pattern. 
        The fitted Linear Regression trendline below models expected salary package scaling as a function of CGPA.
        """)
        fig_lr = plot_cgpa_vs_salary_regression(filtered_df)
        st.plotly_chart(fig_lr, use_container_width=True, key="eda_cgpa_salary_reg")

# ==================== TAB 4: MODELS & EVALUATION ====================
with tab4:
    sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5, sub_tab6, sub_tab7, sub_tab8, sub_tab9, sub_tab10 = st.tabs([
        "🎯 Logistic Regression Classifier",
        "🌴 ID3 Decision Tree",
        "🌾 C4.5 Decision Tree",
        "🌲 CART (Karth) Decision Tree",
        "🚀 Standard GBM (Gradient Boosting)",
        "💡 LightGBM Model",
        "⚡ XGBoost Model",
        "🎒 Bagging & Random Forest",
        "📈 Linear Regression Regressor",
        "⚖️ Model Comparison Table"
    ])
    
    log_m = metadata.get("logistic_metrics", {})
    id3_c_m = metadata.get("id3_classifier_metrics", {})
    id3_r_m = metadata.get("id3_regressor_metrics", {})
    c45_c_m = metadata.get("c45_classifier_metrics", {})
    c45_r_m = metadata.get("c45_regressor_metrics", {})
    cart_c_m = metadata.get("cart_classifier_metrics", {})
    cart_r_m = metadata.get("cart_regressor_metrics", {})
    dt_c_m = metadata.get("dt_classifier_metrics", cart_c_m)
    dt_r_m = metadata.get("dt_regressor_metrics", cart_r_m)
    lr_m = metadata.get("lr_metrics", {})
    simple_lr_m = metadata.get("simple_lr_metrics", {})

    rf_c_m = metadata.get("rf_classifier_metrics", {})
    rf_r_m = metadata.get("rf_regressor_metrics", {})
    bag_c_m = metadata.get("bagging_classifier_metrics", {})
    bag_r_m = metadata.get("bagging_regressor_metrics", {})
    gbm_c_m = metadata.get("gbm_classifier_metrics", {})
    gbm_r_m = metadata.get("gbm_regressor_metrics", {})
    lgb_c_m = metadata.get("lightgbm_classifier_metrics", {})
    lgb_r_m = metadata.get("lightgbm_regressor_metrics", {})
    xgb_c_m = metadata.get("xgb_classifier_metrics", {})
    xgb_r_m = metadata.get("xgb_regressor_metrics", {})
    ada_c_m = metadata.get("adaboost_classifier_metrics", {})
    ada_r_m = metadata.get("adaboost_regressor_metrics", {})

    def render_dt_tab(clf_m, reg_m, title, algo_name, info_text):
        st.markdown(f"### {title}")
        st.info(info_text)
        
        # 🧪 Impurity Measures & Tree Structure Analysis Section
        imp_m = clf_m.get("impurity_metrics", {})
        st.markdown(f"#### 🧪 {algo_name} Impurity Measures & Tree Node Statistics")
        
        ic1, ic2, ic3, ic4, ic5 = st.columns(5)
        with ic1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #8b5cf6;">{imp_m.get('criterion', 'N/A')}</div>
                <div class="metric-lbl">Split Criterion</div>
            </div>
            """, unsafe_allow_html=True)
        with ic2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #dc2626;">{imp_m.get('root_impurity', 0):.4f}</div>
                <div class="metric-lbl">Root Node Impurity</div>
            </div>
            """, unsafe_allow_html=True)
        with ic3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #d97706;">{imp_m.get('weighted_leaf_impurity', 0):.4f}</div>
                <div class="metric-lbl">Weighted Leaf Impurity</div>
            </div>
            """, unsafe_allow_html=True)
        with ic4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #059669;">{imp_m.get('impurity_reduction', 0):.4f}</div>
                <div class="metric-lbl">Total Impurity Gain ΔI</div>
            </div>
            """, unsafe_allow_html=True)
        with ic5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #1d4ed8;">{imp_m.get('leaf_count', 0)} (Depth {imp_m.get('max_depth', 0)})</div>
                <div class="metric-lbl">Total Tree Leaves</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_imp_chart, col_imp_formula = st.columns([1.3, 1])
        with col_imp_chart:
            st.plotly_chart(plot_impurity_comparison_curves(), use_container_width=True, key=f"dt_imp_curve_{algo_name.replace(' ', '_')}")
        with col_imp_formula:
            st.markdown("##### 📐 Impurity Measures Theoretical Formulas")
            st.markdown("""
            **Gini Impurity (CART)**:
            """)
            st.latex(r"Gini(D) = 1 - \sum_{i=1}^{C} p_i^2")
            st.markdown("""
            **Entropy (ID3 & C4.5)**:
            """)
            st.latex(r"Entropy(D) = -\sum_{i=1}^{C} p_i \log_2(p_i)")
            st.markdown("""
            **Information Gain (ID3)**:
            """)
            st.latex(r"Gain(D, A) = Entropy(D) - \sum \frac{|D_v|}{|D|} Entropy(D_v)")
            st.markdown("""
            **Gain Ratio (C4.5)**:
            """)
            st.latex(r"GainRatio(D, A) = \frac{Gain(D, A)}{SplitInfo(A)}")

        st.markdown("---")
        st.markdown("#### 🟢 Classification Metrics (Placement Status)")
        dtc1, dtc2, dtc3, dtc4, dtc5 = st.columns(5)
        with dtc1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #059669;">{clf_m.get('accuracy', 0) * 100:.2f}%</div>
                <div class="metric-lbl">Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with dtc2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #1d4ed8;">{clf_m.get('precision', 0) * 100:.2f}%</div>
                <div class="metric-lbl">Precision</div>
            </div>
            """, unsafe_allow_html=True)
        with dtc3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #d97706;">{clf_m.get('recall', 0) * 100:.2f}%</div>
                <div class="metric-lbl">Recall</div>
            </div>
            """, unsafe_allow_html=True)
        with dtc4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #7c3aed;">{clf_m.get('f1_score', 0):.4f}</div>
                <div class="metric-lbl">F1 Score</div>
            </div>
            """, unsafe_allow_html=True)
        with dtc5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #0f172a;">{clf_m.get('roc_auc', 0):.4f}</div>
                <div class="metric-lbl">ROC-AUC Score</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        col_dt_cm, col_dt_roc = st.columns(2)
        with col_dt_cm:
            st.markdown(f"#### 🔲 Confusion Matrix ({algo_name})")
            cm_dt_data = clf_m.get("confusion_matrix", [[0, 0], [0, 0]])
            st.plotly_chart(plot_confusion_matrix(cm_dt_data), use_container_width=True, key=f"dt_cm_{algo_name.replace(' ', '_')}")
        with col_dt_roc:
            st.markdown(f"#### 📈 ROC Curve ({algo_name})")
            roc_dt_data = clf_m.get("roc_curve", {"fpr": [0, 1], "tpr": [0, 1]})
            st.plotly_chart(plot_roc_curve(roc_dt_data["fpr"], roc_dt_data["tpr"], clf_m.get("roc_auc", 0.5)), use_container_width=True, key=f"dt_roc_{algo_name.replace(' ', '_')}")
            
        st.markdown("---")
        st.markdown(f"#### 🔵 Regression Metrics ({algo_name} — Salary Package)")
        dtr1, dtr2, dtr3, dtr4 = st.columns(4)
        with dtr1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #1d4ed8;">{reg_m.get('r2_score', 0):.4f}</div>
                <div class="metric-lbl">{algo_name} R² Score</div>
            </div>
            """, unsafe_allow_html=True)
        with dtr2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #059669;">₹{reg_m.get('mae', 0)} LPA</div>
                <div class="metric-lbl">Mean Absolute Error</div>
            </div>
            """, unsafe_allow_html=True)
        with dtr3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #475569;">{reg_m.get('mse', 0)}</div>
                <div class="metric-lbl">Mean Squared Error</div>
            </div>
            """, unsafe_allow_html=True)
        with dtr4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #d97706;">₹{reg_m.get('rmse', 0)} LPA</div>
                <div class="metric-lbl">Root Mean Squared Error</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"#### 🌲 {algo_name} Feature Importances")
        dt_clf_imp = clf_m.get("feature_importances", [])
        st.plotly_chart(plot_feature_importances(dt_clf_imp, f"{algo_name} Classifier — Feature Importances"), use_container_width=True, key=f"dt_feat_{algo_name.replace(' ', '_')}")

    # ------------------ SUB-TAB 1: LOGISTIC REGRESSION ------------------
    with sub_tab1:
        st.markdown("### 🟢 Logistic Regression — Placement Status Classifier Metrics")
        st.info("""
        **Logistic Regression Classification**: Predicts binary `PlacementStatus` (1 = Placed, 0 = Not Placed) 
        using sigmoid probability modeling across student academic performance, skills, and background features.
        """)
        
        lc1, lc2, lc3, lc4, lc5 = st.columns(5)
        with lc1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #059669;">{log_m.get('accuracy', 0) * 100:.2f}%</div>
                <div class="metric-lbl">Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with lc2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #1d4ed8;">{log_m.get('precision', 0) * 100:.2f}%</div>
                <div class="metric-lbl">Precision</div>
            </div>
            """, unsafe_allow_html=True)
        with lc3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #d97706;">{log_m.get('recall', 0) * 100:.2f}%</div>
                <div class="metric-lbl">Recall (Sensitivity)</div>
            </div>
            """, unsafe_allow_html=True)
        with lc4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #7c3aed;">{log_m.get('f1_score', 0):.4f}</div>
                <div class="metric-lbl">F1 Score</div>
            </div>
            """, unsafe_allow_html=True)
        with lc5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #0f172a;">{log_m.get('roc_auc', 0):.4f}</div>
                <div class="metric-lbl">ROC-AUC Score</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        col_cm, col_roc = st.columns(2)
        with col_cm:
            st.markdown("#### 🔲 Confusion Matrix")
            cm_data = log_m.get("confusion_matrix", [[0, 0], [0, 0]])
            st.plotly_chart(plot_confusion_matrix(cm_data), use_container_width=True, key="log_cm")
        with col_roc:
            st.markdown("#### 📈 Receiver Operating Characteristic (ROC) Curve")
            roc_data = log_m.get("roc_curve", {"fpr": [0, 1], "tpr": [0, 1]})
            st.plotly_chart(plot_roc_curve(roc_data["fpr"], roc_data["tpr"], log_m.get("roc_auc", 0.5)), use_container_width=True, key="log_roc")
            
        st.markdown("---")
        st.markdown("#### ⚖️ Feature Odds Ratios & Placement Influence")
        odds_recs = log_m.get("odds_ratios", [])
        st.plotly_chart(plot_odds_ratios(odds_recs), use_container_width=True, key="log_odds")

    # ------------------ SUB-TAB 2: ID3 DECISION TREE ------------------
    with sub_tab2:
        render_dt_tab(
            id3_c_m, id3_r_m,
            "🌴 ID3 Decision Tree — Information Gain / Entropy Metrics",
            "ID3",
            "**ID3 (Iterative Dichotomiser 3)**: Uses Information Gain (Entropy reduction) to select split attributes at each node. Categorical and discrete feature partitions are formed by evaluating decrease in disorder."
        )

    # ------------------ SUB-TAB 3: C4.5 DECISION TREE ------------------
    with sub_tab3:
        render_dt_tab(
            c45_c_m, c45_r_m,
            "🌾 C4.5 Decision Tree — Gain Ratio & Pruning Metrics",
            "C4.5",
            "**C4.5 Algorithm**: Ross Quinlan's extension of ID3 using Gain Ratio to overcome ID3's bias towards features with many values. Incorporates continuous attribute splits and post-pruning thresholds to avoid overfitting."
        )

    # ------------------ SUB-TAB 4: CART / KARTH DECISION TREE ------------------
    with sub_tab4:
        render_dt_tab(
            cart_c_m, cart_r_m,
            "🌲 CART (Karth) Decision Tree — Gini Impurity & MSE Metrics",
            "CART (Karth)",
            "**CART (Karth - Classification and Regression Trees)**: Breiman et al. algorithm constructing binary decision trees. Uses Gini Impurity for binary placement classification and Mean Squared Error (MSE) for salary package regression."
        )

    # ------------------ SUB-TAB 5: STANDARD GBM ------------------
    with sub_tab5:
        st.markdown("### 🚀 Standard GBM (Gradient Boosting Machine) Metrics")
        st.info("""
        **Standard Gradient Boosting Machine (GBM)**: Builds decision trees sequentially, optimizing pseudo-residuals 
        of loss functions to minimize structural prediction error.
        """)
        
        gbm1, gbm2, gbm3, gbm4, gbm5 = st.columns(5)
        with gbm1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #059669;">{gbm_c_m.get('accuracy', 0) * 100:.2f}%</div>
                <div class="metric-lbl">GBM Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with gbm2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #1d4ed8;">{gbm_c_m.get('precision', 0) * 100:.2f}%</div>
                <div class="metric-lbl">GBM Precision</div>
            </div>
            """, unsafe_allow_html=True)
        with gbm3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #d97706;">{gbm_c_m.get('recall', 0) * 100:.2f}%</div>
                <div class="metric-lbl">GBM Recall</div>
            </div>
            """, unsafe_allow_html=True)
        with gbm4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #7c3aed;">{gbm_c_m.get('f1_score', 0):.4f}</div>
                <div class="metric-lbl">GBM F1 Score</div>
            </div>
            """, unsafe_allow_html=True)
        with gbm5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #0f172a;">R² {gbm_r_m.get('r2_score', 0):.4f}</div>
                <div class="metric-lbl">GBM Salary R² Score</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_gbm_cm, col_gbm_roc = st.columns(2)
        with col_gbm_cm:
            st.markdown("##### 🔲 Standard GBM Confusion Matrix")
            cm_gbm = gbm_c_m.get("confusion_matrix", [[0, 0], [0, 0]])
            st.plotly_chart(plot_confusion_matrix(cm_gbm), use_container_width=True, key="gbm_cm")
        with col_gbm_roc:
            st.markdown("##### 📈 Standard GBM ROC Curve")
            roc_gbm = gbm_c_m.get("roc_curve", {"fpr": [0, 1], "tpr": [0, 1]})
            st.plotly_chart(plot_roc_curve(roc_gbm["fpr"], roc_gbm["tpr"], gbm_c_m.get("roc_auc", 0.5)), use_container_width=True, key="gbm_roc")

        st.markdown("---")
        st.markdown("#### 🚀 Standard GBM Feature Importances")
        st.plotly_chart(plot_feature_importances(gbm_c_m.get("feature_importances", []), "Standard GBM — Feature Importances"), use_container_width=True, key="gbm_feat")

    # ------------------ SUB-TAB 6: LIGHTGBM ------------------
    with sub_tab6:
        st.markdown("### 💡 LightGBM (Light Gradient Boosting Machine) Metrics")
        st.info("""
        **LightGBM**: Fast, high-performance gradient boosting framework based on decision tree algorithms 
        using leaf-wise tree growth and histogram-based feature binning.
        """)
        
        lgb1, lgb2, lgb3, lgb4, lgb5 = st.columns(5)
        with lgb1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #059669;">{lgb_c_m.get('accuracy', 0) * 100:.2f}%</div>
                <div class="metric-lbl">LightGBM Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with lgb2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #1d4ed8;">{lgb_c_m.get('precision', 0) * 100:.2f}%</div>
                <div class="metric-lbl">LightGBM Precision</div>
            </div>
            """, unsafe_allow_html=True)
        with lgb3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #d97706;">{lgb_c_m.get('recall', 0) * 100:.2f}%</div>
                <div class="metric-lbl">LightGBM Recall</div>
            </div>
            """, unsafe_allow_html=True)
        with lgb4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #7c3aed;">{lgb_c_m.get('f1_score', 0):.4f}</div>
                <div class="metric-lbl">LightGBM F1 Score</div>
            </div>
            """, unsafe_allow_html=True)
        with lgb5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #0f172a;">R² {lgb_r_m.get('r2_score', 0):.4f}</div>
                <div class="metric-lbl">LightGBM Salary R²</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_lgb_cm, col_lgb_roc = st.columns(2)
        with col_lgb_cm:
            st.markdown("##### 🔲 LightGBM Confusion Matrix")
            cm_lgb = lgb_c_m.get("confusion_matrix", [[0, 0], [0, 0]])
            st.plotly_chart(plot_confusion_matrix(cm_lgb), use_container_width=True, key="lgb_cm")
        with col_lgb_roc:
            st.markdown("##### 📈 LightGBM ROC Curve")
            roc_lgb = lgb_c_m.get("roc_curve", {"fpr": [0, 1], "tpr": [0, 1]})
            st.plotly_chart(plot_roc_curve(roc_lgb["fpr"], roc_lgb["tpr"], lgb_c_m.get("roc_auc", 0.5)), use_container_width=True, key="lgb_roc")

        st.markdown("---")
        st.markdown("#### 💡 LightGBM Feature Importances")
        st.plotly_chart(plot_feature_importances(lgb_c_m.get("feature_importances", []), "LightGBM — Feature Importances"), use_container_width=True, key="lgb_feat")

    # ------------------ SUB-TAB 7: XGBOOST ------------------
    with sub_tab7:
        st.markdown("### ⚡ XGBoost (eXtreme Gradient Boosting) Metrics")
        st.info("""
        **XGBoost**: Optimized gradient boosting library designed for efficient, flexible, and portable 
        machine learning with built-in L1/L2 regularization and depth-wise tree growth.
        """)
        
        xgb1, xgb2, xgb3, xgb4, xgb5 = st.columns(5)
        with xgb1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #059669;">{xgb_c_m.get('accuracy', 0) * 100:.2f}%</div>
                <div class="metric-lbl">XGBoost Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with xgb2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #1d4ed8;">{xgb_c_m.get('precision', 0) * 100:.2f}%</div>
                <div class="metric-lbl">XGBoost Precision</div>
            </div>
            """, unsafe_allow_html=True)
        with xgb3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #d97706;">{xgb_c_m.get('recall', 0) * 100:.2f}%</div>
                <div class="metric-lbl">XGBoost Recall</div>
            </div>
            """, unsafe_allow_html=True)
        with xgb4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #7c3aed;">{xgb_c_m.get('f1_score', 0):.4f}</div>
                <div class="metric-lbl">XGBoost F1 Score</div>
            </div>
            """, unsafe_allow_html=True)
        with xgb5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #0f172a;">R² {xgb_r_m.get('r2_score', 0):.4f}</div>
                <div class="metric-lbl">XGBoost Salary R²</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_xgb_cm, col_xgb_roc = st.columns(2)
        with col_xgb_cm:
            st.markdown("##### 🔲 XGBoost Confusion Matrix")
            cm_xgb = xgb_c_m.get("confusion_matrix", [[0, 0], [0, 0]])
            st.plotly_chart(plot_confusion_matrix(cm_xgb), use_container_width=True, key="xgb_cm")
        with col_xgb_roc:
            st.markdown("##### 📈 XGBoost ROC Curve")
            roc_xgb = xgb_c_m.get("roc_curve", {"fpr": [0, 1], "tpr": [0, 1]})
            st.plotly_chart(plot_roc_curve(roc_xgb["fpr"], roc_xgb["tpr"], xgb_c_m.get("roc_auc", 0.5)), use_container_width=True, key="xgb_roc")

        st.markdown("---")
        st.markdown("#### ⚡ XGBoost Feature Importances")
        st.plotly_chart(plot_feature_importances(xgb_c_m.get("feature_importances", []), "XGBoost — Feature Importances"), use_container_width=True, key="xgb_feat")

    # ------------------ SUB-TAB 8: BAGGING & RANDOM FOREST ------------------
    with sub_tab8:
        st.markdown("### 🎒 Bagging & Random Forest Ensemble Metrics")
        
        rfc1, rfc2, rfc3, rfc4, rfc5 = st.columns(5)
        with rfc1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #059669;">{rf_c_m.get('accuracy', 0) * 100:.2f}%</div>
                <div class="metric-lbl">RF Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with rfc2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #1d4ed8;">{rf_c_m.get('precision', 0) * 100:.2f}%</div>
                <div class="metric-lbl">RF Precision</div>
            </div>
            """, unsafe_allow_html=True)
        with rfc3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #d97706;">{rf_c_m.get('recall', 0) * 100:.2f}%</div>
                <div class="metric-lbl">RF Recall</div>
            </div>
            """, unsafe_allow_html=True)
        with rfc4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #7c3aed;">{rf_c_m.get('f1_score', 0):.4f}</div>
                <div class="metric-lbl">RF F1 Score</div>
            </div>
            """, unsafe_allow_html=True)
        with rfc5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #0f172a;">R² {rf_r_m.get('r2_score', 0):.4f}</div>
                <div class="metric-lbl">RF Salary R²</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🌴 Bagging Classifier & Regressor Performance")
        bag1, bag2, bag3, bag4, bag5 = st.columns(5)
        with bag1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #059669;">{bag_c_m.get('accuracy', 0) * 100:.2f}%</div>
                <div class="metric-lbl">Bagging Acc</div>
            </div>
            """, unsafe_allow_html=True)
        with bag2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #1d4ed8;">{bag_c_m.get('precision', 0) * 100:.2f}%</div>
                <div class="metric-lbl">Bagging Prec</div>
            </div>
            """, unsafe_allow_html=True)
        with bag3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #d97706;">{bag_c_m.get('recall', 0) * 100:.2f}%</div>
                <div class="metric-lbl">Bagging Recall</div>
            </div>
            """, unsafe_allow_html=True)
        with bag4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #7c3aed;">{bag_c_m.get('f1_score', 0):.4f}</div>
                <div class="metric-lbl">Bagging F1</div>
            </div>
            """, unsafe_allow_html=True)
        with bag5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #0f172a;">R² {bag_r_m.get('r2_score', 0):.4f}</div>
                <div class="metric-lbl">Bagging Salary R²</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📊 Random Forest Feature Importances")
        st.plotly_chart(plot_feature_importances(rf_c_m.get("feature_importances", []), "Random Forest — Feature Importances"), use_container_width=True, key="rf_feat")

    # ------------------ SUB-TAB 9: LINEAR REGRESSION ------------------
    with sub_tab9:
        st.markdown("### 📊 Linear Regression — Salary Package Regressor Metrics")
        
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #1d4ed8;">{lr_m.get('r2_score', 0):.4f}</div>
                <div class="metric-lbl">Multi-Feature R² Score</div>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #059669;">₹{lr_m.get('mae', 0)} LPA</div>
                <div class="metric-lbl">Mean Absolute Error</div>
            </div>
            """, unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #475569;">{lr_m.get('mse', 0)}</div>
                <div class="metric-lbl">Mean Squared Error</div>
            </div>
            """, unsafe_allow_html=True)
        with r4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #d97706;">₹{lr_m.get('rmse', 0)} LPA</div>
                <div class="metric-lbl">Root Mean Squared Error</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        col_coef, col_simple = st.columns([1.2, 1])
        with col_coef:
            st.markdown("#### 🔍 Feature Impact (Linear Regression Coefficients)")
            coef_records = metadata.get("feature_coefficients", [])
            if coef_records:
                coef_df = pd.DataFrame(coef_records)
                fig_coef = px.bar(
                    coef_df,
                    x="Coefficient",
                    y="Feature",
                    orientation="h",
                    title="Linear Regression Feature Weights / Coefficients",
                    color="Coefficient",
                    color_continuous_scale="RdBu"
                )
                fig_coef = apply_professional_layout(fig_coef, "Linear Regression Feature Weights / Coefficients", height=420)
                fig_coef.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_coef, use_container_width=True, key="lr_coef_chart")
            else:
                st.info("Feature coefficients unavailable.")
                
        with col_simple:
            st.markdown("#### 📈 Simple Linear Regression (CGPA → Salary)")
            slope_val = simple_lr_m.get("slope", 0)
            intercept_val = simple_lr_m.get("intercept", 0)
            corr_val = simple_lr_m.get("correlation", 0)
            
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 15px; border-top: 4px solid #1d4ed8;">
                <div class="metric-val" style="color: #1d4ed8; font-size: 1.3rem;">Salary = {slope_val:.2f} × CGPA + ({intercept_val:.2f})</div>
                <div class="metric-lbl">Fitted OLS Line Equation</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 15px; border-top: 4px solid #059669;">
                <div class="metric-val" style="color: #059669;">{corr_val:.4f}</div>
                <div class="metric-lbl">Pearson Correlation Coefficient (r)</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid #7c3aed;">
                <div class="metric-val" style="color: #7c3aed;">{simple_lr_m.get('r2_score', 0):.4f}</div>
                <div class="metric-lbl">Simple LR R² Score</div>
            </div>
            """, unsafe_allow_html=True)

    # ------------------ SUB-TAB 10: MODEL COMPARISON TABLE ------------------
    with sub_tab10:
        st.markdown("### ⚖️ All Models Performance Summary & Side-by-Side Comparison")
        
        summary_data = [
            # Classification
            {
                "Model Name": "Logistic Regression Classifier",
                "Category": "Base Model",
                "Task Type": "Placement Status (Classification)",
                "Accuracy (%)": f"{log_m.get('accuracy', 0)*100:.2f}%",
                "Precision (%)": f"{log_m.get('precision', 0)*100:.2f}%",
                "Recall (%)": f"{log_m.get('recall', 0)*100:.2f}%",
                "F1 Score": f"{log_m.get('f1_score', 0):.4f}",
                "ROC-AUC / R²": f"ROC-AUC: {log_m.get('roc_auc', 0):.4f}"
            },
            {
                "Model Name": "ID3 Decision Tree Classifier",
                "Category": "Decision Tree (Entropy)",
                "Task Type": "Placement Status (Classification)",
                "Accuracy (%)": f"{id3_c_m.get('accuracy', 0)*100:.2f}%",
                "Precision (%)": f"{id3_c_m.get('precision', 0)*100:.2f}%",
                "Recall (%)": f"{id3_c_m.get('recall', 0)*100:.2f}%",
                "F1 Score": f"{id3_c_m.get('f1_score', 0):.4f}",
                "ROC-AUC / R²": f"ROC-AUC: {id3_c_m.get('roc_auc', 0):.4f}"
            },
            {
                "Model Name": "C4.5 Decision Tree Classifier",
                "Category": "Decision Tree (Gain Ratio)",
                "Task Type": "Placement Status (Classification)",
                "Accuracy (%)": f"{c45_c_m.get('accuracy', 0)*100:.2f}%",
                "Precision (%)": f"{c45_c_m.get('precision', 0)*100:.2f}%",
                "Recall (%)": f"{c45_c_m.get('recall', 0)*100:.2f}%",
                "F1 Score": f"{c45_c_m.get('f1_score', 0):.4f}",
                "ROC-AUC / R²": f"ROC-AUC: {c45_c_m.get('roc_auc', 0):.4f}"
            },
            {
                "Model Name": "CART (Karth) Classifier",
                "Category": "Decision Tree (Gini)",
                "Task Type": "Placement Status (Classification)",
                "Accuracy (%)": f"{cart_c_m.get('accuracy', 0)*100:.2f}%",
                "Precision (%)": f"{cart_c_m.get('precision', 0)*100:.2f}%",
                "Recall (%)": f"{cart_c_m.get('recall', 0)*100:.2f}%",
                "F1 Score": f"{cart_c_m.get('f1_score', 0):.4f}",
                "ROC-AUC / R²": f"ROC-AUC: {cart_c_m.get('roc_auc', 0):.4f}"
            },
            {
                "Model Name": "Standard GBM Classifier",
                "Category": "Gradient Boosting",
                "Task Type": "Placement Status (Classification)",
                "Accuracy (%)": f"{gbm_c_m.get('accuracy', 0)*100:.2f}%",
                "Precision (%)": f"{gbm_c_m.get('precision', 0)*100:.2f}%",
                "Recall (%)": f"{gbm_c_m.get('recall', 0)*100:.2f}%",
                "F1 Score": f"{gbm_c_m.get('f1_score', 0):.4f}",
                "ROC-AUC / R²": f"ROC-AUC: {gbm_c_m.get('roc_auc', 0):.4f}"
            },
            {
                "Model Name": "LightGBM Classifier",
                "Category": "Gradient Boosting",
                "Task Type": "Placement Status (Classification)",
                "Accuracy (%)": f"{lgb_c_m.get('accuracy', 0)*100:.2f}%",
                "Precision (%)": f"{lgb_c_m.get('precision', 0)*100:.2f}%",
                "Recall (%)": f"{lgb_c_m.get('recall', 0)*100:.2f}%",
                "F1 Score": f"{lgb_c_m.get('f1_score', 0):.4f}",
                "ROC-AUC / R²": f"ROC-AUC: {lgb_c_m.get('roc_auc', 0):.4f}"
            },
            {
                "Model Name": "XGBoost Classifier",
                "Category": "Gradient Boosting",
                "Task Type": "Placement Status (Classification)",
                "Accuracy (%)": f"{xgb_c_m.get('accuracy', 0)*100:.2f}%",
                "Precision (%)": f"{xgb_c_m.get('precision', 0)*100:.2f}%",
                "Recall (%)": f"{xgb_c_m.get('recall', 0)*100:.2f}%",
                "F1 Score": f"{xgb_c_m.get('f1_score', 0):.4f}",
                "ROC-AUC / R²": f"ROC-AUC: {xgb_c_m.get('roc_auc', 0):.4f}"
            },
            {
                "Model Name": "Random Forest Classifier",
                "Category": "Bagging Ensemble",
                "Task Type": "Placement Status (Classification)",
                "Accuracy (%)": f"{rf_c_m.get('accuracy', 0)*100:.2f}%",
                "Precision (%)": f"{rf_c_m.get('precision', 0)*100:.2f}%",
                "Recall (%)": f"{rf_c_m.get('recall', 0)*100:.2f}%",
                "F1 Score": f"{rf_c_m.get('f1_score', 0):.4f}",
                "ROC-AUC / R²": f"ROC-AUC: {rf_c_m.get('roc_auc', 0):.4f}"
            },
            {
                "Model Name": "AdaBoost Classifier",
                "Category": "Adaptive Boosting",
                "Task Type": "Placement Status (Classification)",
                "Accuracy (%)": f"{ada_c_m.get('accuracy', 0)*100:.2f}%",
                "Precision (%)": f"{ada_c_m.get('precision', 0)*100:.2f}%",
                "Recall (%)": f"{ada_c_m.get('recall', 0)*100:.2f}%",
                "F1 Score": f"{ada_c_m.get('f1_score', 0):.4f}",
                "ROC-AUC / R²": f"ROC-AUC: {ada_c_m.get('roc_auc', 0):.4f}"
            },
            # Regression
            {
                "Model Name": "Multi-Feature Linear Regression",
                "Category": "Base Model",
                "Task Type": "Salary Package (Regression)",
                "Accuracy (%)": "N/A (MAE: ₹" + str(lr_m.get('mae', 0)) + ")",
                "Precision (%)": "N/A (MSE: " + str(lr_m.get('mse', 0)) + ")",
                "Recall (%)": "N/A (RMSE: ₹" + str(lr_m.get('rmse', 0)) + ")",
                "F1 Score": "N/A",
                "ROC-AUC / R²": f"R²: {lr_m.get('r2_score', 0):.4f}"
            },
            {
                "Model Name": "ID3 Decision Tree Regressor",
                "Category": "Decision Tree (MAE)",
                "Task Type": "Salary Package (Regression)",
                "Accuracy (%)": "N/A (MAE: ₹" + str(id3_r_m.get('mae', 0)) + ")",
                "Precision (%)": "N/A (MSE: " + str(id3_r_m.get('mse', 0)) + ")",
                "Recall (%)": "N/A (RMSE: ₹" + str(id3_r_m.get('rmse', 0)) + ")",
                "F1 Score": "N/A",
                "ROC-AUC / R²": f"R²: {id3_r_m.get('r2_score', 0):.4f}"
            },
            {
                "Model Name": "C4.5 Decision Tree Regressor",
                "Category": "Decision Tree (Friedman MSE)",
                "Task Type": "Salary Package (Regression)",
                "Accuracy (%)": "N/A (MAE: ₹" + str(c45_r_m.get('mae', 0)) + ")",
                "Precision (%)": "N/A (MSE: " + str(c45_r_m.get('mse', 0)) + ")",
                "Recall (%)": "N/A (RMSE: ₹" + str(c45_r_m.get('rmse', 0)) + ")",
                "F1 Score": "N/A",
                "ROC-AUC / R²": f"R²: {c45_r_m.get('r2_score', 0):.4f}"
            },
            {
                "Model Name": "CART (Karth) Regressor",
                "Category": "Decision Tree (MSE)",
                "Task Type": "Salary Package (Regression)",
                "Accuracy (%)": "N/A (MAE: ₹" + str(cart_r_m.get('mae', 0)) + ")",
                "Precision (%)": "N/A (MSE: " + str(cart_r_m.get('mse', 0)) + ")",
                "Recall (%)": "N/A (RMSE: ₹" + str(cart_r_m.get('rmse', 0)) + ")",
                "F1 Score": "N/A",
                "ROC-AUC / R²": f"R²: {cart_r_m.get('r2_score', 0):.4f}"
            },
            {
                "Model Name": "Standard GBM Regressor",
                "Category": "Gradient Boosting",
                "Task Type": "Salary Package (Regression)",
                "Accuracy (%)": "N/A (MAE: ₹" + str(gbm_r_m.get('mae', 0)) + ")",
                "Precision (%)": "N/A (MSE: " + str(gbm_r_m.get('mse', 0)) + ")",
                "Recall (%)": "N/A (RMSE: ₹" + str(gbm_r_m.get('rmse', 0)) + ")",
                "F1 Score": "N/A",
                "ROC-AUC / R²": f"R²: {gbm_r_m.get('r2_score', 0):.4f}"
            },
            {
                "Model Name": "LightGBM Regressor",
                "Category": "Gradient Boosting",
                "Task Type": "Salary Package (Regression)",
                "Accuracy (%)": "N/A (MAE: ₹" + str(lgb_r_m.get('mae', 0)) + ")",
                "Precision (%)": "N/A (MSE: " + str(lgb_r_m.get('mse', 0)) + ")",
                "Recall (%)": "N/A (RMSE: ₹" + str(lgb_r_m.get('rmse', 0)) + ")",
                "F1 Score": "N/A",
                "ROC-AUC / R²": f"R²: {lgb_r_m.get('r2_score', 0):.4f}"
            },
            {
                "Model Name": "XGBoost Regressor",
                "Category": "Gradient Boosting",
                "Task Type": "Salary Package (Regression)",
                "Accuracy (%)": "N/A (MAE: ₹" + str(xgb_r_m.get('mae', 0)) + ")",
                "Precision (%)": "N/A (MSE: " + str(xgb_r_m.get('mse', 0)) + ")",
                "Recall (%)": "N/A (RMSE: ₹" + str(xgb_r_m.get('rmse', 0)) + ")",
                "F1 Score": "N/A",
                "ROC-AUC / R²": f"R²: {xgb_r_m.get('r2_score', 0):.4f}"
            },
            {
                "Model Name": "Random Forest Regressor",
                "Category": "Bagging Ensemble",
                "Task Type": "Salary Package (Regression)",
                "Accuracy (%)": "N/A (MAE: ₹" + str(rf_r_m.get('mae', 0)) + ")",
                "Precision (%)": "N/A (MSE: " + str(rf_r_m.get('mse', 0)) + ")",
                "Recall (%)": "N/A (RMSE: ₹" + str(rf_r_m.get('rmse', 0)) + ")",
                "F1 Score": "N/A",
                "ROC-AUC / R²": f"R²: {rf_r_m.get('r2_score', 0):.4f}"
            },
            {
                "Model Name": "Simple CGPA Linear Regression",
                "Category": "Base Model",
                "Task Type": "Salary Package (Regression)",
                "Accuracy (%)": "N/A (MAE: ₹" + str(simple_lr_m.get('mae', 0)) + ")",
                "Precision (%)": "N/A (Corr r: " + str(simple_lr_m.get('correlation', 0)) + ")",
                "Recall (%)": "N/A (RMSE: ₹" + str(simple_lr_m.get('rmse', 0)) + ")",
                "F1 Score": "N/A",
                "ROC-AUC / R²": f"R²: {simple_lr_m.get('r2_score', 0):.4f}"
            }
        ]
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📊 Classification Accuracy Comparison across All Models")
        comp_cls_df = pd.DataFrame([
            {"Model": "Logistic Reg", "Accuracy (%)": log_m.get('accuracy', 0)*100},
            {"Model": "ID3 Tree", "Accuracy (%)": id3_c_m.get('accuracy', 0)*100},
            {"Model": "C4.5 Tree", "Accuracy (%)": c45_c_m.get('accuracy', 0)*100},
            {"Model": "CART (Karth)", "Accuracy (%)": cart_c_m.get('accuracy', 0)*100},
            {"Model": "GBM", "Accuracy (%)": gbm_c_m.get('accuracy', 0)*100},
            {"Model": "LightGBM", "Accuracy (%)": lgb_c_m.get('accuracy', 0)*100},
            {"Model": "XGBoost", "Accuracy (%)": xgb_c_m.get('accuracy', 0)*100},
            {"Model": "Random Forest", "Accuracy (%)": rf_c_m.get('accuracy', 0)*100},
            {"Model": "Bagging", "Accuracy (%)": bag_c_m.get('accuracy', 0)*100},
            {"Model": "AdaBoost", "Accuracy (%)": ada_c_m.get('accuracy', 0)*100}
        ])
        fig_comp_cls = px.bar(
            comp_cls_df,
            x="Model",
            y="Accuracy (%)",
            color="Model",
            text_auto=".2f",
            title="Classifier Accuracy Comparison (%)",
            color_discrete_sequence=["#1d4ed8", "#10b981", "#8b5cf6", "#059669", "#dc2626", "#d97706", "#7c3aed", "#0284c7", "#475569", "#e11d48"]
        )
        fig_comp_cls = apply_professional_layout(fig_comp_cls, "Classifier Accuracy Comparison (%)", height=420)
        st.plotly_chart(fig_comp_cls, use_container_width=True, key="comp_cls_bar_chart")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📐 Regression R² Score Comparison across All Models")
        comp_reg_df = pd.DataFrame([
            {"Model": "Linear Reg", "R² Score": lr_m.get('r2_score', 0)},
            {"Model": "ID3 Tree", "R² Score": id3_r_m.get('r2_score', 0)},
            {"Model": "C4.5 Tree", "R² Score": c45_r_m.get('r2_score', 0)},
            {"Model": "CART (Karth)", "R² Score": cart_r_m.get('r2_score', 0)},
            {"Model": "GBM", "R² Score": gbm_r_m.get('r2_score', 0)},
            {"Model": "LightGBM", "R² Score": lgb_r_m.get('r2_score', 0)},
            {"Model": "XGBoost", "R² Score": xgb_r_m.get('r2_score', 0)},
            {"Model": "Random Forest", "R² Score": rf_r_m.get('r2_score', 0)},
            {"Model": "Bagging", "R² Score": bag_r_m.get('r2_score', 0)},
            {"Model": "AdaBoost", "R² Score": ada_r_m.get('r2_score', 0)},
            {"Model": "Simple CGPA", "R² Score": simple_lr_m.get('r2_score', 0)}
        ])
        fig_comp_reg = px.bar(
            comp_reg_df,
            x="Model",
            y="R² Score",
            color="Model",
            text_auto=".4f",
            title="Regression R² Score Comparison",
            color_discrete_sequence=["#1d4ed8", "#10b981", "#8b5cf6", "#059669", "#dc2626", "#d97706", "#7c3aed", "#0284c7", "#475569", "#e11d48", "#64748b"]
        )
        fig_comp_reg = apply_professional_layout(fig_comp_reg, "Regression R² Score Comparison", height=420)
        st.plotly_chart(fig_comp_reg, use_container_width=True, key="comp_reg_bar_chart")

# ==================== TAB 5: LIVE PLACEMENT & SALARY PREDICTOR ====================
with tab5:
    st.subheader("🔮 Live Student Placement & Salary Package Predictor")
    st.markdown("Input student profile parameters below to estimate both **Placement Status Probability** and **Expected Salary Package** across all trained Base, GBM, LightGBM, XGBoost, and Ensemble models.")
    
    cat_opts = metadata.get("cat_options", {})
    
    with st.form("prediction_form"):
        col_p1, col_p2, col_p3 = st.columns(3)
        
        with col_p1:
            st.markdown("##### 🎓 Academic Background")
            cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.2, step=0.1)
            sgpa_1 = st.number_input("SGPA Sem 1", min_value=0.0, max_value=10.0, value=7.8, step=0.1)
            sgpa_2 = st.number_input("SGPA Sem 2", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
            sgpa_3 = st.number_input("SGPA Sem 3", min_value=0.0, max_value=10.0, value=8.1, step=0.1)
            sgpa_4 = st.number_input("SGPA Sem 4", min_value=0.0, max_value=10.0, value=8.2, step=0.1)
            sgpa_5 = st.number_input("SGPA Sem 5", min_value=0.0, max_value=10.0, value=8.3, step=0.1)
            sgpa_6 = st.number_input("SGPA Sem 6", min_value=0.0, max_value=10.0, value=8.4, step=0.1)
            sgpa_7 = st.number_input("SGPA Sem 7", min_value=0.0, max_value=10.0, value=8.5, step=0.1)
            sgpa_8 = st.number_input("SGPA Sem 8", min_value=0.0, max_value=10.0, value=8.6, step=0.1)
            
        with col_p2:
            st.markdown("##### 💻 Skills & Test Performance")
            coding_score = st.slider("Coding Test Score", 0.0, 100.0, 80.0)
            aptitude_score = st.slider("Aptitude Test Score", 0.0, 100.0, 78.0)
            soft_skills = st.slider("Soft Skills Rating", 0.0, 10.0, 8.0)
            mock_interview = st.slider("Mock Interview Score", 0.0, 100.0, 75.0)
            attendance = st.slider("Attendance Percentage (%)", 50.0, 100.0, 88.0)
            internships = st.number_input("Internships Completed", 0, 10, 2)
            projects = st.number_input("Projects Completed", 0, 10, 3)
            workshops = st.number_input("Workshops Attended", 0, 10, 2)
            certifications = st.number_input("Certifications", 0, 10, 2)
            publications = st.number_input("Publications", 0, 5, 0)
            extra_curricular = st.number_input("Extra Curricular Activities", 0, 10, 2)
            
        with col_p3:
            st.markdown("##### 🏛️ College & Profile Metadata")
            gender = st.selectbox("Gender", options=cat_opts.get("Gender", ["Male", "Female"]))
            city = st.selectbox("City", options=cat_opts.get("City", ["Mumbai", "Delhi", "Bangalore"]))
            college_tier = st.selectbox("College Tier", options=cat_opts.get("CollegeTier", ["Tier 1", "Tier 2", "Tier 3"]))
            stream = st.selectbox("Stream", options=cat_opts.get("Stream", ["Computer Science", "IT", "Electronics"]))
            specialisation = st.selectbox("Specialisation", options=cat_opts.get("Specialisation", ["Data Science", "Software Engineering"]))
            hostel = st.selectbox("Hostel Resident?", options=cat_opts.get("Hostel", ["Yes", "No"]))
            backlogs = st.selectbox("History of Backlogs", options=cat_opts.get("HistoryOfBacklogs", ["No", "Yes"]))
            cgpa_tier = st.selectbox("CGPA Tier", options=cat_opts.get("CGPA_Tier", ["High", "Medium", "Low"]))
            
        submit_btn = st.form_submit_button("🚀 Predict Placement & Salary Across All Models")

    if submit_btn:
        input_data = {
            "Gender": gender,
            "City": city,
            "CollegeTier": college_tier,
            "Stream": stream,
            "Specialisation": specialisation,
            "Hostel": hostel,
            "HistoryOfBacklogs": backlogs,
            "SGPA_Sem1": sgpa_1,
            "SGPA_Sem2": sgpa_2,
            "SGPA_Sem3": sgpa_3,
            "SGPA_Sem4": sgpa_4,
            "SGPA_Sem5": sgpa_5,
            "SGPA_Sem6": sgpa_6,
            "SGPA_Sem7": sgpa_7,
            "SGPA_Sem8": sgpa_8,
            "CGPA": cgpa,
            "AttendancePercent": attendance,
            "Internships": internships,
            "Projects": projects,
            "Workshops": float(workshops),
            "Certifications": certifications,
            "Publications": publications,
            "AptitudeTestScore": float(aptitude_score),
            "SoftSkillsRating": float(soft_skills),
            "CodingTestScore": float(coding_score),
            "MockInterviewScore": float(mock_interview),
            "ExtraCurricular": extra_curricular,
            "CGPA_Tier": cgpa_tier
        }
        
        input_df = pd.DataFrame([input_data])
        
        # 1. Base & Decision Tree Classifiers Inference
        log_prob = float(logistic_pipeline.predict_proba(input_df)[0][1]) if hasattr(logistic_pipeline, "predict_proba") else 0.0
        log_pred = int(logistic_pipeline.predict(input_df)[0])
        
        id3_c_pipe = ensemble_pipelines.get("id3_c")
        id3_prob = float(id3_c_pipe.predict_proba(input_df)[0][1]) if id3_c_pipe and hasattr(id3_c_pipe, "predict_proba") else 0.0
        id3_pred = int(id3_c_pipe.predict(input_df)[0]) if id3_c_pipe else 0

        c45_c_pipe = ensemble_pipelines.get("c45_c")
        c45_prob = float(c45_c_pipe.predict_proba(input_df)[0][1]) if c45_c_pipe and hasattr(c45_c_pipe, "predict_proba") else 0.0
        c45_pred = int(c45_c_pipe.predict(input_df)[0]) if c45_c_pipe else 0

        cart_c_pipe = ensemble_pipelines.get("cart_c")
        cart_prob = float(cart_c_pipe.predict_proba(input_df)[0][1]) if cart_c_pipe and hasattr(cart_c_pipe, "predict_proba") else 0.0
        cart_pred = int(cart_c_pipe.predict(input_df)[0]) if cart_c_pipe else 0

        # 2. Boosting & Ensemble Classifiers Inference (GBM, LightGBM, XGBoost, AdaBoost, Random Forest)
        gbm_c_pipe = ensemble_pipelines.get("gbm_c")
        gbm_prob = float(gbm_c_pipe.predict_proba(input_df)[0][1]) if gbm_c_pipe and hasattr(gbm_c_pipe, "predict_proba") else 0.0
        gbm_pred = int(gbm_c_pipe.predict(input_df)[0]) if gbm_c_pipe else 0

        lgb_c_pipe = ensemble_pipelines.get("lgb_c")
        lgb_prob = float(lgb_c_pipe.predict_proba(input_df)[0][1]) if lgb_c_pipe and hasattr(lgb_c_pipe, "predict_proba") else 0.0
        lgb_pred = int(lgb_c_pipe.predict(input_df)[0]) if lgb_c_pipe else 0

        xgb_c_pipe = ensemble_pipelines.get("xgb_c")
        xgb_prob = float(xgb_c_pipe.predict_proba(input_df)[0][1]) if xgb_c_pipe and hasattr(xgb_c_pipe, "predict_proba") else 0.0
        xgb_pred = int(xgb_c_pipe.predict(input_df)[0]) if xgb_c_pipe else 0

        ada_c_pipe = ensemble_pipelines.get("ada_c")
        ada_prob = float(ada_c_pipe.predict_proba(input_df)[0][1]) if ada_c_pipe and hasattr(ada_c_pipe, "predict_proba") else 0.0
        ada_pred = int(ada_c_pipe.predict(input_df)[0]) if ada_c_pipe else 0

        rf_c_pipe = ensemble_pipelines.get("rf_c")
        rf_prob = float(rf_c_pipe.predict_proba(input_df)[0][1]) if rf_c_pipe and hasattr(rf_c_pipe, "predict_proba") else 0.0
        rf_pred = int(rf_c_pipe.predict(input_df)[0]) if rf_c_pipe else 0

        # 3. Regressors Inference (Linear, ID3, C4.5, CART, GBM, LightGBM, XGBoost, Random Forest)
        salary_lr = max(0.0, float(lr_pipeline.predict(input_df)[0]))
        
        id3_r_pipe = ensemble_pipelines.get("id3_r")
        salary_id3 = max(0.0, float(id3_r_pipe.predict(input_df)[0])) if id3_r_pipe else 0.0

        c45_r_pipe = ensemble_pipelines.get("c45_r")
        salary_c45 = max(0.0, float(c45_r_pipe.predict(input_df)[0])) if c45_r_pipe else 0.0

        cart_r_pipe = ensemble_pipelines.get("cart_r")
        salary_cart = max(0.0, float(cart_r_pipe.predict(input_df)[0])) if cart_r_pipe else 0.0
        
        gbm_r_pipe = ensemble_pipelines.get("gbm_r")
        salary_gbm = max(0.0, float(gbm_r_pipe.predict(input_df)[0])) if gbm_r_pipe else 0.0

        lgb_r_pipe = ensemble_pipelines.get("lgb_r")
        salary_lgb = max(0.0, float(lgb_r_pipe.predict(input_df)[0])) if lgb_r_pipe else 0.0

        xgb_r_pipe = ensemble_pipelines.get("xgb_r")
        salary_xgb = max(0.0, float(xgb_r_pipe.predict(input_df)[0])) if xgb_r_pipe else 0.0

        rf_r_pipe = ensemble_pipelines.get("rf_r")
        salary_rf = max(0.0, float(rf_r_pipe.predict(input_df)[0])) if rf_r_pipe else 0.0

        cgpa_df = pd.DataFrame([{"CGPA": cgpa}])
        salary_simple = max(0.0, float(simple_lr.predict(cgpa_df)[0]))
        
        st.markdown("---")
        st.subheader("🎯 Model Prediction Results Comparison across All Models")
        
        st.markdown("#### 🟢 Placement Classification Outcomes")
        c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: {'#059669' if log_pred==1 else '#dc2626'}; font-size: 1.0rem;">{'PLACED ✅' if log_pred==1 else 'NOT PLACED ❌'}</div>
                <div class="metric-lbl">Logistic ({log_prob*100:.0f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: {'#059669' if id3_pred==1 else '#dc2626'}; font-size: 1.0rem;">{'PLACED ✅' if id3_pred==1 else 'NOT PLACED ❌'}</div>
                <div class="metric-lbl">ID3 ({id3_prob*100:.0f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: {'#059669' if c45_pred==1 else '#dc2626'}; font-size: 1.0rem;">{'PLACED ✅' if c45_pred==1 else 'NOT PLACED ❌'}</div>
                <div class="metric-lbl">C4.5 ({c45_prob*100:.0f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: {'#059669' if cart_pred==1 else '#dc2626'}; font-size: 1.0rem;">{'PLACED ✅' if cart_pred==1 else 'NOT PLACED ❌'}</div>
                <div class="metric-lbl">CART ({cart_prob*100:.0f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        with c5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: {'#059669' if gbm_pred==1 else '#dc2626'}; font-size: 1.0rem;">{'PLACED ✅' if gbm_pred==1 else 'NOT PLACED ❌'}</div>
                <div class="metric-lbl">GBM ({gbm_prob*100:.0f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        with c6:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: {'#059669' if lgb_pred==1 else '#dc2626'}; font-size: 1.0rem;">{'PLACED ✅' if lgb_pred==1 else 'NOT PLACED ❌'}</div>
                <div class="metric-lbl">LightGBM ({lgb_prob*100:.0f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        with c7:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: {'#059669' if xgb_pred==1 else '#dc2626'}; font-size: 1.0rem;">{'PLACED ✅' if xgb_pred==1 else 'NOT PLACED ❌'}</div>
                <div class="metric-lbl">XGBoost ({xgb_prob*100:.0f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        with c8:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: {'#059669' if rf_pred==1 else '#dc2626'}; font-size: 1.0rem;">{'PLACED ✅' if rf_pred==1 else 'NOT PLACED ❌'}</div>
                <div class="metric-lbl">RF ({rf_prob*100:.0f}%)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📐 Estimated Salary Packages (₹ LPA)")
        rc1, rc2, rc3, rc4, rc5, rc6, rc7, rc8 = st.columns(8)
        with rc1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #1d4ed8; font-size: 1.0rem;">₹{salary_lr:.2f}</div>
                <div class="metric-lbl">Linear Reg</div>
            </div>
            """, unsafe_allow_html=True)
        with rc2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #10b981; font-size: 1.0rem;">₹{salary_id3:.2f}</div>
                <div class="metric-lbl">ID3 Tree</div>
            </div>
            """, unsafe_allow_html=True)
        with rc3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #8b5cf6; font-size: 1.0rem;">₹{salary_c45:.2f}</div>
                <div class="metric-lbl">C4.5 Tree</div>
            </div>
            """, unsafe_allow_html=True)
        with rc4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #059669; font-size: 1.0rem;">₹{salary_cart:.2f}</div>
                <div class="metric-lbl">CART (Karth)</div>
            </div>
            """, unsafe_allow_html=True)
        with rc5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #dc2626; font-size: 1.0rem;">₹{salary_gbm:.2f}</div>
                <div class="metric-lbl">GBM</div>
            </div>
            """, unsafe_allow_html=True)
        with rc6:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #d97706; font-size: 1.0rem;">₹{salary_lgb:.2f}</div>
                <div class="metric-lbl">LightGBM</div>
            </div>
            """, unsafe_allow_html=True)
        with rc7:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #7c3aed; font-size: 1.0rem;">₹{salary_xgb:.2f}</div>
                <div class="metric-lbl">XGBoost</div>
            </div>
            """, unsafe_allow_html=True)
        with rc8:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #0284c7; font-size: 1.0rem;">₹{salary_rf:.2f}</div>
                <div class="metric-lbl">Random Forest</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Consensus Placement Gauge
        all_probs = [log_prob, id3_prob, c45_prob, cart_prob, gbm_prob, lgb_prob, xgb_prob, ada_prob, rf_prob]
        avg_prob = float(np.mean(all_probs))
        st.markdown(f"**Multi-Model Consensus Placement Confidence ({avg_prob * 100:.1f}%):**")
        st.progress(avg_prob)
        
        placed_votes = sum([log_pred, id3_pred, c45_pred, cart_pred, gbm_pred, lgb_pred, xgb_pred, ada_pred, rf_pred])
        if placed_votes >= 5:
            st.success(f"🎉 **High Placement Likelihood ({placed_votes}/9 Model Votes for Placement)!** Consensus Confidence: **{avg_prob*100:.1f}%** | Expected Salary Package (LightGBM): **₹{salary_lgb:.2f} LPA** (XGBoost: **₹{salary_xgb:.2f} LPA**).")
        else:
            st.warning(f"⚠️ **Low Placement Likelihood ({placed_votes}/9 Model Votes for Placement)** | Consensus Confidence: **{avg_prob*100:.1f}%**. Consider strengthening academic standing and interview skills.")


