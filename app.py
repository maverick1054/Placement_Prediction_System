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
    plot_correlation_heatmap, apply_professional_layout,
    COLOR_PRIMARY, COLOR_PLACED, COLOR_NOT_PLACED
)
from model_trainer import load_trained_pipelines

# Page Configuration
st.set_page_config(
    page_title="Placement Analytics & ML Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Professional Light Theme CSS
st.markdown("""
<style>
    /* Main App Light Background & Font */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.025em;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }
    
    /* Professional Card Containers */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 15px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    
    .metric-val {
        font-size: 1.9rem;
        font-weight: 800;
        color: #1e3a8a;
        line-height: 1.2;
    }
    
    .metric-lbl {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    
    /* High Contrast Prediction Cards */
    .prediction-card-high {
        background: #ecfdf5;
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    
    .prediction-card-medium {
        background: #fffbeb;
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    
    .prediction-card-low {
        background: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f5f9;
        padding: 6px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        white-space: pre-wrap;
        border-radius: 8px;
        font-weight: 600;
        color: #475569;
    }

    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #1d4ed8 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: #ffffff;
        font-weight: 700;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_cached_df():
    return load_data()

# Load Data & Pipelines
df = get_cached_df()
clf_pipeline, reg_pipeline, metadata = load_trained_pipelines()

# Sidebar Filters
st.sidebar.title("🎛️ Dynamic Filters")
st.sidebar.markdown("Filter dataset metrics and charts across all tabs:")

selected_stream = st.sidebar.multiselect(
    "Filter by Stream:",
    options=sorted(df["Stream"].dropna().unique().tolist()),
    default=df["Stream"].dropna().unique().tolist()
)

selected_tier = st.sidebar.multiselect(
    "Filter by College Tier:",
    options=sorted(df["CollegeTier"].dropna().unique().tolist()),
    default=df["CollegeTier"].dropna().unique().tolist()
)

selected_placement = st.sidebar.selectbox(
    "Filter Placement Outcome:",
    options=["All Students", "Placed Only (1)", "Not Placed Only (0)"]
)

# Apply Filters
filtered_df = df.copy()
if selected_stream:
    filtered_df = filtered_df[filtered_df["Stream"].isin(selected_stream)]
if selected_tier:
    filtered_df = filtered_df[filtered_df["CollegeTier"].isin(selected_tier)]
if selected_placement == "Placed Only (1)":
    filtered_df = filtered_df[filtered_df["PlacementStatus"] == 1]
elif selected_placement == "Not Placed Only (0)":
    filtered_df = filtered_df[filtered_df["PlacementStatus"] == 0]

# Dashboard Header
st.markdown('<div class="main-title">🎓 Student Placement Analytics & ML Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">High-Precision Exploratory Data Analysis, Null Profiling, and Real-Time Placement Calculator</div>', unsafe_allow_html=True)

# Tabs Navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset Overview",
    "🧹 Data Health & Nulls",
    "📈 EDA & Graphs",
    "🤖 ML Models & Evaluation",
    "🔮 Live Predictor"
])

# ==================== TAB 1: DATASET OVERVIEW ====================
with tab1:
    st.subheader("📌 Key Dataset Metrics")
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
            <div class="metric-lbl">Avg Placed Salary</div>
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
        st.markdown("### 📋 Interactive Dataset Explorer")
        st.dataframe(filtered_df.head(100), height=420, use_container_width=True)
        
    with col_right:
        st.markdown("### 🔍 Column Metadata & Data Types")
        st.dataframe(meta_info["columns_summary"], height=420, use_container_width=True)
        
    st.markdown("### 📈 Numerical Features Statistical Summary")
    st.dataframe(filtered_df.describe().T.style.format("{:.2f}"), use_container_width=True)

# ==================== TAB 2: MISSING VALUES & DATA HEALTH ====================
with tab2:
    st.subheader("🧹 Missing Values Analysis (Data Health)")
    
    missing_summary = get_missing_values_summary(df)
    
    if not missing_summary.empty:
        col_m1, col_m2 = st.columns([1, 1.2])
        
        with col_m1:
            st.markdown("#### Null Value Summary Table")
            st.dataframe(missing_summary, use_container_width=True)
            
            st.info("""
            **Missing Value Handling Strategy**:
            - Missing values are concentrated in test scores and soft skill ratings (`Workshops`, `MockInterviewScore`, `AptitudeTestScore`, `SoftSkillsRating`, `CodingTestScore`).
            - The ML preprocessing pipeline uses **SimpleImputer with Median strategy** for numerical variables to handle skewed test score distributions.
            """)
            
        with col_m2:
            st.markdown("#### Null Records Breakdown")
            fig_missing = plot_missing_values_chart(missing_summary)
            st.plotly_chart(fig_missing, use_container_width=True)
    else:
        st.success("🎉 Fantastic news! No missing values found in the dataset.")

# ==================== TAB 3: EDA & VISUALIZATIONS ====================
with tab3:
    st.subheader("📈 Exploratory Data Analysis & Visualizations")
    
    eda_option = st.selectbox(
        "Select EDA Analysis View:",
        [
            "1. Univariate Feature Distributions",
            "2. Categorical Features vs Placement Outcome",
            "3. Academic & Test Scores vs Placement Status",
            "4. Numerical Features Correlation Matrix",
            "5. Anomaly Detection Profiling"
        ]
    )
    
    st.markdown("---")
    
    if eda_option.startswith("1"):
        st.markdown("### Univariate Feature Distributions")
        num_cat_cols = [c for c in filtered_df.columns if c not in ["StudentID"]]
        selected_col = st.selectbox("Select Feature to Plot:", options=num_cat_cols, index=num_cat_cols.index("CGPA"))
        fig_dist = plot_univariate_distribution(filtered_df, selected_col)
        st.plotly_chart(fig_dist, use_container_width=True)
        
    elif eda_option.startswith("2"):
        st.markdown("### Categorical Features vs Placement Status")
        cat_options = ["Gender", "Stream", "Specialisation", "CollegeTier", "HistoryOfBacklogs", "Hostel"]
        selected_cat = st.selectbox("Select Categorical Feature:", options=cat_options, index=0)
        fig_cat = plot_placement_by_category(filtered_df, selected_cat)
        st.plotly_chart(fig_cat, use_container_width=True)
        
    elif eda_option.startswith("3"):
        st.markdown("### Feature Performance vs Placement Status")
        box_options = ["CGPA", "CodingTestScore", "AptitudeTestScore", "SoftSkillsRating", "MockInterviewScore", "AttendancePercent", "Internships", "Salary Package"]
        selected_box = st.selectbox("Select Numerical Feature for Box Plot:", options=box_options, index=0)
        fig_box = plot_feature_vs_placement_box(filtered_df, selected_box)
        st.plotly_chart(fig_box, use_container_width=True)
        
        st.markdown("#### Scatter Plot: Aptitude Score vs Coding Test Score")
        fig_scat = px.scatter(
            filtered_df,
            x="AptitudeTestScore",
            y="CodingTestScore",
            color=filtered_df["PlacementStatus"].map({0: "Not Placed", 1: "Placed"}),
            hover_data=["CGPA", "Salary Package"],
            title="Coding Test Score vs Aptitude Test Score (Colored by Placement Outcome)",
            color_discrete_map={"Not Placed": COLOR_NOT_PLACED, "Placed": COLOR_PLACED},
            opacity=0.8
        )
        fig_scat = apply_professional_layout(fig_scat, "Coding Test Score vs Aptitude Test Score", height=500)
        st.plotly_chart(fig_scat, use_container_width=True)
        
    elif eda_option.startswith("4"):
        st.markdown("### Features Correlation Matrix")
        fig_corr = plot_correlation_heatmap(filtered_df)
        st.plotly_chart(fig_corr, use_container_width=True)
        
    elif eda_option.startswith("5"):
        st.markdown("### Anomaly Detection Profiling")
        anomaly_counts = filtered_df["IsAnomaly"].value_counts().reset_index()
        anomaly_counts.columns = ["IsAnomaly", "Count"]
        anomaly_counts["Status"] = anomaly_counts["IsAnomaly"].map({0: "Normal Record", 1: "Anomaly Record"})
        
        fig_anom = px.pie(
            anomaly_counts,
            values="Count",
            names="Status",
            title="Anomaly Records Ratio in Dataset",
            color_discrete_sequence=["#1d4ed8", "#dc2626"],
            hole=0.4
        )
        fig_anom = apply_professional_layout(fig_anom, "Anomaly Records Ratio in Dataset", height=450)
        st.plotly_chart(fig_anom, use_container_width=True)

# ==================== TAB 4: ML MODELS & EVALUATION ====================
with tab4:
    st.subheader("🤖 Machine Learning Model Performance & Metrics")
    
    clf_m = metadata["clf_metrics"]
    reg_m = metadata["reg_metrics"]
    
    st.markdown("### 🎯 1. Placement Classification Model (Random Forest)")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #059669;">{clf_m['accuracy'] * 100:.2f}%</div>
            <div class="metric-lbl">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #1d4ed8;">{clf_m['precision'] * 100:.2f}%</div>
            <div class="metric-lbl">Precision</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #7c3aed;">{clf_m['recall'] * 100:.2f}%</div>
            <div class="metric-lbl">Recall</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #0284c7;">{clf_m['f1_score'] * 100:.2f}%</div>
            <div class="metric-lbl">F1 Score</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #d97706;">{clf_m['roc_auc']:.4f}</div>
            <div class="metric-lbl">ROC-AUC</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    col_cm, col_imp = st.columns([1, 1.2])
    
    with col_cm:
        st.markdown("#### Confusion Matrix")
        cm = np.array(clf_m["confusion_matrix"])
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=["Predicted Not Placed", "Predicted Placed"],
            y=["Actual Not Placed", "Actual Placed"],
            color_continuous_scale="Blues",
            title="Confusion Matrix"
        )
        fig_cm.update_traces(textfont=dict(color="#0f172a", size=14))
        fig_cm = apply_professional_layout(fig_cm, "Confusion Matrix", height=380)
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with col_imp:
        st.markdown("#### Top 15 Key Feature Importances")
        imp_df = pd.DataFrame(metadata["feature_importances"])
        fig_imp = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Key Drivers of Placement Success",
            color="Importance",
            color_continuous_scale="Blues"
        )
        fig_imp = apply_professional_layout(fig_imp, "Key Drivers of Placement Success", height=380)
        fig_imp.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_imp, use_container_width=True)
        
    st.markdown("---")
    st.markdown("### 💵 2. Salary Regressor Model (Random Forest)")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #1d4ed8;">{reg_m['r2_score']:.4f}</div>
            <div class="metric-lbl">R² Score</div>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #059669;">₹{reg_m['mae']} LPA</div>
            <div class="metric-lbl">Mean Absolute Error</div>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #475569;">{reg_m['mse']}</div>
            <div class="metric-lbl">Mean Squared Error</div>
        </div>
        """, unsafe_allow_html=True)
    with r4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #d97706;">₹{reg_m['rmse']} LPA</div>
            <div class="metric-lbl">Root Mean Squared Error</div>
        </div>
        """, unsafe_allow_html=True)

# ==================== TAB 5: LIVE PLACEMENT PREDICTOR ====================
with tab5:
    st.subheader("🔮 Live Student Placement & Salary Predictor")
    st.markdown("Fill out the student details below to calculate placement probability and estimated salary package in real time.")
    
    cat_opts = metadata["cat_options"]
    
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
            
        submit_btn = st.form_submit_button("🚀 Calculate Placement Probability")

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
        
        # ML Inference
        placement_prob = clf_pipeline.predict_proba(input_df)[0][1] * 100
        predicted_status = clf_pipeline.predict(input_df)[0]
        predicted_salary = reg_pipeline.predict(input_df)[0] if predicted_status == 1 or placement_prob > 50 else 0.0
        
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        res1, res2, res3 = st.columns(3)
        
        with res1:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid #1d4ed8;">
                <div class="metric-val" style="color: #1d4ed8;">{placement_prob:.1f}%</div>
                <div class="metric-lbl">Placement Probability</div>
            </div>
            """, unsafe_allow_html=True)
            
        with res2:
            if placement_prob >= 70:
                card_style = "border-top: 4px solid #059669; background: #f0fdf4;"
                val_color = "#047857"
                status_text = "✅ High Chance of Placement"
            elif placement_prob >= 40:
                card_style = "border-top: 4px solid #d97706; background: #fffbeb;"
                val_color = "#b45309"
                status_text = "⚠️ Moderate Chance"
            else:
                card_style = "border-top: 4px solid #dc2626; background: #fef2f2;"
                val_color = "#b91c1c"
                status_text = "❌ Low Chance of Placement"
                
            st.markdown(f"""
            <div class="metric-card" style="{card_style}">
                <div class="metric-val" style="color: {val_color}; font-size: 1.4rem;">{status_text}</div>
                <div class="metric-lbl">Placement Prospect</div>
            </div>
            """, unsafe_allow_html=True)
            
        with res3:
            salary_str = f"₹{predicted_salary:.2f} LPA" if predicted_salary > 0 else "N/A"
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid #059669;">
                <div class="metric-val" style="color: #059669;">{salary_str}</div>
                <div class="metric-lbl">Estimated Salary Package</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Probability Gauge
        fig_gauge = px_go.Figure(px_go.Indicator(
            mode="gauge+number",
            value=placement_prob,
            number={'suffix': "%", 'font': {'color': "#0f172a", 'size': 36}},
            title={'text': "Placement Likelihood Index", 'font': {'color': "#0f172a", 'size': 16}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': "#0f172a"},
                'bar': {'color': "#1d4ed8"},
                'bgcolor': "#ffffff",
                'borderwidth': 1,
                'bordercolor': "#cbd5e1",
                'steps': [
                    {'range': [0, 40], 'color': "#fee2e2"},
                    {'range': [40, 70], 'color': "#fef3c7"},
                    {'range': [70, 100], 'color': "#d1fae5"}
                ]
            }
        ))
        fig_gauge = apply_professional_layout(fig_gauge, height=350)
        st.plotly_chart(fig_gauge, use_container_width=True)
