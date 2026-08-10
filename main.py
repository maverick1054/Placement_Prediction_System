"""
Placement Prediction & EDA Dashboard Entrypoint Script
Run this script to train/update models and launch the Streamlit Dashboard.
"""

import sys
import subprocess
from model_trainer import train_and_evaluate_models

def main():
    print("=" * 60)
    print("🎓 PLACEMENT PREDICTION & EDA DASHBOARD")
    print("=" * 60)
    
    print("\n[1/2] Training & Evaluating Placement & Salary ML Models...")
    metadata = train_and_evaluate_models()
    
    print(f"✓ Placement Classifier Accuracy: {metadata['clf_metrics']['accuracy']*100:.2f}%")
    print(f"✓ Salary Regressor R² Score: {metadata['reg_metrics']['r2_score']:.4f}")
    
    print("\n[2/2] Launching Streamlit Interactive Dashboard...")
    print("Opening browser at http://localhost:8501 ...")
    
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()