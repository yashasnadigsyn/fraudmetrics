## NOTE: Examples here are generated using Ai

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import time

# Import the reporting generator
from generator import generate_html_report, compare_models_report

# Import metrics for terminal output
from threshold_free import get_roc_auc_score
from thresholded import get_f1_score

import altair as alt
alt.data_transformers.enable("vegafusion")


def run():
    """
    Main function to load data, train models, and generate reports.
    """
    # --- 1. Load and Prepare Data ---
    print("🔄 Loading and preparing data...")
    try:
        # Using a smaller sample for speed, remove .sample() for full dataset
        df = pd.read_csv('output.csv', parse_dates=['TX_DATETIME'])
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print("❌ Error: 'output.csv' not found. Please place the dataset in the same directory.")
        return

    # Define features (X) and target (y)
    # Exclude identifiers, datetime, and leakage variables
    features = [
        'TX_AMOUNT', 'TX_DURING_WEEKEND', 'TX_DURING_NIGHT',
        'CUSTOMER_ID_NB_TX_1DAY_WINDOW', 'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW',
        'CUSTOMER_ID_NB_TX_7DAY_WINDOW', 'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW',
        'CUSTOMER_ID_NB_TX_30DAY_WINDOW', 'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW',
        'TERMINAL_ID_NB_TX_1DAY_WINDOW', 'TERMINAL_ID_RISK_1DAY_WINDOW',
        'TERMINAL_ID_NB_TX_7DAY_WINDOW', 'TERMINAL_ID_RISK_7DAY_WINDOW',
        'TERMINAL_ID_NB_TX_30DAY_WINDOW', 'TERMINAL_ID_RISK_30DAY_WINDOW'
    ]
    target = 'TX_FRAUD'

    # Split data while preserving all columns for reporting
    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        stratify=df[target]  # Important for imbalanced datasets
    )

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    print(f"Fraud cases in test set: {y_test.sum()} ({y_test.mean()*100:.2f}%)")

    # --- 2. Define and Train Models ---
    print("\n🚀 Training models...")
    models = {
        "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42),
        "Decision Tree (Depth 2)": DecisionTreeClassifier(max_depth=2, random_state=42),
        "Decision Tree (Unlimited)": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
    }

    model_predictions = {}
    
    for name, model in models.items():
        start_time = time.time()
        print(f"  - Training {name}...")
        model.fit(X_train, y_train)
        # Get probabilities for the positive class (fraud)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        model_predictions[name] = y_pred_proba
        duration = time.time() - start_time
        print(f"    Done in {duration:.2f}s.")

    # --- 3. Terminal Report ---
    print("\n📊--- Terminal Performance Summary ---📊")
    print(f"{'Model':<30} | {'ROC-AUC':<10} | {'F1-Score (t=0.5)':<18}")
    print("-" * 65)
    for name, y_pred_proba in model_predictions.items():
        roc_auc = get_roc_auc_score(list(y_test), list(y_pred_proba))
        f1 = get_f1_score(list(y_test), list(y_pred_proba), threshold=0.5)
        print(f"{name:<30} | {roc_auc:<10.4f} | {f1:<18.4f}")
    print("-" * 65)


    # --- 4. Generate HTML Reports ---
    print("\n📄 Generating HTML reports...")
    
    # Generate single-model report for the best model (usually XGBoost)
    xgb_preds = model_predictions.get("XGBoost")
    if xgb_preds is not None:
        generate_html_report(
            y_true=list(y_test),
            y_pred_proba=list(xgb_preds),
            threshold=0.5,
            transaction_values=list(test_df['TX_AMOUNT']),
            ids=list(test_df['CUSTOMER_ID']),
            output_file="xgboost_performance_report.html",
            title="XGBoost Fraud Model Performance Report"
        )

    # Generate model comparison report
    compare_models_report(
        y_true=list(y_test),
        models=model_predictions,
        threshold=0.5,
        output_file="model_comparison_report.html",
        title="Fraud Models Comparison Report"
    )

if __name__ == "__main__":
    run()