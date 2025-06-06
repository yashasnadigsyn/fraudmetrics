import pandas as pd
import numpy as np
import altair as alt
from typing import Union, Any, List, Dict, Optional
from datetime import datetime
import json
import os

from thresholded import *
from threshold_free import *
from rank_based import *
from plot_curves import *

## Tooltips
METRIC_EXPLANATIONS = {
    "Accuracy": "Overall correctness of the model.",
    "Precision": "Of all positive predictions, how many were actually positive. (TP / (TP + FP))",
    "Recall": "Of all actual positives, how many did the model find. (TP / (TP + FN))",
    "Recall (Sensitivity)": "Of all actual positives, how many did the model find. (TP / (TP + FN))",
    "F1-Score": "The harmonic mean of Precision and Recall.",
    "Specificity": "Of all actual negatives, how many were correctly identified. (TN / (TN + FP))",
    "Balanced Error Rate (BER)": "Average of the error rates on each class.",
    "G-Mean": "Geometric mean of sensitivity and specificity. Good for imbalanced data.",
    "ROC-AUC": "Area Under the Receiver Operating Characteristic Curve. Measures ability to distinguish between classes.",
    "PR-AUC": "Area Under the Precision-Recall Curve. More informative for highly imbalanced datasets.",
    "PR-AUC (AP)": "Area Under the Precision-Recall Curve. More informative for highly imbalanced datasets.",
    "Precision@K": "Precision among the top K highest-scored transactions.",
    "Recall@K": "Recall among the top K highest-scored transactions.",
    "Card Precision@K": "The fraction of unique cards with at least one fraudulent transaction in the top K alerts."
}

def create_fallback_chart(title="Chart", width=400, height=300):
    """Create a simple fallback chart when main charts fail"""
    data = pd.DataFrame({'x': [1], 'y': [1], 'message': ['Chart data unavailable']})
    return alt.Chart(data).mark_text(
        align='center',
        baseline='middle',
        fontSize=16,
        color='gray'
    ).encode(
        x=alt.value(width/2),
        y=alt.value(height/2),
        text='message:N'
    ).properties(
        title=title,
        width=width,
        height=height
    )

def safe_chart_to_json(chart_func, *args, fallback_title="Chart", **kwargs):
    """Safely convert chart to JSON with fallback"""
    try:
        chart = chart_func(*args, **kwargs)
        if chart is None:
            raise ValueError("Chart function returned None")
        return json.dumps(chart.to_dict(format="vega"))
    except Exception as e:
        print(f"Warning: Chart generation failed ({fallback_title}): {e}")
        fallback_chart = create_fallback_chart(fallback_title)
        return json.dumps(fallback_chart.to_dict(format="vega"))

def generate_html_report(
    y_true: Union[list, np.ndarray],
    y_pred_proba: Union[list, np.ndarray],
    threshold: float = 0.5,
    pos_label: Any = 1,
    transaction_values: Optional[Union[list, np.ndarray]] = None,
    ids: Optional[Union[list, np.ndarray]] = None,
    k_values: List[int] = [10, 50, 100, 500, 1000],
    output_file: str = "fraud_performance_report.html",
    title: str = "Fraud Detection Model Performance Report"
) -> None:
    """Generate a beautiful HTML report without Jinja2 templates"""
    
    ## Calculate metrics
    try:
        thresholded_metrics = {
            "Accuracy": get_accuracy_score(y_true, y_pred_proba, threshold, pos_label),
            "Precision": get_precision_score(y_true, y_pred_proba, threshold, pos_label),
            "Recall (Sensitivity)": get_recall_score(y_true, y_pred_proba, threshold, pos_label),
            "F1-Score": get_f1_score(y_true, y_pred_proba, threshold, pos_label),
            "Specificity": get_specificity_score(y_true, y_pred_proba, threshold, pos_label),
            "BER": get_ber_score(y_true, y_pred_proba, threshold, pos_label),
            "G-Mean": get_gmean_score(y_true, y_pred_proba, threshold, pos_label)
        }
        
        threshold_free_metrics = {
            "ROC-AUC": get_roc_auc_score(y_true, y_pred_proba, pos_label),
            "PR-AUC (AP)": get_AP_score(y_true, y_pred_proba, pos_label)
        }
        
        rank_metrics = {
            "Precision@K": {k: get_precision_at_topk(y_true, y_pred_proba, k, pos_label) for k in k_values},
            "Recall@K": {k: get_recall_at_topk(y_true, y_pred_proba, k, pos_label) for k in k_values}
        }
        
        if ids is not None:
            rank_metrics["Card Precision@K"] = {k: get_card_precision_at_topk(y_true=y_true, y_pred_proba=y_pred_proba, ids=ids, k=k, pos_label=pos_label) for k in k_values}
    
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return
    
    ## Generate chart JSONs safely
    try:
        cm = get_binary_confusion_matrix(y_true, y_pred_proba, threshold, pos_label)
        confusion_json = safe_chart_to_json(plot_confusion_matrix, cm, ["Non-Fraud", "Fraud"], fallback_title="Confusion Matrix")
    except Exception as e:
        print(f"Warning: Could not generate confusion matrix: {e}")
        confusion_json = json.dumps(create_fallback_chart("Confusion Matrix").to_dict(format="vega"))
    
    roc_json = safe_chart_to_json(plot_roc_curve, y_true, y_pred_proba, pos_label, fallback_title="ROC Curve")
    pr_json = safe_chart_to_json(plot_pr_curve, y_true, y_pred_proba, pos_label, fallback_title="PR Curve")
    gains_json = safe_chart_to_json(plot_cumulative_gains, y_true, y_pred_proba, pos_label, fallback_title="Cumulative Gains")
    
    value_json = None
    if transaction_values is not None:
        value_json = safe_chart_to_json(plot_value_vs_alerts, y_true, y_pred_proba, transaction_values, pos_label, fallback_title="Value vs Alerts")
    
    ## Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #2d3748; line-height: 1.6; min-height: 100vh; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 2rem; }}
        .header {{ background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); border-radius: 20px; padding: 2rem; margin-bottom: 2rem; text-align: center; box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1); border: 1px solid rgba(255, 255, 255, 0.2); }}
        .header h1 {{ font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 0.5rem; }}
        .header .subtitle {{ color: #64748b; font-size: 1.1rem; margin-bottom: 1rem; }}
        .header .info {{ background: linear-gradient(135deg, #f8fafc, #e2e8f0); padding: 1rem; border-radius: 10px; display: inline-block; font-size: 0.9rem; color: #475569; }}
        .section {{ background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); border-radius: 20px; padding: 2rem; margin-bottom: 2rem; box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1); border: 1px solid rgba(255, 255, 255, 0.2); }}
        .section h2 {{ font-size: 1.8rem; font-weight: 600; color: #1e293b; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 0.5rem; }}
        .section h2::before {{ content: ""; width: 4px; height: 2rem; background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 2px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin-top: 1rem; }}
        .metric-card {{ background: linear-gradient(135deg, #f8fafc, #e2e8f0); border-radius: 15px; padding: 1.5rem; text-align: center; transition: all 0.3s ease; border: 1px solid rgba(255, 255, 255, 0.8); position: relative; overflow: hidden; }}
        .metric-card::before {{ content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(135deg, #667eea, #764ba2); }}
        .metric-card:hover {{ transform: translateY(-5px); box-shadow: 0 15px 30px rgba(0, 0, 0, 0.15); }}
        .metric-value {{ font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 0.5rem; }}
        .metric-label {{ font-size: 1rem; color: #64748b; font-weight: 500; }}
        .tooltip {{ cursor: help; position: relative; display: inline-block; }}
        .tooltip .tooltiptext {{ visibility: hidden; width: 250px; background: #1e293b; color: #fff; text-align: center; border-radius: 8px; padding: 10px; position: absolute; z-index: 1000; bottom: 125%; left: 50%; margin-left: -125px; opacity: 0; transition: opacity 0.3s; font-size: 0.85rem; box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2); }}
        .tooltip:hover .tooltiptext {{ visibility: visible; opacity: 1; }}
        .plots-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 2rem; margin-top: 1rem; }}
        .plot-card {{ background: rgba(255, 255, 255, 0.8); border-radius: 15px; padding: 1.5rem; box-shadow: 0 10px 20px rgba(0, 0, 0, 0.05); border: 1px solid rgba(255, 255, 255, 0.9); }}
        .plot-card h3 {{ font-size: 1.2rem; font-weight: 600; color: #1e293b; margin-bottom: 1rem; text-align: center; }}
        .plot-container {{ min-height: 400px; background: white; border-radius: 10px; padding: 1rem; }}
        .rank-table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05); }}
        .rank-table th {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 1rem; text-align: left; font-weight: 600; }}
        .rank-table td {{ padding: 1rem; border-bottom: 1px solid #e2e8f0; color: #475569; }}
        .rank-table tr:hover {{ background: #f8fafc; }}
        .rank-table th:first-child, .rank-table td:first-child {{ font-weight: 600; }}
        .rank-table th:not(:first-child), .rank-table td:not(:first-child) {{ text-align: right; }}
        .footer {{ text-align: center; margin-top: 2rem; padding: 1rem; color: rgba(255, 255, 255, 0.8); font-size: 0.9rem; }}
        @media (max-width: 768px) {{ .container {{ padding: 1rem; }} .header h1 {{ font-size: 2rem; }} .metrics-grid, .plots-grid {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <div class="subtitle">Comprehensive Performance Analysis</div>
            <div class="info">
                Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | 
                Threshold: {threshold:.3f}
            </div>
        </div>
        <div class="section">
            <h2>Thresholded Metrics</h2>
            <div class="metrics-grid">
"""
    for metric, value in thresholded_metrics.items():
        explanation = METRIC_EXPLANATIONS.get(metric, "")
        html_content += f"""
                <div class="metric-card">
                    <div class="metric-value">{value:.4f}</div>
                    <div class="metric-label tooltip">{metric}
                        <span class="tooltiptext">{explanation}</span>
                    </div>
                </div>"""
    html_content += """
            </div>
        </div>
        <div class="section">
            <h2>Threshold-Free Metrics</h2>
            <div class="metrics-grid">
"""
    for metric, value in threshold_free_metrics.items():
        explanation = METRIC_EXPLANATIONS.get(metric, "")
        html_content += f"""
                <div class="metric-card">
                    <div class="metric-value">{value:.4f}</div>
                    <div class="metric-label tooltip">{metric}
                        <span class="tooltiptext">{explanation}</span>
                    </div>
                </div>"""
    html_content += """
            </div>
        </div>
        <div class="section">
            <h2>Performance Visualizations</h2>
            <div class="plots-grid">
                <div class="plot-card"><h3>Confusion Matrix</h3><div id="confusion_plot" class="plot-container"></div></div>
                <div class="plot-card"><h3>ROC Curve</h3><div id="roc_plot" class="plot-container"></div></div>
                <div class="plot-card"><h3>Precision-Recall Curve</h3><div id="pr_plot" class="plot-container"></div></div>
                <div class="plot-card"><h3>Cumulative Gains</h3><div id="gains_plot" class="plot-container"></div></div>
"""
    if value_json:
        html_content += """
                <div class="plot-card"><h3>Value vs Alerts</h3><div id="value_plot" class="plot-container"></div></div>"""
    html_content += """
            </div>
        </div>
        <div class="section">
            <h2>Rank-Based Metrics</h2>
            <table class="rank-table">
                <thead><tr><th>Metric</th>
"""
    for k in k_values:
        html_content += f"<th>@ K={k}</th>"
    html_content += "</tr></thead><tbody>"
    for metric_name, values in rank_metrics.items():
        explanation = METRIC_EXPLANATIONS.get(metric_name, "")
        html_content += f"""
                    <tr><td class="tooltip">{metric_name}<span class="tooltiptext">{explanation}</span></td>"""
        for k in k_values:
            html_content += f"<td>{values[k]:.4f}</td>"
        html_content += "</tr>"
    html_content += """
                </tbody>
            </table>
        </div>
        <div class="footer">
            Fraud Detection Performance Report | Powered by Advanced Analytics
        </div>
    </div>
    <script>
        const embedOptions = { "actions": true, "theme": "default", "renderer": "svg" };
        vegaEmbed('#confusion_plot', """ + confusion_json + """, embedOptions).catch(err => console.error('Confusion matrix error:', err));
        vegaEmbed('#roc_plot', """ + roc_json + """, embedOptions).catch(err => console.error('ROC curve error:', err));
        vegaEmbed('#pr_plot', """ + pr_json + """, embedOptions).catch(err => console.error('PR curve error:', err));
        vegaEmbed('#gains_plot', """ + gains_json + """, embedOptions).catch(err => console.error('Gains curve error:', err));
"""
    if value_json:
        html_content += f"""
        vegaEmbed('#value_plot', {value_json}, embedOptions).catch(err => console.error('Value plot error:', err));
"""
    html_content += """
    </script>
</body>
</html>
"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML report saved to {os.path.abspath(output_file)}")

def compare_models_report(
    y_true: Union[list, np.ndarray],
    models: Dict[str, Union[list, np.ndarray]],
    threshold: float = 0.5,
    pos_label: Any = 1,
    output_file: str = "model_comparison_report.html",
    title: str = "Fraud Models Comparison Report"
) -> None:
    """Generate a beautiful model comparison report"""
    
    comparison_data = []
    for model_name, y_pred_proba in models.items():
        try:
            metrics = {
                "Model": model_name,
                "ROC-AUC": get_roc_auc_score(y_true, y_pred_proba, pos_label),
                "PR-AUC": get_AP_score(y_true, y_pred_proba, pos_label),
                "Precision": get_precision_score(y_true, y_pred_proba, threshold, pos_label),
                "Recall": get_recall_score(y_true, y_pred_proba, threshold, pos_label),
                "F1-Score": get_f1_score(y_true, y_pred_proba, threshold, pos_label),
            }
            comparison_data.append(metrics)
        except Exception as e:
            print(f"Warning: Could not calculate metrics for {model_name}: {e}")
    
    if not comparison_data:
        print("Error: No valid model data to compare. Report not generated.")
        return
    
    df_comparison = pd.DataFrame(comparison_data).sort_values("ROC-AUC", ascending=False)
    
    table_html = '<table class="comparison-table"><thead><tr>'
    for key in df_comparison.columns:
        table_html += f'<th>{key}</th>'
    table_html += '</tr></thead><tbody>'
    
    for _, row in df_comparison.iterrows():
        table_html += '<tr>'
        for key, value in row.items():
            if key == "Model":
                table_html += f'<td class="model-name">{value}</td>'
            else:
                table_html += f'<td>{value:.4f}</td>'
        table_html += '</tr>'
    table_html += '</tbody></table>'
    
    roc_data, pr_data = [], []
    for model_name, y_pred_proba in models.items():
        try:
            fpr, tpr, _ = get_roc_curve_points(y_true, y_pred_proba, pos_label=pos_label)
            roc_data.append(pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr, 'Model': model_name}))
            precision, recall, _ = get_pr_curve_points(y_true, y_pred_proba, pos_label=pos_label)
            pr_data.append(pd.DataFrame({'Recall': recall, 'Precision': precision, 'Model': model_name}))
        except Exception as e:
            print(f"Warning: Could not generate curve data for {model_name}: {e}")
            
    roc_json, pr_json = [json.dumps(create_fallback_chart(t).to_dict(format="vega")) for t in ["ROC Comparison", "PR Comparison"]]
    
    try:
        if roc_data:
            roc_plot = alt.Chart(pd.concat(roc_data)).mark_line(interpolate='linear', strokeWidth=3).encode(
                x=alt.X('False Positive Rate:Q', title='False Positive Rate'),
                y=alt.Y('True Positive Rate:Q', title='True Positive Rate'),
                color=alt.Color('Model:N', scale=alt.Scale(scheme='category10'), legend=alt.Legend(title="Model")),
                tooltip=['Model', 'False Positive Rate', 'True Positive Rate']
            ).interactive()
            roc_json = json.dumps(roc_plot.to_dict(format="vega"))
        if pr_data:
            pr_plot = alt.Chart(pd.concat(pr_data)).mark_line(interpolate='linear', strokeWidth=3).encode(
                x=alt.X('Recall:Q', scale=alt.Scale(domain=(0, 1))),
                y=alt.Y('Precision:Q', scale=alt.Scale(domain=(0, 1))),
                color=alt.Color('Model:N', scale=alt.Scale(scheme='category10'), legend=alt.Legend(title="Model")),
                tooltip=['Model', 'Recall', 'Precision']
            ).interactive()
            pr_json = json.dumps(pr_plot.to_dict(format="vega"))
    except Exception as e:
        print(f"Error creating comparison charts: {e}")

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #2d3748; line-height: 1.6; min-height: 100vh; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 2rem; }}
        .header {{ background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); border-radius: 20px; padding: 2rem; margin-bottom: 2rem; text-align: center; box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1); border: 1px solid rgba(255, 255, 255, 0.2); }}
        .header h1 {{ font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 0.5rem; }}
        .header .subtitle {{ color: #64748b; font-size: 1.1rem; margin-bottom: 1rem; }}
        .header .info {{ background: linear-gradient(135deg, #f8fafc, #e2e8f0); padding: 1rem; border-radius: 10px; display: inline-block; font-size: 0.9rem; color: #475569; }}
        .section {{ background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); border-radius: 20px; padding: 2rem; margin-bottom: 2rem; box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1); border: 1px solid rgba(255, 255, 255, 0.2); }}
        .section h2 {{ font-size: 1.8rem; font-weight: 600; color: #1e293b; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 0.5rem; }}
        .section h2::before {{ content: ""; width: 4px; height: 2rem; background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 2px; }}
        .comparison-table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05); }}
        .comparison-table th {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 1rem; text-align: left; font-weight: 600; }}
        .comparison-table td {{ padding: 1rem; border-bottom: 1px solid #e2e8f0; color: #475569; }}
        .comparison-table tr:hover {{ background: #f8fafc; }}
        .model-name {{ font-weight: 600; color: #1e293b; }}
        .comparison-table th:not(:first-child), .comparison-table td:not(:first-child) {{ text-align: right; }}
        .plots-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 2rem; margin-top: 1rem; }}
        .plot-card {{ background: rgba(255, 255, 255, 0.8); border-radius: 15px; padding: 1.5rem; box-shadow: 0 10px 20px rgba(0, 0, 0, 0.05); border: 1px solid rgba(255, 255, 255, 0.9); }}
        .plot-card h3 {{ font-size: 1.2rem; font-weight: 600; color: #1e293b; margin-bottom: 1rem; text-align: center; }}
        .plot-container {{ min-height: 400px; background: white; border-radius: 10px; padding: 1rem; }}
        .footer {{ text-align: center; margin-top: 2rem; padding: 1rem; color: rgba(255, 255, 255, 0.8); font-size: 0.9rem; }}
        @media (max-width: 768px) {{ .container {{ padding: 1rem; }} .header h1 {{ font-size: 2rem; }} .plots-grid {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <div class="subtitle">Side-by-Side Model Performance Analysis</div>
            <div class="info">
                Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | 
                Threshold: {threshold:.3f}
            </div>
        </div>
        <div class="section">
            <h2>Metrics Comparison</h2>
            {table_html}
        </div>
        <div class="section">
            <h2>Performance Curves</h2>
            <div class="plots-grid">
                <div class="plot-card">
                    <h3>Comparative ROC Curves</h3>
                    <div id="roc_plot_container" class="plot-container"></div>
                </div>
                <div class="plot-card">
                    <h3>Comparative Precision-Recall Curves</h3>
                    <div id="pr_plot_container" class="plot-container"></div>
                </div>
            </div>
        </div>
        <div class="footer">
            Model Comparison Report | Powered by Advanced Analytics
        </div>
    </div>
    <script>
        const embedOptions = {{ "actions": "true", "theme": "default", "renderer": "svg" }};
        vegaEmbed('#roc_plot_container', {roc_json}, embedOptions).catch(err => console.error('ROC comparison error:', err));
        vegaEmbed('#pr_plot_container', {pr_json}, embedOptions).catch(err => console.error('PR comparison error:', err));
    </script>
</body>
</html>
"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Model comparison report saved to {os.path.abspath(output_file)}")