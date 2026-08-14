from flask import Flask, render_template_string
import os
import pickle
import json

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_pickle(name):
    path = os.path.join(BASE_DIR, name)
    with open(path, "rb") as f:
        return pickle.load(f)

feature_info = load_pickle("feature_info.pkl")
model_metrics = load_pickle("model_metrics.pkl")

rf = model_metrics.get("rf", {})
ann = model_metrics.get("ann", {})

def metric_percent(model, key):
    value = model.get(key, 0)
    try:
        return round(float(value) * 100, 2)
    except Exception:
        return 0

chart_payload = {
    "labels": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
    "rf": [
        metric_percent(rf, "accuracy"),
        metric_percent(rf, "precision"),
        metric_percent(rf, "recall"),
        metric_percent(rf, "f1"),
        metric_percent(rf, "roc_auc"),
    ],
    "ann": [
        metric_percent(ann, "accuracy"),
        metric_percent(ann, "precision"),
        metric_percent(ann, "recall"),
        metric_percent(ann, "f1"),
        metric_percent(ann, "roc_auc"),
    ],
    "feature_counts": {
        "numerical": len(feature_info.get("numerical_features", [])),
        "categorical": len(feature_info.get("categorical_features", [])),
    },
}

HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Insurance Claim Prediction</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
*{box-sizing:border-box}
body{margin:0;font-family:Inter,Arial,sans-serif;background:#eef4fb;color:#1d2738}
.layout{display:flex;min-height:100vh}
.sidebar{width:245px;background:linear-gradient(160deg,#111827,#1f3b54);color:white;padding:28px 18px;position:fixed;top:0;bottom:0}
.sidebar h2{font-size:22px;margin:0 0 28px;text-align:center}
.sidebar a{display:block;color:#dce7f5;text-decoration:none;padding:12px 14px;border-radius:9px;margin:6px 0}
.sidebar a:hover,.sidebar a.active{background:rgba(255,255,255,.12)}
.main{margin-left:245px;width:calc(100% - 245px);padding:38px}
.hero{background:linear-gradient(135deg,#ffffff,#f6faff);border:1px solid #dce6f1;border-radius:20px;padding:28px;box-shadow:0 12px 35px rgba(26,54,93,.08)}
.hero h1{margin:0 0 10px;font-size:38px}.muted{color:#6a7890}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:18px;margin-top:22px}
.card{background:white;border:1px solid #dfe7f1;border-radius:16px;padding:22px;box-shadow:0 8px 25px rgba(32,58,94,.06)}
.card h3{margin-top:0}
.metric-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-top:14px}
.metric{background:#f4f8fc;padding:13px;border-radius:10px;text-align:center}.metric b{display:block;font-size:18px;margin-top:5px}
.badge{display:inline-block;padding:6px 10px;border-radius:999px;background:#e8eefc;color:#3154a5;margin:3px;font-size:13px}
.notice{background:#fff6df;border:1px solid #f3d990;border-radius:12px;padding:14px;margin-top:18px}
table{width:100%;border-collapse:collapse}th,td{padding:11px;border-bottom:1px solid #e8edf3;text-align:left}
.chart-wrap{position:relative;height:360px}
.chart-grid{display:grid;grid-template-columns:2fr 1fr;gap:18px;margin-top:22px}
.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:22px}
.kpi{background:white;border:1px solid #dfe7f1;border-radius:16px;padding:20px;box-shadow:0 8px 25px rgba(32,58,94,.06)}
.kpi .num{font-size:32px;font-weight:800;margin-top:6px}
@media(max-width:1000px){.chart-grid{grid-template-columns:1fr}.kpi-row{grid-template-columns:1fr 1fr}}
@media(max-width:900px){.sidebar{position:static;width:100%;height:auto}.layout{display:block}.main{margin:0;width:100%;padding:20px}.grid,.metric-grid{grid-template-columns:1fr 1fr}}
</style>
</head>
<body>
<div class="layout">
<aside class="sidebar">
<h2>Insurance Claim Prediction</h2>
<a class="{{'active' if page=='overview' else ''}}" href="/">Overview</a>
<a class="{{'active' if page=='features' else ''}}" href="/features">Feature Information</a>
<a class="{{'active' if page=='metrics' else ''}}" href="/metrics">Model Metrics</a>
<a class="{{'active' if page=='about' else ''}}" href="/about">About Project</a>
</aside>

<main class="main">

{% if page == 'overview' %}
<div class="hero">
<h1>Insurance Claim Prediction</h1>
<p class="muted">An AI semester project based on the Porto Seguro Safe Driver Prediction problem, covering preprocessing, model training, evaluation, and dashboard-based presentation.</p>
<div class="notice"><strong>Portfolio demo mode:</strong> the original project references files that are no longer available in the project folder, including the Random Forest model, scaler, and training dataset. To avoid presenting a misleading prediction, this live deployment focuses on verified saved metadata and recorded evaluation results.</div>
</div>

<div class="kpi-row">
<div class="kpi"><div class="muted">Original Features</div><div class="num">{{ feature_info.original_features|length }}</div></div>
<div class="kpi"><div class="muted">Numerical Features</div><div class="num">{{ feature_info.numerical_features|length }}</div></div>
<div class="kpi"><div class="muted">Categorical Features</div><div class="num">{{ feature_info.categorical_features|length }}</div></div>
<div class="kpi"><div class="muted">Saved Models Evaluated</div><div class="num">{{ model_metrics|length }}</div></div>
</div>

<div class="chart-grid">
<div class="card">
<h3>Model Performance Comparison</h3>
<p class="muted">Recorded evaluation metrics for Random Forest and ANN.</p>
<div class="chart-wrap"><canvas id="performanceChart"></canvas></div>
</div>
<div class="card">
<h3>Feature Type Distribution</h3>
<p class="muted">Distribution of saved numerical and categorical features.</p>
<div class="chart-wrap"><canvas id="featureChart"></canvas></div>
</div>
</div>

<div class="grid">
<div class="card"><h3>Models Used</h3><p>Random Forest Classifier</p><p>Artificial Neural Network (MLP)</p></div>
<div class="card"><h3>Project Scope</h3><p>Data analysis, preprocessing, categorical encoding, missing-value handling, classification-model training, evaluation, and Flask dashboard development.</p></div>
</div>
{% endif %}

{% if page == 'features' %}
<div class="hero"><h1>Feature Information</h1><p class="muted">Saved feature metadata from the original preprocessing pipeline.</p></div>
<div class="card" style="margin-top:22px"><h3>Categorical Features</h3>
{% for f in feature_info.categorical_features %}<span class="badge">{{f}}</span>{% endfor %}
</div>
<div class="card" style="margin-top:18px"><h3>Numerical Features</h3>
{% for f in feature_info.numerical_features %}<span class="badge">{{f}}</span>{% endfor %}
</div>
<div class="card" style="margin-top:18px"><h3>Sample Saved Medians</h3>
<table><tr><th>Feature</th><th>Median</th></tr>
{% for key,value in median_items %}<tr><td>{{key}}</td><td>{{value}}</td></tr>{% endfor %}
</table></div>
<div class="card" style="margin-top:18px"><h3>Feature Distribution</h3>
<div class="chart-wrap"><canvas id="featureChart"></canvas></div>
</div>
{% endif %}

{% if page == 'metrics' %}
<div class="hero"><h1>Saved Model Metrics</h1><p class="muted">Recorded evaluation values from the original project.</p></div>
<div class="card" style="margin-top:22px">
<h3>Random Forest vs ANN</h3>
<div class="chart-wrap"><canvas id="performanceChart"></canvas></div>
</div>

{% for model_name, m in model_metrics.items() %}
<div class="card" style="margin-top:22px">
<h3>{{ 'Random Forest' if model_name == 'rf' else 'Artificial Neural Network' }}</h3>
<div class="metric-grid">
<div class="metric">Accuracy<b>{{ "%.2f"|format(m.accuracy*100) }}%</b></div>
<div class="metric">Precision<b>{{ "%.2f"|format(m.precision*100) }}%</b></div>
<div class="metric">Recall<b>{{ "%.2f"|format(m.recall*100) }}%</b></div>
<div class="metric">F1 Score<b>{{ "%.2f"|format(m.f1*100) }}%</b></div>
<div class="metric">ROC AUC<b>{{ "%.2f"|format(m.roc_auc*100) }}%</b></div>
</div></div>
{% endfor %}
<div class="notice">These figures are displayed directly from the saved <code>model_metrics.pkl</code> artifact and are not recomputed during deployment.</div>
{% endif %}

{% if page == 'about' %}
<div class="hero"><h1>About the Project</h1><p class="muted">BS Computer Science AI semester project, Spring 2025.</p></div>
<div class="grid">
<div class="card"><h3>Muhammad Hashir & Saad Akbar</h3><p><strong>Role:</strong> Model Training & Preprocessing</p><p>Worked on data cleaning, feature engineering, and predictive-model development.</p></div>
<div class="card"><h3>Hassan Alizai</h3><p><strong>Role:</strong> Frontend, Backend Integration & Documentation</p><p>Worked on interface design, backend integration, and documentation.</p></div>
</div>
{% endif %}

</main></div>

<script>
const chartData = {{ chart_payload | safe }};

function renderPerformanceChart() {
  const el = document.getElementById('performanceChart');
  if (!el) return;
  new Chart(el, {
    type: 'bar',
    data: {
      labels: chartData.labels,
      datasets: [
        { label: 'Random Forest', data: chartData.rf },
        { label: 'ANN', data: chartData.ann }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true, max: 100, ticks: { callback: v => v + '%' } }
      }
    }
  });
}

function renderFeatureChart() {
  const el = document.getElementById('featureChart');
  if (!el) return;
  new Chart(el, {
    type: 'doughnut',
    data: {
      labels: ['Numerical', 'Categorical'],
      datasets: [{
        data: [chartData.feature_counts.numerical, chartData.feature_counts.categorical]
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: 'bottom' } }
    }
  });
}

renderPerformanceChart();
renderFeatureChart();
</script>

</body>
</html>
"""

def context(page):
    return dict(
        page=page,
        feature_info=feature_info,
        model_metrics=model_metrics,
        median_items=list(feature_info.get("medians", {}).items())[:12],
        chart_payload=json.dumps(chart_payload),
    )

@app.route("/")
def overview():
    return render_template_string(HTML, **context("overview"))

@app.route("/features")
def features():
    return render_template_string(HTML, **context("features"))

@app.route("/metrics")
def metrics():
    return render_template_string(HTML, **context("metrics"))

@app.route("/about")
def about():
    return render_template_string(HTML, **context("about"))

if __name__ == "__main__":
    app.run(debug=True)