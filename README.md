# 🚗 Insurance Claim Prediction

An AI semester project built around the **Porto Seguro Safe Driver Prediction** problem. The original project included data analysis, preprocessing, Random Forest and Artificial Neural Network models, evaluation metrics, and a Flask dashboard.

## Live Portfolio Version

The available project folder no longer contains the original Random Forest model, scaler, or training CSV dataset. Because those files are necessary to reproduce the original prediction pipeline accurately, the deployed version does not generate an incomplete or misleading claim prediction.

Instead, the live portfolio dashboard presents verified saved artifacts and recorded project results.

## Dashboard Features

- Project overview
- Model performance comparison chart
- Feature type distribution chart
- Random Forest vs ANN metrics
- Saved feature metadata
- Categorical and numerical feature lists
- Sample preprocessing medians
- Contributor information
- Responsive Flask-based dashboard

## Models Used

- Random Forest Classifier
- Artificial Neural Network (MLP)

## Recorded Evaluation Metrics

The dashboard reads the original evaluation values from:

`model_metrics.pkl`

The graphical model comparison displays:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

## Feature Metadata

Feature information is loaded from:

`feature_info.pkl`

This includes the original feature list, numerical features, categorical features, and saved preprocessing medians.

## Technologies

- Python
- Flask
- Machine Learning
- HTML/CSS
- Chart.js
- Vercel

## Required Files

```text
app.py
feature_info.pkl
model_metrics.pkl
requirements.txt
README.md
```

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Deployment

Push the repository to GitHub and import it into Vercel. Keep `app.py` at the repository root.

No `vercel.json` is required for the root-level Flask application.

## Project Note

The graphs shown in the live dashboard are based only on saved project metadata and stored model evaluation metrics. Dataset-dependent charts are intentionally not fabricated because the original CSV files are not present in the available repository.
