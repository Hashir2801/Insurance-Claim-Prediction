# 🚗 Insurance Claim Prediction – AI Semester Project (Spring 2025)

A complete end-to-end Machine Learning project built using **Flask**, **scikit-learn**, and the **Porto Seguro Safe Driver Prediction** dataset. This system predicts whether a customer is likely to make an insurance claim, using powerful classification models and an interactive web interface.

---

## 📁 Project Structure

```
INSURANCE_CLAIM_PREDICTOR/
│
├── app.py                    # Main Flask application
├── configure.py              # Configuration settings
├── data_analysis.py          # Data summary logic
├── preprocessing.py          # Data cleaning and transformation logic
├── code.py                   # Model loading and prediction logic
│
├── static/                   # CSS styles and static assets
│   └── css/style.css
│
├── templates/                # HTML pages (Jinja2 templates)
│   ├── base2.html
│   ├── analysis.html
│   ├── preprocessing.html
│   ├── prediction.html
│   ├── visualization.html
│   └── about.html
│
├── project_dataset/          # CSV data files
│   ├── train.csv
│   └── test.csv
│
├── Project_Models/           # Trained ML model files and visualizations
│   ├── rf_model_v1.0.pkl
│   ├── ann_model_v1.0.pkl
│   ├── confusion matrices
│   └── ROC curves
│
└── uploads/                  # Any uploaded files (optional)
```

---

## 🧠 Models Used

- **Random Forest Classifier**  
- **Artificial Neural Network (ANN)**
- Hyperparameter tuning and model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC AUC

---

## 📊 Features

- Clean and modern UI (custom HTML/CSS)
- Dynamic prediction interface
- Data analysis and EDA visualizations
- Preprocessing stats: missing values, encoders, median values
- Results with performance metrics and charts

---

## 💡 Technologies

| Area        | Tools Used                          |
|-------------|-------------------------------------|
| Language    | Python                              |
| Framework   | Flask                               |
| Libraries   | pandas, scikit-learn, matplotlib    |
| Frontend    | HTML, CSS, Bootstrap                |
| Versioning  | Git + GitHub                        |

---

## 📥 How to Run

```bash
git clone https://github.com/Hashir2801/Insurance-Claim-Prediction.git
cd INSURANCE_CLAIM_PREDICTOR
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt   # If you have it
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## 👨‍💻 Contributors

| Name              | Role                                     |
|-------------------|------------------------------------------|
| Muhammad Hashir   | Data Analysis, Model Training, UI Design |
| Hassan Alizai     | Frontend, Backend Integration, Docs      |
| Saad Akbar        | Data Cleaning, Optimization               |

---

## 📚 Dataset

- Porto Seguro’s Safe Driver Prediction – [Kaggle Dataset](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction)

---

## 📘 License

This project is part of **BS-CS AI Semester Project (Spring 2025)** and is intended for educational use only.
