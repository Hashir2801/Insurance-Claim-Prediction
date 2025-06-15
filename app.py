from flask import Flask, render_template, request, redirect, url_for
import pickle, pandas as pd, numpy as np, os
from data_analysis import perform_data_analysis
from preprocessing import perform_preprocessing
from visualization import create_visualizations

app = Flask(__name__)

class Config:
    MODEL_DIR = 'Project_Models'
    DATA_DIR = 'Project_Dataset'
    STATIC_IMG_DIR = 'static/Project_images'

app.config.from_object(Config)

def load_models():
    models = {}
    try:
        with open(f'{app.config["MODEL_DIR"]}/rf_model_v1.0.pkl', 'rb') as f:
            models['rf'] = pickle.load(f)
        with open(f'{app.config["MODEL_DIR"]}/ann_model_v1.0.pkl', 'rb') as f:
            models['ann'] = pickle.load(f)
        with open(f'{app.config["MODEL_DIR"]}/encoder.pkl', 'rb') as f:
            models['encoder'] = pickle.load(f)
        with open(f'{app.config["MODEL_DIR"]}/scaler.pkl', 'rb') as f:
            models['scaler'] = pickle.load(f)
        with open(f'{app.config["MODEL_DIR"]}/imputer.pkl', 'rb') as f:
            models['imputer'] = pickle.load(f)
        with open(f'{app.config["MODEL_DIR"]}/feature_info.pkl', 'rb') as f:
            models['feature_info'] = pickle.load(f)
        with open(f'{app.config["MODEL_DIR"]}/model_metrics.pkl', 'rb') as f:
            models['metrics'] = pickle.load(f)
        return models
    except Exception as e:
        raise RuntimeError(f"Failed to load models: {str(e)}")

def load_data():
    try:
        train = pd.read_csv(f'{app.config["DATA_DIR"]}/train.csv')
        test = pd.read_csv(f'{app.config["DATA_DIR"]}/test.csv')
        return train, test
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {str(e)}")

try:
    models = load_models()
    train_data, test_data = load_data()
    os.makedirs(f'{app.config["STATIC_IMG_DIR"]}/visualizations', exist_ok=True)
    with app.app_context():
        create_visualizations(train_data)
except Exception as e:
    print(f"Initialization error: {e}")
    raise

@app.route('/')
def home():
    return redirect(url_for('data_analysis'))

@app.route('/data_analysis')
def data_analysis():
    res = perform_data_analysis(train_data)
    return render_template('analysis.html', results=res)

@app.route('/preprocessing')
def preprocessing():
    res = perform_preprocessing(train_data)
    return render_template('preprocessing.html', results=res, feature_info=models['feature_info'], encoder=models['encoder'])

@app.route('/visualization')
def visualization():
    files = [f for f in os.listdir(f'{app.config["STATIC_IMG_DIR"]}/visualizations') if f.endswith('.png')]
    return render_template('visualization.html', viz_files=files, feature_info=models['feature_info'])

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        try:
            form = request.form.to_dict()
            inp = preprocess_input(form, models)
            rf_pred, rf_prob = predict_with_rf(inp, models)
            ann_pred, ann_prob = predict_with_ann(inp, models)
            return render_template('prediction.html', rf_pred=rf_pred, rf_prob=rf_prob, ann_pred=ann_pred, ann_prob=ann_prob, model_metrics=models['metrics'], feature_info=models['feature_info'])
        except Exception as e:
            return render_template('prediction.html', error=str(e), model_metrics=models['metrics'], feature_info=models['feature_info'])
    return render_template('prediction.html', model_metrics=models['metrics'], feature_info=models['feature_info'])

@app.route('/about')
def about():
    return render_template('about.html', model_metrics=models['metrics'])

def preprocess_input(form_data, models):
    df = pd.DataFrame(columns=models['feature_info']['original_features'])
    for f in df.columns:
        if f in form_data:
            try: df[f] = [float(form_data[f])]
            except: df[f] = [models['feature_info']['medians'].get(f, 0)]
        else:
            df[f] = [models['feature_info']['medians'].get(f, 0)]
    for col in models['feature_info']['numerical_features']:
        if col in df.columns:
            df[col] = df[col].replace(-1, models['feature_info']['medians'].get(col, 0))
    cat = df[models['feature_info']['categorical_features']].astype(str)
    enc = models['encoder'].transform(cat)
    combined = np.hstack([df[models['feature_info']['numerical_features']].values, enc])
    return models['imputer'].transform(combined)

def predict_with_rf(data, models):
    return models['rf'].predict(data)[0], models['rf'].predict_proba(data)[0,1]

def predict_with_ann(data, models):
    scaled = models['scaler'].transform(data)
    return models['ann'].predict(scaled)[0], models['ann'].predict_proba(scaled)[0,1]

if __name__ == '__main__':
    app.run(debug=False)
