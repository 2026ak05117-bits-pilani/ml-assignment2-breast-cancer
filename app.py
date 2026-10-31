# ML Assignment 2 - Streamlit app
# Ravish H S - 2026AK05117
# This app only loads already trained models. It does not retrain.

import os
import json
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title='ML Assign 2 - Breast Cancer', layout='wide')

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, 'model')


@st.cache_resource
def get_meta():
    # read model info saved during training
    f = open(os.path.join(MODEL_DIR, 'meta.json'), 'r', encoding='utf-8')
    data = json.load(f)
    f.close()
    return data


@st.cache_resource
def get_models(meta):
    # load all 5 saved models from model folder
    out = {}
    for name, fname in meta['model_files'].items():
        out[name] = joblib.load(os.path.join(MODEL_DIR, fname))
    return out


def calc_metrics(model, X, y):
    # calculate all 6 metrics needed for the assignment
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    scores = {
        'Accuracy': round(float(accuracy_score(y, pred)), 4),
        'AUC': round(float(roc_auc_score(y, prob)), 4),
        'Precision': round(float(precision_score(y, pred)), 4),
        'Recall': round(float(recall_score(y, pred)), 4),
        'F1 Score': round(float(f1_score(y, pred)), 4),
        'MCC': round(float(matthews_corrcoef(y, pred)), 4),
    }
    return scores, pred


def draw_cm(y_true, y_pred, title):
    # simple confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Malignant', 'Benign'],
                yticklabels=['Malignant', 'Benign'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    plt.tight_layout()
    return fig


def main():
    meta = get_meta()
    models = get_models(meta)
    feats = meta['feature_columns']
    target = meta['target_column']

    st.title('Breast Cancer Classification')
    st.write('ML Assignment 2 | Dataset: UCI Breast Cancer Wisconsin')
    st.write('Upload test_data.csv below. Models are already trained and saved.')

    with st.expander('About dataset'):
        st.write('Dataset:', meta['dataset'])
        st.write('Features:', meta['n_features'], '| Total rows:', meta['n_instances'])
        st.write('Target column is', target, '(0 = malignant, 1 = benign)')
        st.write('Models:', ', '.join(models.keys()))

    file = st.file_uploader('Upload test CSV', type=['csv'])

    if file is None:
        st.info('Please upload test_data.csv from the github repo')
        # show metrics from training time as reference
        st.subheader('Metrics from training (test split)')
        ref = pd.DataFrame(meta['metrics'])
        ref = ref[['model', 'accuracy', 'auc', 'precision', 'recall', 'f1', 'mcc']]
        ref.columns = ['ML Model Name', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        st.dataframe(ref, use_container_width=True)
        return

    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error('Error reading csv: ' + str(e))
        return

    # basic checks
    missing = [c for c in feats if c not in df.columns]
    if len(missing) > 0:
        st.error('Some feature columns are missing in the csv')
        return

    if target not in df.columns:
        st.error('Target column diagnosis is missing')
        return

    X = df[feats]
    y = df[target]
    st.success('Loaded ' + str(len(df)) + ' rows')

    choice = st.radio('Choose mode', ['One model', 'All models'], horizontal=True)

    if choice == 'One model':
        # dropdown for model selection
        name = st.selectbox('Select model', list(models.keys()))
        model = models[name]
        scores, pred = calc_metrics(model, X, y)

        st.subheader('Metrics - ' + name)
        cols = st.columns(6)
        i = 0
        for k, v in scores.items():
            cols[i].metric(k, str(v))
            i += 1

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(draw_cm(y, pred, 'Confusion Matrix - ' + name))
        with c2:
            st.subheader('Classification Report')
            st.code(classification_report(y, pred, target_names=['Malignant', 'Benign']))

    else:
        # compare all models on the uploaded file
        st.subheader('Comparison of all models')
        rows = []
        preds = {}
        for name, model in models.items():
            scores, pred = calc_metrics(model, X, y)
            preds[name] = pred
            rows.append({
                'ML Model Name': name,
                'Accuracy': scores['Accuracy'],
                'AUC': scores['AUC'],
                'Precision': scores['Precision'],
                'Recall': scores['Recall'],
                'F1': scores['F1 Score'],
                'MCC': scores['MCC'],
            })

        table = pd.DataFrame(rows)
        st.dataframe(table, use_container_width=True)

        best = table.loc[table['Accuracy'].idxmax(), 'ML Model Name']
        st.success('Best model on this file (Accuracy): ' + str(best))

        pick = st.selectbox('Show confusion matrix for', list(models.keys()))
        st.pyplot(draw_cm(y, preds[pick], 'Confusion Matrix - ' + pick))
        st.subheader('Classification Report - ' + pick)
        st.code(classification_report(y, preds[pick], target_names=['Malignant', 'Benign']))

    st.write('Note: models are loaded from model folder. No training happens in this app.')


if __name__ == '__main__':
    main()
