# ML Assignment 2
# Train 5 classifiers on UCI breast cancer data and save them
# Ravish H S - 2026AK05117

import warnings
warnings.filterwarnings('ignore')

import os
import json
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)
import joblib

SEED = 42
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    # using sklearn loader - same UCI breast cancer wisconsin dataset
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df = df.rename(columns={'target': 'diagnosis'})
    return df


def get_scores(model, X_test, y_test):
    # all 6 metrics from the assignment
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return {
        'accuracy': round(float(accuracy_score(y_test, pred)), 4),
        'auc': round(float(roc_auc_score(y_test, prob)), 4),
        'precision': round(float(precision_score(y_test, pred)), 4),
        'recall': round(float(recall_score(y_test, pred)), 4),
        'f1': round(float(f1_score(y_test, pred)), 4),
        'mcc': round(float(matthews_corrcoef(y_test, pred)), 4),
    }


def main():
    df = load_data()
    print('shape:', df.shape)
    print('features:', df.shape[1] - 1)
    print(df['diagnosis'].value_counts())

    feat_cols = [c for c in df.columns if c != 'diagnosis']
    X = df[feat_cols]
    y = df['diagnosis']

    # 75-25 split, keep class ratio same
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )

    # save csv files for github / streamlit
    df.to_csv(os.path.join(BASE, 'full_dataset.csv'), index=False)

    train_df = X_train.copy()
    train_df['diagnosis'] = y_train.values
    train_df.to_csv(os.path.join(BASE, 'train_data.csv'), index=False)

    test_df = X_test.copy()
    test_df['diagnosis'] = y_test.values
    test_df.to_csv(os.path.join(BASE, 'test_data.csv'), index=False)
    print('test rows:', len(test_df))

    # all 5 models with scaler inside pipeline
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, random_state=SEED)),
        ]),
        'Decision Tree': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', DecisionTreeClassifier(random_state=SEED, max_depth=5)),
        ]),
        'KNN': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=5)),
        ]),
        'Naive Bayes': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GaussianNB()),
        ]),
        'Support Vector Machine': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', probability=True, random_state=SEED)),
        ]),
    }

    files = {
        'Logistic Regression': 'logistic_regression.joblib',
        'Decision Tree': 'decision_tree.joblib',
        'KNN': 'knn.joblib',
        'Naive Bayes': 'naive_bayes.joblib',
        'Support Vector Machine': 'svm.joblib',
    }

    metrics_list = []
    print('\nTest set metrics:')
    for name, model in models.items():
        model.fit(X_train, y_train)
        scores = get_scores(model, X_test, y_test)
        scores['model'] = name
        metrics_list.append(scores)

        # save model
        joblib.dump(model, os.path.join(MODEL_DIR, files[name]))
        print(name, scores)

    # save some info so streamlit app can use it
    meta = {
        'dataset': 'Breast Cancer Wisconsin (Diagnostic) - UCI',
        'source_url': 'https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic',
        'target_column': 'diagnosis',
        'target_meaning': {'0': 'malignant', '1': 'benign'},
        'feature_columns': feat_cols,
        'model_files': files,
        'metrics': metrics_list,
        'train_size': int(len(X_train)),
        'test_size': int(len(X_test)),
        'n_features': len(feat_cols),
        'n_instances': int(len(df)),
        'random_state': SEED,
    }
    with open(os.path.join(MODEL_DIR, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print('\nDone. Files saved in', BASE)


if __name__ == '__main__':
    main()
