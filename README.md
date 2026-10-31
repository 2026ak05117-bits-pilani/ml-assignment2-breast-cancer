# ML Assignment 2 - Breast Cancer Classification

Name: Ravish H S  
ID: 2026AK05117  
Course: M.Tech AIML - Machine Learning  

---

## a. Problem statement

I chose a binary classification problem using the UCI Breast Cancer Wisconsin dataset. The aim is to predict if a sample is malignant (0) or benign (1) using 30 numeric features.

I trained 5 classifiers on the same data, compared them using Accuracy, AUC, Precision, Recall, F1 and MCC, then built a Streamlit app to upload test data and show metrics for the saved models.

---

## b. Dataset description

- Dataset: Breast Cancer Wisconsin (Diagnostic)
- Source: UCI ML Repository
- Link: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Type: Binary classification
- Rows: 569 (more than 500)
- Features: 30 (more than 12)
- Target: diagnosis (0 = malignant, 1 = benign)
- Class counts: Benign 357, Malignant 212
- Missing values: none

Features are mean / error / worst values of radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension.

Train-test split: 75% / 25%, stratified, random_state=42  
Train rows: 426  
Test rows: 143 (saved as test_data.csv)

I picked this dataset because it is from UCI, has enough features/rows, and works well for all required metrics.

---

## c. Github Repository Link

Github username: 2026ak05117-bits-pilani

Repo: https://github.com/2026ak05117-bits-pilani/ml-assignment2-breast-cancer

Live Streamlit App Link: https://2026ak05117-bits-pilani-ml-assignment2-breast-cancer-app-8c45ms.streamlit.app/

---

## d. Models used

All 5 models were trained on the same train split. I used StandardScaler inside Pipeline so preprocessing is saved with each model.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9860 | 0.9977 | 0.9889 | 0.9889 | 0.9889 | 0.9700 |
| Decision Tree | 0.9371 | 0.9186 | 0.9551 | 0.9444 | 0.9497 | 0.8657 |
| KNN | 0.9790 | 0.9845 | 0.9677 | 1.0000 | 0.9836 | 0.9555 |
| Naive Bayes | 0.9371 | 0.9878 | 0.9355 | 0.9667 | 0.9508 | 0.8644 |
| Support Vector Machine | 0.9790 | 0.9969 | 0.9888 | 0.9778 | 0.9832 | 0.9553 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best model on this dataset. Highest Accuracy (0.9860), AUC (0.9977), F1 and MCC. After scaling, a linear boundary separates malignant vs benign cases very well. |
| Decision Tree | Weakest among the five. Accuracy 0.9371 and AUC 0.9186 are lowest. Tree with max_depth=5 is a bit unstable on these continuous medical features compared to LR/SVM. |
| KNN | Strong Accuracy (0.9790). Recall is 1.0 on the test split, so it did not miss benign cases here. Needs scaling because it is distance based. Precision is slightly lower than LR/SVM. |
| Naive Bayes | AUC is good (0.9878) but Accuracy and MCC are lower. GaussianNB assumes feature independence, which is not fully true for these correlated cell measurements. |
| Support Vector Machine | Almost as good as Logistic Regression. AUC 0.9969 and strong Precision/F1. RBF kernel helps a little with mild non-linearity, but gain over LR is small on this data. |
| Overall, Winner for your dataset? | Logistic Regression |

---

## Files in this repo

- app.py : streamlit app
- train_models.py : training script
- ml_assignment2.ipynb : notebook for BITS lab
- requirements.txt
- README.md
- test_data.csv
- train_data.csv
- full_dataset.csv
- model/ : saved .joblib models + meta.json

---

## How to run

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

In the app, upload test_data.csv, select a model, and check metrics / confusion matrix.

The streamlit app only loads saved models. It does not retrain.
