# Diabetes Prediction

A machine-learning project that predicts whether a patient has diabetes from basic clinical data. It covers the full workflow: exploratory data analysis, preprocessing, model comparison with cross-validation, hyperparameter tuning, evaluation on a held-out test set, and a small interface to try the model.

- **Task:** binary classification (`Diabetes_Outcome`: 0 = non-diabetic, 1 = diabetic)
- **Dataset:** [Diabetes Dataset on Kaggle](https://www.kaggle.com/datasets/prince7489/diabetes-dataset) (500 patients, 8 features)
- **Best model:** Random Forest, F1 = 0.982 on the test set
- **Tools:** Python, pandas, scikit-learn, imbalanced-learn, seaborn, Streamlit

## Table of contents
1. [Project structure](#project-structure)
2. [Dataset](#dataset)
3. [Method](#method)
4. [Results](#results)
5. [Limitations](#limitations)
6. [How to run](#how-to-run)

## Project structure

```
diabetes-prediction/
├── data/
│   ├── raw/diabetes_dataset.csv        original dataset
│   └── processed/                      train.csv and test.csv (created by notebook 02)
├── notebooks/
│   ├── 01_eda.ipynb                    exploratory data analysis
│   ├── 02_preprocessing.ipynb          cleaning and stratified train/test split
│   ├── 03_modeling.ipynb               pipeline, cross-validation, tuning, evaluation
│   └── processing.ipynb                early draft of the EDA (can be ignored)
├── models/best_diabetes_model.joblib   saved pipeline (preprocessing + model)
├── app/
│   ├── app.py                          Streamlit web interface
│   └── app_test.ipynb                  notebook interface (ipywidgets)
├── requirements.txt
└── README.md
```

## Dataset

500 patients, no missing values, no duplicates. The classes are almost balanced: 54.2% diabetic and 45.8% non-diabetic.

| Column | Type | Description | Range / values |
|---|---|---|---|
| `Patient_ID` | text | Patient identifier (dropped, no predictive value) | P0001, P0002, ... |
| `Age` | integer | Age of the patient | 20 to 79 |
| `Gender` | category | Gender | `F`, `M` |
| `BMI` | float | Body mass index | 18.1 to 40.0 |
| `Glucose_Level` | integer | Glucose level | 70 to 199 |
| `Blood_Pressure` | integer | Blood pressure | 60 to 159 |
| `Insulin` | float | Insulin level | 15.4 to 299.5 |
| `Physical_Activity` | integer | Physical activity score | 0 to 119 |
| `Family_History` | category | Family history of diabetes | `Yes`, `No` |
| `Diabetes_Outcome` | integer | **Target** | 0 or 1 |

## Method

1. **Exploratory analysis** (`01_eda`): data quality, class balance, distributions, outliers (IQR rule), relation of each feature to the target, correlations.
2. **Preprocessing** (`02_preprocessing`): drop `Patient_ID`, then a stratified 80/20 split (400 training rows, 100 test rows) saved to `data/processed/`.
3. **Pipeline** (`03_modeling`): numeric features are standardised (`StandardScaler`) and categorical features are one-hot encoded, inside a scikit-learn `Pipeline`. Fitting these steps only on training folds avoids data leakage.
4. **Model comparison:** 5-fold stratified cross-validation of Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting and SVM, scored with accuracy, precision, recall, F1 and ROC-AUC.
5. **Class imbalance:** no correction vs `class_weight="balanced"` vs SMOTE. The classes are almost balanced, so none of them brings a meaningful gain.
6. **Tuning:** `GridSearchCV` on a Random Forest, optimising F1. Best parameters: `max_depth=5`, `min_samples_leaf=1`, `n_estimators=100`.
7. **Final evaluation:** the test set is used once, at the end.

## Results

Cross-validation on the training set (5 folds):

| Model | F1 |
|---|---|
| Gradient Boosting | 0.993 |
| Decision Tree | 0.993 |
| Random Forest | 0.989 |
| SVM (RBF) | 0.905 |
| Logistic Regression | 0.901 |
| KNN | 0.819 |

Final model (Random Forest) on the held-out test set of 100 patients:

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0.98 | 0.964 | 1.00 | 0.982 | 1.00 |

Confusion matrix: **TN = 44, FP = 2, FN = 0, TP = 54**. No diabetic patient is missed, which matters most in a medical setting. The cross-validated F1 (0.991) and the test F1 (0.982) are close, so there is no sign of overfitting.

The most important features (permutation importance) are `Family_History`, `BMI` and `Glucose_Level`. The other features have almost no influence on the model.

## Limitations

- The dataset is small (500 patients) and most likely **synthetic**: simple decision trees reach about 99% in cross-validation, which suggests that the target follows threshold rules on a few variables. Scores on real clinical data would be lower.
- The test set has only 100 patients, so each error changes the accuracy by 1 point.
- This is an educational project. **It is not medical advice** and must not be used for diagnosis.

## How to run

```bash
git clone https://github.com/Manar-Mahmoud-hub/diabetes-prediction.git
cd diabetes-prediction
pip install -r requirements.txt
```

Run the notebooks in order (`01_eda`, `02_preprocessing`, `03_modeling`):

```bash
jupyter notebook
```

Try the saved model on a single patient with the web interface (needs `models/best_diabetes_model.joblib`):

```bash
streamlit run app/app.py
```

Or open `app/app_test.ipynb` in Jupyter and run its three cells.

Example patients to test the interface:

| Patient | Age | Gender | BMI | Glucose | Blood pressure | Insulin | Activity | Family history | Expected |
|---|---|---|---|---|---|---|---|---|---|
| At risk | 58 | F | 36.4 | 151 | 102 | 98.5 | 86 | Yes | Diabetic (about 97%) |
| Healthy | 34 | F | 24.0 | 86 | 139 | 181.0 | 46 | No | Non-diabetic (about 3%) |
