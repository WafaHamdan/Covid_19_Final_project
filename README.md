# COVID-19 Status Prediction

A machine learning project that predicts COVID-19 status using binary symptom, health-condition, and exposure-related indicators.

> **Disclaimer:** This project is intended for educational purposes only. It must not be used for medical diagnosis or clinical decision-making.

## Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WafaHamdan/Covid_19_Final_project/blob/main/COVID19_Prediction_Model_Updated.ipynb)

## Project Overview

This project explores and prepares a binary COVID-19 dataset, handles class imbalance, trains multiple classification models, and compares their performance.

The following models were evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree

## Project Objectives

- Examine the structure, quality, and class distribution of the dataset.
- Explore symptoms and exposure indicators associated with COVID-19 status.
- Prevent repeated feature patterns from appearing in both the training and test sets.
- Handle class imbalance using random oversampling on the training data only.
- Compare classification models using holdout testing and group-aware cross-validation.
- Evaluate performance using multiple classification metrics.

## Dataset

The dataset was obtained from Kaggle and contains binary values representing symptoms, existing health conditions, exposure indicators, and COVID-19 status.

The target variable is `COVID-19`:

- `Yes`: COVID-19 positive record
- `No`: COVID-19 negative record

All remaining columns are used as input features.

The dataset used by the notebook is available in this repository as:

`covid_dataset.csv`

## Project Workflow

1. Import the required Python libraries.
2. Load and inspect the dataset.
3. Clean column names and encode binary values.
4. Perform exploratory data analysis.
5. Examine the target-class distribution.
6. Compare feature prevalence by COVID-19 status.
7. Group identical feature patterns.
8. Create group-separated training and holdout test sets.
9. Apply random oversampling to the training set only.
10. Train Logistic Regression, KNN, and Decision Tree models.
11. Evaluate the models on the untouched holdout test set.
12. Perform 10-fold stratified group cross-validation.
13. Compare the models and select the final model.

## Data Leakage Prevention

The dataset contains repeated feature patterns. To reduce the risk of data leakage, rows with identical input-feature values were assigned to the same group.

Group-aware splitting was used to prevent the same feature pattern from appearing in both the training and test sets.

Random oversampling was applied only after the training and test sets were separated. The holdout test set remained untouched.

## Model Evaluation

The models were evaluated using:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1-score
- Macro F1-score
- Confusion matrix
- 10-fold stratified group cross-validation

## Key Results

| Evaluation result | Best model | Score |
|---|---|---:|
| Holdout accuracy | Decision Tree | 93.31% |
| Mean cross-validation accuracy | Decision Tree | 95.24% |
| Mean cross-validation balanced accuracy | Logistic Regression | 92.80% |
| Mean cross-validation macro F1-score | Decision Tree | 90.91% |

On the holdout test set, the Decision Tree correctly identified 749 of 757 COVID-19 positive records.

Decision Tree achieved the strongest overall accuracy, macro F1-score, and positive-class recall. Logistic Regression provided more balanced discrimination between the two target classes.

Based on the selected evaluation criteria, Decision Tree was chosen as the final model.

## Confusion Matrix Summary

The holdout confusion matrices produced the following results:

| Model | True Negative | False Positive | False Negative | True Positive |
|---|---:|---:|---:|---:|
| Logistic Regression | 264 | 11 | 82 | 675 |
| KNN | 206 | 69 | 93 | 664 |
| Decision Tree | 214 | 61 | 8 | 749 |

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Google Colab

## How to Run the Project

### Option 1: Google Colab

1. Click the **Open in Colab** button at the top of this page.
2. Select `Runtime`.
3. Click `Run all`.
4. If Colab displays a security warning, select `Run anyway`.
5. Wait until all notebook cells finish running.

The dataset is loaded directly from this GitHub repository.

### Option 2: Run Locally

Clone the repository:

```bash
git clone https://github.com/WafaHamdan/Covid_19_Final_project.git
cd Covid_19_Final_project
```

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

Start Jupyter Notebook:

```bash
jupyter notebook COVID19_Prediction_Model_Updated.ipynb
```

## Conclusion

Three classification models were evaluated using an untouched, group-separated holdout test set and 10-fold stratified group cross-validation.

Decision Tree achieved the highest mean cross-validation accuracy and macro F1-score, while Logistic Regression achieved the highest balanced accuracy. Decision Tree was selected as the final model based on its overall performance and positive-class recall.

## Developed By

**Wafa Hamdan Alshehri**
