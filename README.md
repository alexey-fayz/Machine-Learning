# Machine Learning & Data Analysis Projects

A collection of practical machine learning and data analysis projects covering the complete workflow of a typical data science pipeline: data exploration, preprocessing, model training, hyperparameter optimization, evaluation, and model persistence.

The repository contains several independent tasks focused on different machine learning techniques, ranging from statistical analysis with NumPy and Pandas to classification, regression, ensemble learning, and gradient boosting.

---

## Project Overview

| Project                 | Description                                                                                                                                                        | Technologies & Skills                          |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------- |
| **pandas_math_task**    | Statistical analysis of datasets using NumPy, Pandas and SciPy. Includes descriptive statistics, grouping, filtering, correlations, and probability distributions. | NumPy, Pandas, SciPy, Statistics               |
| **knn_task**            | K-Nearest Neighbors classification with feature scaling, cross-validation and hyperparameter tuning.                                                               | Scikit-learn, KNN, Data Preprocessing          |
| **knn_task_advanced**   | Improved KNN implementation with extensive parameter search and model optimization.                                                                                | Model Optimization, Cross-Validation           |
| **linear_reg_task**     | Linear regression pipeline with feature engineering and polynomial transformations.                                                                                | Regression, Pipelines, Feature Engineering     |
| **logreg_task**         | Logistic regression for imbalanced datasets with threshold optimization and precision-recall analysis.                                                             | Classification, Imbalanced Learning            |
| **decision_trees_task** | Comparison of Decision Tree, Random Forest and CatBoost models using multiple evaluation metrics.                                                                  | Ensemble Learning, Gradient Boosting, CatBoost |

---

## Repository Structure

```text
.
├── pandas_math_task/
│   ├── tasks.py
│   └── data/
│
├── knn_task/
│   ├── knn_task.py
│   ├── data/
│   └── result/
│
├── knn_task_advanced/
│   ├── knn_task_advanced.py
│   ├── data/
│   └── result/
│
├── linear_reg_task/
│   ├── linear_reg_task.py
│   ├── data/
│   └── result/
│
├── logreg_task/
│   ├── logreg_task.py
│   ├── data/
│   └── result/
│
├── decision_trees_task/
│   ├── decision_trees_task.py
│   └── data/
│
└── README.md
```

---

## Key Features

### Data Analysis

* Exploratory data analysis (EDA)
* Statistical metrics and descriptive statistics
* Correlation analysis
* Probability distributions
* Data filtering and aggregation

### Machine Learning

* Classification models
* Regression models
* Ensemble methods
* Gradient boosting
* Hyperparameter optimization
* Cross-validation

### Model Evaluation

* Accuracy
* Precision & Recall
* F1-score
* ROC-AUC
* Mean Squared Error (MSE)
* R² Score

### Model Persistence

* Saving trained models
* Saving preprocessing pipelines
* Reusing trained artifacts for inference

---

## Technologies

### Programming Language

* Python 3.8+

### Data Processing

* NumPy
* Pandas
* SciPy

### Machine Learning

* Scikit-learn
* CatBoost

### Utilities

* Pickle
* Virtual Environments

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your_username/machine-learning-projects.git
cd machine-learning-projects
```

Install dependencies:

```bash
pip install numpy pandas scipy scikit-learn catboost
```

---

## Running a Project

Navigate to the desired project folder and run the corresponding script.

Example:

```bash
cd knn_task
python knn_task.py
```

or

```bash
cd linear_reg_task
python linear_reg_task.py
```

---

## Learning Objectives

The purpose of this repository is to demonstrate practical experience with:

* Data preprocessing and feature engineering
* Statistical data analysis
* Machine learning model development
* Hyperparameter tuning
* Evaluation of classification and regression models
* Working with imbalanced datasets
* Ensemble learning techniques
* Building reproducible ML workflows

---

## Notes

* Training datasets are included in the repository.
* Test datasets and labels are intentionally hidden to simulate real-world machine learning scenarios.
* Some projects require saving trained models and preprocessing artifacts for later evaluation.
* Performance targets are defined individually for each task.

---

## Skills Demonstrated

* Python Programming
* Data Analysis
* Machine Learning
* Feature Engineering
* Model Optimization
* Statistical Analysis
* Ensemble Methods
* Gradient Boosting
* Data Preprocessing
* Model Evaluation
* Software Engineering Practices
