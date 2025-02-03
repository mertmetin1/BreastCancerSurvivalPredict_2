# Breast Cancer Survival Prediction Project

## Overview
This project is a comprehensive pipeline for predicting breast cancer survival using state-of-the-art machine learning techniques. It leverages advanced data preprocessing, exploratory data analysis (EDA), and model evaluation to identify the most effective algorithms for predicting outcomes. The workflow includes hyperparameter tuning, feature importance analysis, and detailed performance reporting.

## Features
- **Data Preprocessing**:
  - Handling missing values and outliers.
  - Encoding categorical variables.
  - Feature scaling and engineering.
- **Exploratory Data Analysis (EDA)**:
  - Statistical summaries and visual insights into the dataset.
  - Correlation analysis between features and target variable.
- **Class Balancing**:
  - Addressing class imbalance using SMOTE (Synthetic Minority Oversampling Technique).
- **Machine Learning Models**:
  - Logistic Regression.
  - Random Forest Classifier.
  - XGBoost Classifier.
- **Hyperparameter Tuning**:
  - Optimizing model parameters using GridSearchCV.
- **Feature Importance Analysis**:
  - SHAP (SHapley Additive exPlanations).
  - Partial Dependence Plots (PDP).
- **Comprehensive Evaluation**:
  - Performance metrics including accuracy, precision, recall, F1-score, and ROC-AUC.
  - Visualization of confusion matrices and ROC curves.
- **Model Saving and Reusability**:
  - All trained models are saved for future inference.

## Project Structure
```
├── data
│   ├── raw                  # Original dataset
│   └── processed            # Preprocessed dataset
├── models                   # Saved machine learning models
├── plots                    # Visualizations generated during analysis
├── results                  # Performance metrics and comparison tables
├── src                      # Modular source code (if applicable)
├── BreastCancerClassificationReport.pdf  # Summary of findings
├── main.py                  # Main script to execute the pipeline
├── README.md                # Documentation
└── requirements.txt         # Dependencies
```

## Installation
Follow these steps to set up and run the project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/mertmetin1/breast-cancer-survival-prediction.git
   cd breast-cancer-survival-prediction
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
The dataset is located in the `data/raw/` directory. It includes key features such as:
- Age.
- Tumor Size.
- Survival Months.
- Hormone receptor statuses (e.g., Estrogen, Progesterone).
- Regional lymph node examination results.

### Data Preprocessing
- Missing values are imputed or dropped based on relevance.
- Categorical variables are encoded using Label Encoding.
- Numerical features are normalized using MinMaxScaler.

## Usage
Run the entire pipeline by executing:
```bash
python main.py
```
### Outputs
- **Visualizations**: Saved in the `plots/` directory.
- **Trained Models**: Saved in the `models/` directory.
- **Evaluation Results**: Saved in the `results/` directory as CSV and visual summaries.

## Models and Evaluation
The following models are implemented and evaluated:
1. **Logistic Regression**: Baseline linear model with and without hyperparameter tuning.
2. **Random Forest**: Ensemble method for robust performance.
3. **XGBoost**: Gradient boosting algorithm for optimal results.

### Performance Metrics
- Accuracy: Overall correctness of the model.
- Precision & Recall: Evaluates class-wise performance.
- F1-score: Balances precision and recall.
- ROC-AUC: Measures discriminatory power of the classifier.

### Model Comparisons
All models are compared using detailed metrics saved in `results/comparison_results.csv`. Confusion matrices and ROC curves are visualized for insights into classification performance.

## Visualizations
1. **EDA Visualizations**:
   - Pair plots to identify feature relationships.
   - ![resim](https://github.com/user-attachments/assets/6a7c3369-d2b5-453a-b47c-d8678b26b2bf)

   - ![resim](https://github.com/user-attachments/assets/b20e6f86-1118-4489-a0fa-836bc747fd09)

2. **Model Performance**:
   - Confusion matrices for error analysis.
 
   - ![resim](https://github.com/user-attachments/assets/d9ce78f3-4fdb-4af9-bb13-19e1a9b11727)
   - ![resim](https://github.com/user-attachments/assets/cbecb92f-f23e-4203-95b9-4192bda50200)
   - ![resim](https://github.com/user-attachments/assets/733a437e-c98a-4be0-a964-3888f4a324bd)





   - ROC and Precision-Recall curves for evaluation.
   - ![resim](https://github.com/user-attachments/assets/76ec4688-3088-4815-8088-dbf952cbf0a8)
   - ![resim](https://github.com/user-attachments/assets/48ec8c85-ee33-42fc-9640-2eb576bdda2e)
   - ![resim](https://github.com/user-attachments/assets/5d1a1a7c-c0ed-4c92-8bea-cac02bffc410)




   - 
3. **Feature Importance**:
   - ![resim](https://github.com/user-attachments/assets/a889c345-f964-4f48-be65-1c5f8138c264)

   -![resim](https://github.com/user-attachments/assets/11f5953c-2e52-4032-a655-806b176f5b92)


## Results
The best-performing models and hyperparameter configurations are highlighted in the results. Key findings include:
- Significant predictors of survival.
- Optimized model parameters for maximum performance.
- Visual explanations for feature importance.


