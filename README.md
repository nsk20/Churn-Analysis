# Telco Customer Churn Analysis and Prediction

## Project Overview
This project focuses on analyzing and predicting customer churn for a telecommunications company using machine learning techniques. The analysis involves data preprocessing, exploratory data analysis (EDA), and building a predictive model using PySpark and scikit-learn.

## Data Preprocessing

### Data Loading
The dataset, `Telco-Customer-Churn.csv`, is loaded into a Pandas DataFrame for initial exploration.

### Data Cleaning
- The `customerID` column is dropped as it is not relevant for analysis.
- The `TotalCharges` column, initially of type `object`, is converted to a numeric type to facilitate analysis. Missing values in this column are imputed using the mean value.

### Feature Categorization
Features are categorized into categorical and continuous types. Categorical features include `gender`, `Partner`, `Dependents`, etc., while continuous features include `SeniorCitizen`, `tenure`, `MonthlyCharges`, and `TotalCharges`.

## Exploratory Data Analysis (EDA)

### Churn Distribution
A count plot is used to visualize the distribution of churn, showing that 73% of customers did not churn, while 27% did.

### Categorical Features
Bar charts are used to visualize the frequency of categorical features relative to churn. For example, the analysis shows higher churn rates for customers with "Fiber optic" internet service compared to those with "DSL".

### Continuous Features
Histograms are used to visualize the distribution of continuous features, such as `tenure` and `MonthlyCharges`, for churned and non-churned customers.

## Machine Learning Model

### Data Preparation
- The dataset is split into training (80%) and testing (20%) sets.
- `StringIndexer` is used to convert categorical features into a format suitable for machine learning algorithms.
- `VectorAssembler` is used to assemble all feature columns into a single vector.

### Model Training
A `RandomForestClassifier` is used to build the predictive model due to its ability to handle non-linear relationships and interactions between features. The model is trained using the training dataset.

### Model Evaluation
The model's performance is evaluated using the Area Under the ROC Curve (AUC-ROC) metric, which is found to be 0.66913. The model is saved for future use, and the predictions are exported to CSV files for further analysis.

## Conclusion
This project demonstrates a comprehensive approach to customer churn analysis and prediction using machine learning. By preprocessing the data, performing EDA, and building a predictive model, valuable insights are gained into customer behavior and churn patterns. The model can be used to identify at-risk customers and inform retention strategies.
