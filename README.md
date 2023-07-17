# Module_20_Challenge-Credit_Risk_Classification
This project aims to develop a logistic regression model for credit risk classification. The objective is to predict whether a loan is healthy or high-risk based on various features provided by the lending company.

## Requirements
* Python v3.10.9
* The following libraries are required to run the analysis code:
* numpy (imported as np): A library for numerical computations and array operations.
* pandas (imported as pd): A library for data manipulation and analysis.
* pathlib from the Python Standard Library: A module for working with file paths.
* confusion_matrix from sklearn.metrics: A function to compute the confusion matrix for model evaluation.
* classification_report from sklearn.metrics: A function to generate a classification report, including precision, recall, and F1-score.
* LogisticRegression from sklearn.linear_model: A class for logistic regression model implementation.
Make sure to have these libraries installed in your environment before running the analysis code.


## Project Workflow
The project follows the following workflow:
1) Splitting the Data into Training and Testing Sets: The lending_data.csv file from the Resources folder is read into a Pandas DataFrame. The "loan_status" column is used to create the labels set (y), and the remaining columns are used to create the features DataFrame (X). The data is then split into training and testing datasets using the train_test_split function.
2) Creating a Logistic Regression Model with the Original Data: The logistic regression model is implemented with the training data (X_train and y_train) using the logistic regression algorithm. The model is then used to predict the labels for the testing data (X_test). The performance of the model is evaluated by generating a confusion matrix and printing the classification report. Additionally, the model's ability to predict both healthy loans (0) and high-risk loans (1) is assessed.
3) Credit Risk Analysis Report
The project includes a Credit Risk Analysis Report file named "CreditRiskAnalysisReport.md". The report provides a summary and analysis of the performance of the logistic regression model. It follows the structure outlined in the provided report template and includes the following sections:
* Overview of the Analysis: This section explains the purpose of the analysis, which is to develop a logistic regression model for credit risk classification based on the provided dataset.
* Results: A bulleted list describes the accuracy score, precision score, and recall score of the logistic regression model. These metrics provide an understanding of the model's performance in predicting loan classifications.
* Summary: This section summarizes the results from the logistic regression model. It includes a justification for recommending the model for use by the lending company. If the model is not recommended, the reasoning behind this decision is also provided.

## Repository Structure
* The Credit_Risk_Classification_Analysis.ipynb file contains the code for the analysis, including data preprocessing, model creation, and evaluation.
* The CreditRiskAnalysisReport.md file houses the Credit Risk Analysis Report, which provides a summary and analysis of the model's performance.
* The Resources folder contains the lending_data.csv file, which is the dataset used for the analysis.


## Conclusion
In conclusion, the logistic regression model developed for credit risk classification exhibited a strong performance in predicting loan statuses. The model leveraged the provided dataset and utilized various features to distinguish between healthy and high-risk loans. The Credit Risk Analysis Report, which provides a comprehensive summary and analysis of the model's performance, can be accessed [here](https://github.com/Akif23Hasan/Module_20_Challenge-Credit_Risk_Classification/blob/main/CreditRiskAnalysisReport.md). It includes an overview of the analysis, detailed results, and a justification for recommending the model for use by the lending company.

For further details and insights, please refer to the Credit Risk Analysis Report linked above.
