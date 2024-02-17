# Kidney Stone Treatment Success Analysis

## Overview
This project involves the analysis of kidney stone treatment success rates using logistic regression models. The analysis aims to understand the factors that contribute to the success of kidney stone treatments and to predict the success of future treatments based on patient data.

## Data
The dataset `kidney_stone_data.csv` includes information on kidney stone treatments, patient stone sizes, and the success of the treatments.

## Analysis Approach
The analysis follows these steps:

1. **Data Preparation**
   - The dataset is loaded and inspected for structure and missing values.
   - Summary statistics are generated to understand the distribution of the data.
   - The dataset is split into training and testing sets for model validation.

2. **Exploratory Data Analysis**
   - The number and frequency of treatment success and failure are calculated and visualized.
   - Bar plots are created to visualize the proportion of success by treatment type and stratified by stone size.

3. **Feature Selection**
   - Relevant features (`stone_size` and `treatment`) are selected for modeling.
   - Categorical variables are converted to numerical using one-hot encoding.

4. **Model Building**
   - Multiple logistic regression models are built to predict treatment success.
   - The glmnet package is used to build Lasso regression models to enhance feature selection.

5. **Model Evaluation**
   - Predictions are made on the test dataset.
   - Confusion matrices are generated to evaluate model performance.
   - Receiver Operating Characteristic (ROC) curves and Area Under the Curve (AUC) metrics are calculated to assess model accuracy.
   - Accuracy scores are reported for logistic and Lasso regression models.

## Results
The models' performance metrics, including confusion matrices and accuracy scores, are reported. The Lasso regression model's accuracy is also provided for comparison.

## Repository Contents
- `kidney_stone_data.csv`: The dataset used for analysis.
- R scripts containing the analysis workflow.

## Requirements
The analysis requires the following R packages:
- readr
- dplyr
- ggplot2
- broom
- glmnet
- pROC
- caret

## Usage
Set the working directory to the location of the `kidney_stone_data.csv` file before running the script. The script will load the data, perform the analysis, and output the results to the R console.

## Acknowledgements
This analysis is part of a study on the success rates of kidney stone treatments. Data confidentiality is maintained, and no personal patient information is disclosed.
