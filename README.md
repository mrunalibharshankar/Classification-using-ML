# Classification of TikTok videos using XGBoost Binary logistic and Random forest tree based Model

Build a machine learning model that can be used to determine whether a video contains a claim or whether it offers an opinion. With a successful prediction model, TikTok can reduce the backlog of user reports and prioritize them more efficiently.

# Overview
## Select an evaluation metric
To determine which evaluation metric might be best, consider how the model might be wrong. There are two possibilities for bad predictions:
False positives: When the model predicts a video is a claim when in fact it is an opinion
False negatives: When the model predicts a video is an opinion when in fact it is a claim

## Ethical Implication
In the given scenario, it's better for the model to predict false positives when it makes a mistake, and worse for it to predict false negatives. It's very important to identify videos that break the terms of service, even if that means some opinion videos are misclassified as claims. The worst case for an opinion misclassified as a claim is that the video goes to human review. The worst case for a claim that's misclassified as an opinion is that the video does not get reviewed and it violates the terms of service. A video that violates the terms of service would be considered posted from a "banned" author, as referenced in the data dictionary.

Because it's more important to minimize false negatives, the model evaluation metric will be recall.

## Modeling workflow and model selection process
1. Split the data into train/validation/test sets (60/20/20)
2. Fit models and tune hyperparameters on the training set
3. Perform final model selection on the validation set
4. Assess the champion model's performance on the test set

## Dataset 
The data file has 12 columns with names:
1. number(#)
2. claim_status
3. video_id 
4. video_duration_sec
5. video_transcription_text 
6. verified_status  
7. author_ban_status     
8. video_view_count          
9. video_like_count
10. video_share_count
11. video_download_count 
12. video_comment_count  

The original files were exported from the [Kaggle](https://www.kaggle.com/datasets/yakhyojon/tiktok), and is available in this repository as an [.CSV file](https://github.com/mrunalibharshankar/Classification-using-ML/blob/33a081569f338ea40466af7aabdb4c4e5ddb76ad/tiktok_dataset.csv) document.


## Importing relevant libraries and packages
We have used Jupiter Notebook of Anaconda to evaluate and build the model in python. Started off with importing relevant libraries and packages:
1. For data manipulation: Numpy and Pandas
2. For data visualization: Matplotlib.pyplot and Seaborn
3. For data modeling: XGboost(XGBClassifier, XGBRegressor, plot_importance),sklearn.ensemble(RandomForestClassifier), sklearn.feature_extraction.text(CountVectorizer)
4. For metrics and helpful functions: sklearn.model_selection(GridSearchCV, train_test_split), sklearn.metrics(accuracy_score, precision_score, recall_score,roc_auc_score, roc_curve, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report)
5. For saving models: pickle

The data is looking like this,
![Alt Text](https://github.com/mrunalibharshankar/Classification-using-ML/blob/99494574f8239b83aea1bee4e9dc939c4a889027/Data.png)

## Data Exploration(Initial EDA, data cleaning and data visualization)
Examining and visualizing data to understand its characteristics, uncover patterns, and identify potential insights.
1. Basic Info
2. Descriptive Statistic
3. Missing Values
4. Univariate Analysis
5. Bivariate Analysis

## Feature Engineering


![Alt Text](https://github.com/mrunalibharshankar/Classification-using-ML/blob/99494574f8239b83aea1bee4e9dc939c4a889027/FE_vectorisation.png)

## Data Visualization of Categorical and Numerical data

![Alt Text](https://github.com/mrunalibharshankar/Classification-using-ML/blob/99494574f8239b83aea1bee4e9dc939c4a889027/Viz_after_conversion.png)

## Splitting the data into Train,Validation and Test



## Built Random Forest GridSearch CV Model and Classification Report

Hypertune Parameters = 'max_depth': [5, 7,None], 'max_features': [0.3, 0.6], 'max_samples': [0.7], 'min_samples_leaf': [1,2], 'min_samples_split': [2,3],'n_estimators': [75,100,200]
             

![Alt Text](https://github.com/mrunalibharshankar/Classification-using-ML/blob/99494574f8239b83aea1bee4e9dc939c4a889027/RF_Val.png)


On Test Data: 
![Alt Text](https://github.com/mrunalibharshankar/Classification-using-ML/blob/99494574f8239b83aea1bee4e9dc939c4a889027/RF_test.png)


## Built XGBoost GridSearch CV and Classification Report

Hypertune Parameters = 'max_depth': [4,8,12], 'min_child_weight': [3, 5], 'learning_rate': [0.01, 0.1], 'n_estimators': [300, 500]

![Alt Text](https://github.com/mrunalibharshankar/Classification-using-ML/blob/99494574f8239b83aea1bee4e9dc939c4a889027/XGB_Val.png)
             

On Test data:
![Alt Text](https://github.com/mrunalibharshankar/Classification-using-ML/blob/99494574f8239b83aea1bee4e9dc939c4a889027/XGB_test.png)


## Final Results of XGBoost and Random Forest Tree based Model

The results of the XGBoost model were also nearly perfect. However, its errors tended to be false negatives. Identifying claims was the priority, so it's important that the model be good at capturing all actual claim videos. The random forest model has a better recall score, and is therefore the champion model.

![Alt Text](https://github.com/mrunalibharshankar/Classification-using-ML/blob/99494574f8239b83aea1bee4e9dc939c4a889027/Results.png)

## Random forest feature importance
![Alt Text](https://github.com/mrunalibharshankar/Classification-using-ML/blob/99494574f8239b83aea1bee4e9dc939c4a889027/FI.png)

The plot above shows that in this random forest model, last_evaluation, number_project, tenure, and overworked have the highest importance, in that order. These variables are most helpful in predicting the outcome variable, left, and they are the same as the ones used by the decision tree model.



The barplot above shows that in this decision tree model,satisfaction_level,tenure, last_evaluation, number_project, and average_monthly_hours have the highest importance, in that order. These variables are most helpful in predicting the outcome variable, left.

# Conclusion and Summary

1. Logistic Regression

The logistic regression model achieved precision of 80%, recall of 83%, f1-score of 80% (all weighted averages), and accuracy of 83%, on the test set.

2. Tree-based Machine Learning

The decision tree model achieved AUC of 95.65%, precision of 97.59%, recall of 90.4%, f1-score of 94.58%, and accuracy of 98.23%, on the test set. The random forest modestly outperformed the decision tree model.

**To retain employees, the following recommendations could be presented to the stakeholders:**
- Cap the number of projects that employees can work on.
- Consider promoting employees who have been with the company for atleast four years, or conduct further investigation about why four-year tenured employees are so dissatisfied.
- Either reward employees for working longer hours, or don't require them to do so.
- If employees aren't familiar with the company's overtime pay policies, inform them about this. If the expectations around workload and time off aren't explicit, make them clear.
- Hold company-wide and within-team discussions to understand and address the company work culture, across the board and in specific contexts.
- High evaluation scores should not be reserved for employees who work 200+ hours per month. Consider a proportionate scale for rewarding employees who contribute more/put in more 
  effort.


  















