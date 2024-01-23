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
The data file has 10 columns with names:
1. claim_status
2. video_id 
3. video_duration_sec
4. video_transcription_text 
5. verified_status  
6. author_ban_status     
7. video_view_count          
8. video_like_count
9. video_share_count
10. video_download_count 
11. video_comment_count  

The original files were exported from the [Kaggle](https://www.kaggle.com/datasets/mrunalibharshankar/hr-employee-retention#:~:text=HR_capstone_dataset.-,csv,-Summary), and is available in this repository as an [.CSV file](https://github.com/mrunalibharshankar/Python/blob/98905f38ef3704a651371c66b1cb6c6f71452c46/HR_capstone_dataset.csv) document.


## Importing relevant libraries and packages
We have used Jupiter Notebook of Anaconda to evaluate and build the model in python. Started off with importing relevant libraries and packages:
1. For data manipulation: Numpy and Pandas
2. For data visualization: Matplotlib.pyplot and Seaborn
3. For data modeling: XGboost(XGBClassifier, XGBRegressor, plot_importance), sklearn.linear_model(LogisticRegression), sklearn.tree(DecisionTreeClassifier), sklearn.ensemble(RandomForestClassifier)
4. For metrics and helpful functions: sklearn.model_selection(GridSearchCV, train_test_split), sklearn.metrics(accuracy_score, precision_score, recall_score,roc_auc_score, roc_curve, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report), sklearn.tree(plot_tree)
5. For saving models: pickle

The data is looking like this,
![Alt Text](https://github.com/mrunalibharshankar/Python/blob/bb2702859046781e845ed7bdbb66d134d9039946/Dataset.png)

## Data Exploration(Initial EDA, data cleaning and data visualization)
Examining and visualizing data to understand its characteristics, uncover patterns, and identify potential insights.
1. Basic Info
2. Descriptive Statistic
3. Missing Values
4. Univariate Analysis
5. Bivariate Analysis

## Correlations between variables
The correlation heatmap confirms that the number of projects, monthly hours, and evaluation scores and to some extend tenure all have some positive correlation with each other, and whether an employee leaves is negatively correlated with their satisfaction level.

![Alt Text](https://github.com/mrunalibharshankar/Python/blob/e79d7fbc299d8b9bab2fe9dc5cdf4eb9ad84045b/Heatmap.png)

## Logistic Regression model and Classification Report
Goal is to predict whether an employee leaves the company, which is a categorical outcome variable. So this task involves classification. More specifically, this involves binary classification, since the outcome variable left can be either 1 (indicating employee left) or 0 (indicating employee didn't leave).


![Alt Text](https://github.com/mrunalibharshankar/Python/blob/e79d7fbc299d8b9bab2fe9dc5cdf4eb9ad84045b/LR.png)

## Cross Validation with k-fold

![Alt Text](https://github.com/mrunalibharshankar/Python/blob/e79d7fbc299d8b9bab2fe9dc5cdf4eb9ad84045b/LR_CV.png)

The classification report above shows that the logistic regression cv model achieved a precision of 47%, recall of 25%, f1-score of 32% (all weighted averages), and accuracy of 89%. However, if it's most important to predict employees who leave, then the scores are significantly lower.

## Recall evaluation metrics
- AUC is the area under the ROC curve; it's also considered the probability that the model ranks a random positive example more highly than a random negative example.
- Precision measures the proportion of data points predicted as True that are actually True, in other words, the proportion of positive predictions that are true positives.
- Recall measures the proportion of data points that are predicted as True, out of all the data points that are actually True. In other words, it measures the proportion of positives that are correctly classified.
- Accuracy measures the proportion of data points that are correctly classified.
- F1-score is an aggregation of precision and recall.

## Build Decision Tree Model and Classification Report

![Alt Text](https://github.com/mrunalibharshankar/Python/blob/e79d7fbc299d8b9bab2fe9dc5cdf4eb9ad84045b/DT.png)

## Cross-validated with grid-search

![Alt Text]()

## Built Random Forest Model and Classification Report

![Alt Text](https://github.com/mrunalibharshankar/Python/blob/e79d7fbc299d8b9bab2fe9dc5cdf4eb9ad84045b/RF.png)

## Cross-validated with grid-search

![Alt Text]()


## Final Results of Regression and Tree Based Models

The evaluation scores of the random forest model are slightly better than those of the decision tree model, with the exception of recall (the recall score of the random forest model is approximately 0.001 lower, which is a negligible amount). This indicates that the random forest model mostly outperforms the decision tree model.

![Alt Text](https://github.com/mrunalibharshankar/Python/blob/e79d7fbc299d8b9bab2fe9dc5cdf4eb9ad84045b/Result_tb.png)

### Random forest feature importance
![Alt Text](https://github.com/mrunalibharshankar/Python/blob/e79d7fbc299d8b9bab2fe9dc5cdf4eb9ad84045b/RF_FI.png)

The plot above shows that in this random forest model, last_evaluation, number_project, tenure, and overworked have the highest importance, in that order. These variables are most helpful in predicting the outcome variable, left, and they are the same as the ones used by the decision tree model.

### Decision Tree feature importance

![Alt Text](https://github.com/mrunalibharshankar/Python/blob/e79d7fbc299d8b9bab2fe9dc5cdf4eb9ad84045b/DT_Feature_I.png)

The barplot above shows that in this decision tree model,satisfaction_level,tenure, last_evaluation, number_project, and average_monthly_hours have the highest importance, in that order. These variables are most helpful in predicting the outcome variable, left.

## Conclusion and Summary

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


  















