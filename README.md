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
Split the training set into training and validation sets, 75/25, to result in a final ratio of 60/20/20 for train/validate/test sets.
The number of features (11) aligns between the training and testing sets.
The number of rows aligns between the features and the outcome variable for training (11,450) and both validation and testing data (3,817).

## Built Random Forest GridSearch CV Model and Classification Report

Hypertune Parameters = 'max_depth': [5, 7,None], 'max_features': [0.3, 0.6], 'max_samples': [0.7], 'min_samples_leaf': [1,2], 'min_samples_split': [2,3],'n_estimators': [75,100,200]
             
On Training Data:
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

The plot above shows that in this random forest model, video_view_count highest importance followed by video_like_count, video_shared_count.

# Conclusion and Summary
- One can recommend this model because it performed well on both the validation and test holdout data. Furthermore, both precision and F1 scores were consistently high. The model very successfully classified claims and opinions.

- The model's most predictive features were all related to the user engagement levels associated with each video. It was classifying videos based on how many views, likes, shares, and downloads they received.

- The model currently performs nearly perfectly, there is no need to engineer any new features.

- The current version of the model does not need any new features. However, it would be helpful to have the number of times the video was reported. It would also be useful to have the total number of user reports for all videos posted by each author.


  















