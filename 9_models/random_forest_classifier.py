import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, roc_curve, auc


input_data = r'D:\ids_for_cloud_env\8_pre_processing\normalized_clean_output_1sec.csv'

# Load the CSV data
data = pd.read_csv(input_data)

# Split the data into features and labels
X = data.drop(['batch-number', 'is-malicious'], axis=1)  # features
y = data['is-malicious']  # labels


precision_scores = []
recall_scores = []
auc_scores = []
accuracy_scores = []
y_true_list = []
y_pred_list = []
y_pred_prob_list = []

# Initialize the Random Forest Classifier with default parameters
rfc = RandomForestClassifier()

# perform cross validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=None)

# Perform 5-fold cross-validation and compute the evaluation metrics
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the Random Forest Classifier on the training data
    rfc.fit(X_train, y_train)

    # Predict the labels of the testing data
    y_pred = rfc.predict(X_test)

    # Compute the evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred)


    y_pred_prob = rfc.predict_proba(X_test)[:, 1]

    y_true_list.extend(y_test)
    y_pred_list.extend(y_pred)
    y_pred_prob_list.extend(y_pred_prob)

    print('====================Individual Report==========================')
    print(f'Test precision is : {precision:.2f}')
    print(f'Test recall is : {recall:.2f}')
    print(f'Test auc is : {auc_score:.2f}')
    print(f'Test accuracy is : {accuracy:.2f}')
    print('====================end========================================')

    # Append the scores to the lists
    precision_scores.append(precision)
    recall_scores.append(recall)
    auc_scores.append(auc_score)
    accuracy_scores.append(accuracy)

y_true = np.array(y_true_list)
y_pred = np.array(y_pred_list)
y_pred_prob = np.array(y_pred_prob_list)
# print(classification_report(y_true, y_pred))


print('\n\n******************************** Collective Report ********************************************\n')
print(f'Average precision across {n_splits} folds: {np.mean(precision_scores):.2f}')
print(f'Average recall across {n_splits} folds: {np.mean(recall_scores):.2f}')
print(f'Average auc_score across {n_splits} folds: {np.mean(auc_scores):.2f}')
print(f'Average accuracy across {n_splits} folds: {np.mean(accuracy_scores):.2f}')
print('\n*********************************** End ********************************************************\n')

# # predicted probabilities of positive class
# y_pred_prob = rfc.predict_proba(X_test)[:, 1]

# calculate false positive rate and true positive rate for different probability thresholds
fpr, tpr, thresholds= roc_curve(y_true, y_pred_prob)

# calculate area under the curve
roc_auc = auc(fpr, tpr)

# plot ROC curve
plt.plot(fpr, tpr, color='Blue', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve\nRandom Forest Classifier')
plt.legend(loc="lower right")
plt.show()