from sklearn import linear_model, preprocessing
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np

def F(beta, precision, recall):
    
    """
    Function that calculate f1, f2, and f0.5 scores.
    
    @params: beta, Float, type of f score
             precision: Float, average precision
             recall: Float, average recall
    
    @return: Float, f scores
    """
    
    return (beta*beta + 1)*precision*recall / (beta*beta*precision + recall)

def train_linear_regression_model(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    """
    Trains linear regression model for sepsis prediction
    Returns model and model metrics in a dictionary format 

    Parameters
    - X_train: training dataset
    - y_train: training labels
    - X_test: testing dataset
    - y_test: testing labels
    """

    # define scaler for data
    scalar = preprocessing.StandardScaler()

    # fit and transform data
    scalar.fit(X_train)
    X_train = scalar.transform(X_train)
    X_test = scalar.transform(X_test)

    # Initialize linear regression model
    linear_regression_model = linear_model

    # fit model
    linear_regression_model.fit(X_train, y_train)

    # compute and return metrics
    y_pred = linear_regression_model.predict(X_test)
    y_pred_proba = linear_regression_model.predict_proba(X_test)[::,1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    auc_test = auc(recall, precision)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auroc = auc(fpr, tpr)

    return {'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred), 
            'auprc': auc_test,
            'auc': auroc,
            'accuracy': accuracy_score,
            'model': linear_regression_model}