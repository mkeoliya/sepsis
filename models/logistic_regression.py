from sklearn import linear_model, preprocessing
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
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

def train_logistic_regression_model(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    """
    Trains logistic regression model for sepsis prediction
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

    # hyperparameter tuning to retrieve best model given dataset
    logistic_regression_model = linear_model.LogisticRegression(solver='saga', max_iter=1000)

    # create hyperparameter search space
    # create regularization penalty space
    penalty = ['l1', 'l2']

    # create regularization hyperparameter space
    C = np.logspace(-1, 4, 10)

    # create hyperparameter options
    hyperparameters = dict(C=C, penalty=penalty)

    # create grid search using 5-fold cross validation
    clf = GridSearchCV(logistic_regression_model, hyperparameters, cv=5, verbose=0)    

    best_model = clf.fit(X_train, y_train)

    # predictions of best model
    y_predict = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[::,1]

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    return {
        'f1': F(1, np.mean(precision), np.mean(recall)),
        'precision': precision,
        'recall': recall,
        'auprc': auc(recall, precision),
        'auc': auc(fpr, tpr),
        'accuracy': accuracy_score(y_test, y_predict),
        'model': best_model
    }
