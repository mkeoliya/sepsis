from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, auc, precision_recall_curve, roc_curve
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV
import pandas as pd


def train_rf_model(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame):
    """
    Trains random forest model for sepsis prediction
    Returns model and model metrics in a dictionary format 

    Parameters
    - X_train: training dataset
    - y_train: training labels
    - X_test: testing dataset
    - y_test: testing labels
    """

    # initialize model
    model = RandomForestClassifier(random_state=1211, oob_score=True, max_depth=10, n_estimators=150, class_weight='balanced_subsample')

    # parameters
    param_grid = {
        'n_estimators': [25, 50, 75, 100, 125, 150, 175, 200],
        'max_depth': [None, 5, 10, 15, 20, 25, 30, 35, 40, 50],
        'max_features': [None, 'sqrt', 'log2'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, verbose=0, n_jobs=-1)

    # train model
    rf_model = grid_search.fit(X_train,y_train)

    # testing and analysis of trained model
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[::,1]
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
            'model': rf_model}


