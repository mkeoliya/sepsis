import shap

def treeShap(tree_model, X_test):
    """
    Function that produces a shap explanation given a random forest model

    Parameters
    - tree_model: an sklearn random forest tree model
    - X_test: test dataset
    """

    explainer = shap.TreeExplainer(tree_model)

    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values[0], X_test)

    shap.dependence_plot(0, shap_values[0], X_test)

def regressionShap(regression_model, X_test):
    """
    Function that produces a shap explanation given a random forest model

    Parameters
    - tree_model: an sklearn random forest tree model
    - X_test: test dataset
    """

    explainer = shap.TreeExplainer(regression_model)

    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values[0], X_test)

    shap.dependence_plot(0, shap_values[0], X_test)

