from lime.lime_tabular import LimeTabularExplainer

def LIMEExplanations(model, X_train, y_train, features, classes, instance):
    """
    Provide LIMEExplanations for a specific test case give model information

    Parameters
    - model: model used to provide explanation
    - X_train: training dataset
    - y_train: training labels
    - features: training features
    - classes: names of the target classes
    - instance: particular test case to provide explanation for
    """

    explainer = LimeTabularExplainer(
        training_data=X_train,
        training_labels=y_train,
        feature_names=features,
        class_names=classes,
        mode='classification'
    )

    explanation = explainer.explain_instance(
        data_row=instance,
        predict_fn=model.predict_proba,
    )

    return explanation