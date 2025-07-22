from sklearn.metrics import classification_report, confusion_matrix
import json
import logging

logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and generate metrics"""
    logger.info("Evaluating model performance")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Generate metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    metrics = {
        "accuracy": report["accuracy"],
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1_score": report["weighted avg"]["f1-score"],
        "confusion_matrix": cm
    }
    
    return metrics