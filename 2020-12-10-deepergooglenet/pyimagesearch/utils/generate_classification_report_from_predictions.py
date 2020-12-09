from sklearn.metrics import classification_report


def generate_classification_report_from_predictions(predictions, testY, labelNames):
    report = classification_report(testY.argmax(axis=1),
                                   predictions.argmax(axis=1),
                                   target_names=labelNames)
    return report
