from sklearn.metrics import classification_report


def generate_classification_report_from_predictions(predictions, testY, labelNames):
    """
    predictions: softmax output, an array of probability arrays

    testY: label-binarized output, an array of probability arrays (with entries only 0 or 1)

    labelNames: array of class names
    """
    report = classification_report(testY.argmax(axis=1),
                                   predictions.argmax(axis=1),
                                   target_names=labelNames)
    return report
