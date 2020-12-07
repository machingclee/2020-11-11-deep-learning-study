from sklearn.metrics import classification_report


def generate_classification_report(model, testX, testY, batch_size, labelNames):
    predictions = model.predict(testX, batch_size=batch_size)
    report = classification_report(testY.argmax(axis=1),
                                   predictions.argmax(axis=1),
                                   target_names=labelNames)
    return report
