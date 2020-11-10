from matplotlib.pyplot import axis, plot
from pyimagesearch.nn.conv import LeNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

# since LeNet input shape must be a list of np.array of shape (height, width, depth),
# we must reshape the row vectors in dataset:

if K.image_data_format() == "channel_first":
    trainData = trainData.reshape(-1, 1, 28, 28)
    testData = testData.reshape(-1, 1, 28, 28)
else:
    trainData = trainData.reshape(-1, 28, 28, 1)
    testData = testData.reshape(-1, 28, 28, 1)

trainData = trainData.astype("float32")/255.0
testData = testData.astype("float32")/255.0

labelBinarizer = LabelBinarizer()
trainLabels = labelBinarizer.fit_transform(trainLabels)
testLabels = labelBinarizer.transform(testLabels)

opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, numOfClasses=10)
model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])

print("[INFO] training ...")

H = model.fit(trainData, trainLabels, validation_data=(
    testData, testLabels), batch_size=128, epochs=20, verbose=1)

predictions = model.predict(testData, batch_size=128)

print(
    classification_report(
        testLabels.argmax(axis=1),
        predictions.argmax(axis=1),
        target_names=[str(x) for x in labelBinarizer.classes_]
    )
)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="validation_loss")
plt.plot(np.arange(0, 20), H.history["accuracy"], label="train_loss")
plt.plot(np.arange(0, 20),
         H.history["val_accuracy"], label="validation_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch Number")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
